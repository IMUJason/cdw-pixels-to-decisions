"""E3 首个 DFL 实验：组成预测 × 多物料 LIRP 两阶段遗憾对比。

方法列：no-vision（训练集边际均值）、MLP+MSE、MLP+Pinball(τ=0.5/0.9/0.95)、oracle。
τ 由验证集遗憾选定（决策感知分位数——论文1"内点最优分位数"发现在多物料场景的延伸）。

用法:
    python scripts/run_composition_dfl.py --n-train 100 --n-val 30 --n-test 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.cdw_lirp_multi import MATERIALS  # noqa: E402
from src.models.composition_dfl import (  # noqa: E402
    snapshot_features, train_predictor, predict_supply, supply_dict, two_stage_regret, solve_y)
from src.vision.site_scenario_bridge import load_thetas, to_group_space  # noqa: E402


def make_scenario(rng, theta_pool, D_pool, n_sites=5, d_scale=2000.0,
                  sites_override=None, cap_factor=0.6):
    """采样一个场景：拓扑/成本/兼容 + 各工地真值 (theta, D)。

    sites_override=(theta_sites, D_sites)：指定工地真值（需求/容量按其规模生成），
    供视觉特征版实验复用同一拓扑逻辑。"""
    n_f, n_d, periods, grid = 3, 6, 2, 50.0
    S = list(range(1, n_sites + 1))
    F = list(range(10, 10 + n_f))
    Dp = list(range(20, 20 + n_d))
    T = list(range(periods))
    coords = {n: (rng.uniform(0, grid), rng.uniform(0, grid)) for n in S + F + Dp}
    node_types = {**{s: "site" for s in S}, **{f: "facility" for f in F},
                  **{d: "demand" for d in Dp}}
    if sites_override is not None:
        theta_sites, D_sites = sites_override
    else:
        theta_sites = theta_pool[rng.choice(len(theta_pool), n_sites, replace=True)]
        D_sites = D_pool[rng.choice(len(D_pool), n_sites, replace=True)] * d_scale
    demand = {(d, mm, t): float(rng.uniform(0.5, 2.5) * D_sites.mean() / n_d / periods)
              for d in Dp for mm in MATERIALS for t in T}
    compat = {}
    for f in F:
        groups = {"mineral"}
        while len(groups) < int(rng.integers(2, 4)):
            groups.add(str(MATERIALS[int(rng.integers(0, len(MATERIALS)))]))
        for mm in MATERIALS:
            compat[(f, mm)] = 1.0 if mm in groups else 0.0
    kwargs = dict(coords=coords, node_types=node_types, demand=demand,
                  capacity={f: float(cap_factor * D_sites.sum() / n_f) for f in F},
                  opening_cost={f: float(rng.uniform(4000, 6000)) for f in F},
                  holding_cost={f: 2.0 for f in F}, compat=compat,
                  proc_cost={mm: 5.0 for mm in MATERIALS},
                  revenue={"mineral": 12.0, "wood": 15.0, "gypsum": 6.0,
                           "plastic_mix": 10.0, "general": 2.0},
                  vehicle_capacity=100.0, penalty_cost=10000.0,
                  time_periods=T)
    return kwargs, S, T, theta_sites, D_sites


def eval_method(name, scenarios, get_pred_supply):
    regs = []
    zstars = {}
    from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel
    for k, (kwargs, S, T, theta, D) in enumerate(scenarios):
        true_sup = supply_dict(theta, D, S, T)
        if k not in zstars:
            _, z = solve_y(kwargs, true_sup)
            zstars[k] = z
        pred_sup = get_pred_supply(k, kwargs, S, T, theta, D)
        y_hat, _ = solve_y(kwargs, pred_sup)
        if y_hat is None or zstars[k] is None:
            regs.append(np.nan); continue
        mdl_rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                      lp_relaxation=True, **kwargs)
        sol_rec = mdl_rec.solve(time_limit=30.0)
        regs.append(sol_rec["objective"] - zstars[k] if sol_rec else np.nan)
    regs = np.array(regs, dtype=float)
    return {"method": name, "mean_regret": float(np.nanmean(regs)),
            "median": float(np.nanmedian(regs)), "worst": float(np.nanmax(regs)),
            "nan": int(np.isnan(regs).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-csv", default=r"D:\2026 cdw 0911 datasets\processed\composition_testing_trial.csv")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-snap", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"D:\2026 cdw 0911 datasets\processed\dfl_e3_trial.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    thetas, D = load_thetas(args.comp_csv)
    theta_pool = to_group_space(thetas)          # [n_img, 5]

    def gen(n):
        return [make_scenario(rng, theta_pool, D) for _ in range(n)]

    train_sc, val_sc, test_sc = gen(args.n_train), gen(args.n_val), gen(args.n_test)

    # 站点级训练数据（池化所有场景的所有工地）
    def build_xy(scenarios, seed):
        r = np.random.default_rng(seed)
        xs, ys = [], []
        for _, S, T, theta, Ds in scenarios:
            for si in range(len(S)):
                sup_vec = Ds[si] * theta[si]
                xs.append(snapshot_features(sup_vec, args.n_snap, args.sigma, r))
                ys.append(sup_vec)
        return np.array(xs), np.array(ys)

    Xtr, Ytr = build_xy(train_sc, args.seed + 1)
    print(f"train sites: {len(Xtr)}, feature dim: {Xtr.shape[1:]}")

    models = {"mse": train_predictor(Xtr, Ytr, "mse", epochs=300, seed=args.seed)}
    for tau in (0.5, 0.9, 0.95):
        models[f"pinball_{tau}"] = train_predictor(Xtr, Ytr, "pinball", tau=tau,
                                                    epochs=300, seed=args.seed)
    models["hybrid_0.9"] = train_predictor(Xtr, Ytr, "hybrid", tau=0.9,
                                            epochs=300, seed=args.seed)

    mean_mix = Ytr.mean(axis=0) / Ytr.mean(axis=0).sum()
    mean_tot = float(Ytr.sum(axis=1).mean())

    def method_pred(model_or_none, kind):
        def f(k, kwargs, S, T, theta, D):
            if kind == "oracle":
                return supply_dict(theta, D, S, T)
            if kind == "novision":
                th = np.tile(mean_mix, (len(S), 1))
                return supply_dict(th, np.full(len(S), mean_tot), S, T)
            snaps = []
            r = np.random.default_rng(1000 + k)
            for si in range(len(S)):
                snaps.append(snapshot_features(D[si] * theta[si], args.n_snap, args.sigma, r))
            pred = predict_supply(model_or_none, np.array(snaps))
            return supply_dict(pred / pred.sum(axis=1, keepdims=True),
                               pred.sum(axis=1), S, T)
        return f

    # 验证集选方法（MSE 与 hybrid 也参与）
    val_rows = []
    for name in ("mse", "pinball_0.5", "pinball_0.9", "pinball_0.95", "hybrid_0.9"):
        row = eval_method(name, val_sc, method_pred(models[name], "model"))
        val_rows.append(row)
        print("VAL", row)
    best = min(val_rows, key=lambda r: r["mean_regret"])["method"]
    print("best method on validation:", best)
    best_label = {"mse": "MLP+MSE", "hybrid_0.9": "MLP+Hybrid(τ=.9,pin-tot/mse-mix)"}.get(
        best, f"MLP+Pinball({best.split('_')[-1]})")

    # 测试集（避免与验证选出的方法重复打印）
    rows = [eval_method("no-vision(mean mix)", test_sc, method_pred(None, "novision")),
            eval_method("MLP+MSE", test_sc, method_pred(models["mse"], "model")),
            eval_method("MLP+Hybrid(τ=.9)", test_sc, method_pred(models["hybrid_0.9"], "model"))]
    if best not in ("mse", "hybrid_0.9"):
        rows.append(eval_method(best_label, test_sc, method_pred(models[best], "model")))
    rows.append(eval_method("oracle(true comp)", test_sc, method_pred(None, "oracle")))
    out = {"config": vars(args), "val": val_rows, "test": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== TEST regret (two-stage protocol) ===")
    print(f"{'method':24s} {'mean':>12s} {'median':>12s} {'worst':>12s}")
    for r in rows:
        print(f"{r['method']:24s} {r['mean_regret']:12.1f} {r['median']:12.1f} {r['worst']:12.1f}")
    print("->", args.out)


if __name__ == "__main__":
    main()
