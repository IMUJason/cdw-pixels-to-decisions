"""统一比较框架：no-vision / vision+MSE / SAA(K) / oracle × 多种子（含中位数）。

协议与 run_composition_dfl_vision 相同（原型工地、两阶段评估）；
SAA 的 K 个采样取自训练分布（每工地按原型簇重采，不用测试场景信息——
这正是 SAA 与预测型方法的本质区别：分布对冲 vs 场景定制）。

用法: python scripts/run_baselines_compare.py --seeds 11 22 33 --n-test 30
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel, MATERIALS  # noqa: E402
from src.models.composition_dfl import train_predictor, predict_supply, supply_dict  # noqa: E402
from src.models.saa_baseline import solve_saa_y  # noqa: E402
from run_composition_dfl import make_scenario, solve_y  # noqa: E402
from run_composition_dfl_vision import load_split  # noqa: E402

N_SITES, N_IMGS, PURITY, D_SCALE, K_SAA = 5, 8, 0.7, 2000.0, 5


def sample_site(P, G, clusters, rng):
    cl = clusters[rng.integers(0, 4)]
    n_in = int(PURITY * N_IMGS)
    idx = np.concatenate([rng.choice(cl, n_in, replace=True),
                          rng.choice(len(P), N_IMGS - n_in, replace=True)])
    gm = G[idx].sum(0) * D_SCALE
    return gm / max(gm.sum(), 1e-9), gm.sum(), P[idx] / (P[idx].sum(1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--out", default=r"D:\2026 cdw 0911 datasets\processed\baselines_compare.json")
    args = ap.parse_args()

    P_tr, G_tr, _, LAB_tr = load_split("train")
    P_te, G_te, _, LAB_te = load_split("test")
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    cl_te = [np.where(LAB_te == c)[0] for c in range(4)]

    # 训练预测器（种子 0 固定；场景种子变化）
    rng0 = np.random.default_rng(1)
    xs, ys = [], []
    for _ in range(500):
        th, D, feat = sample_site(P_tr, G_tr, cl_tr, rng0)
        xs.append(feat); ys.append(D * th)
    model = train_predictor(np.array(xs), np.array(ys), "mse", epochs=400, seed=0)
    mean_mix = np.array(ys).mean(0); mean_mix = mean_mix / mean_mix.sum()
    mean_tot = float(np.array(ys).sum(1).mean())

    methods = ["no-vision", "vision+MSE", f"SAA(K={K_SAA})", "oracle"]
    all_regs = {m: [] for m in methods}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for k in range(args.n_test):
            theta, D, feats = [], [], []
            for s in range(N_SITES):
                th, d, feat = sample_site(P_te, G_te, cl_te, rng)
                theta.append(th); D.append(d); feats.append(feat)
            theta = np.stack(theta); D = np.array(D)
            kwargs, S, T, _, _ = make_scenario(rng, np.eye(5), np.ones(1),
                                               n_sites=N_SITES, cap_factor=0.6,
                                               sites_override=(theta, D))
            true_sup = supply_dict(theta, D, S, T)
            _, z_star = solve_y(kwargs, true_sup)

            def regret(y_hat):
                rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                          lp_relaxation=True, **kwargs).solve(time_limit=30)
                return rec["objective"] - z_star if rec else np.nan

            y_o, _ = solve_y(kwargs, true_sup)
            all_regs["oracle"].append(regret(y_o))
            th_n = np.tile(mean_mix, (N_SITES, 1))
            sup_n = supply_dict(th_n, np.full(N_SITES, mean_tot / N_SITES), S, T)
            y_n, _ = solve_y(kwargs, sup_n)
            all_regs["no-vision"].append(regret(y_n))
            pred = np.clip(predict_supply(model, np.array(feats)), 1e-6, None)
            sup_v = supply_dict(pred / pred.sum(1, keepdims=True), pred.sum(1), S, T)
            y_v, _ = solve_y(kwargs, sup_v)
            all_regs["vision+MSE"].append(regret(y_v))
            # SAA：K 个训练分布采样（无测试信息）
            draws = []
            r2 = np.random.default_rng(seed * 1000 + k)
            for kk in range(K_SAA):
                th_k, D_k = [], []
                for s in range(N_SITES):
                    t2, d2, _ = sample_site(P_tr, G_tr, cl_tr, r2)
                    th_k.append(t2); D_k.append(d2)
                draws.append(supply_dict(np.stack(th_k), np.array(D_k), S, T))
            y_saa = solve_saa_y(kwargs, draws, time_limit=30)
            all_regs[f"SAA(K={K_SAA})"].append(
                regret({f: 1.0 if v > .5 else 0.0 for f, v in y_saa.items()}) if y_saa else np.nan)
        print(f"seed {seed} done", flush=True)

    print(f"\n=== 统一比较（{len(args.seeds)} 种子 × {args.n_test} 场景，含中位数）===")
    print(f"{'method':16s} {'mean':>10s} {'median':>10s} {'p90':>10s} {'nan':>4s}")
    summary = {}
    for mth, regs in all_regs.items():
        r = np.array(regs, dtype=float)
        summary[mth] = {"mean": float(np.nanmean(r)), "median": float(np.nanmedian(r)),
                        "p90": float(np.nanquantile(r, 0.9)), "nan": int(np.isnan(r).sum())}
        s = summary[mth]
        print(f"{mth:16s} {s['mean']:10.1f} {s['median']:10.1f} {s['p90']:10.1f} {s['nan']:4d}")
    json.dump({"config": {"seeds": args.seeds, "n_test": args.n_test, "purity": PURITY,
                          "n_imgs": N_IMGS, "k_saa": K_SAA},
               "summary": summary, "raw": {k: [None if np.isnan(x) else float(x)
                                               for x in v] for k, v in all_regs.items()}},
              open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
