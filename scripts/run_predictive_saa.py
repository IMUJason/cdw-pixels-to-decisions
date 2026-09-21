"""预测分布 SAA（SAA-pred）：以视觉预测为中心采样 K 个组成做 extensive-form 对冲。

协议统一版（2026-09-15 修正）：
- --seeds 多种子循环，与 run_baselines_compare.py 完全同协议（{11,22,33} × 30 场景）；
- 场景生成 rng 流与 baselines 完全一致（同种子 → 同场景序列，可直接对齐）；
- Dirichlet/lognormal 扰动改用**独立 rng 流**（旧版复用场景 rng 流，导致第 2 场景起与
  baselines 错位——协议不对称根源）；
- 输出 per-seed 与 pooled(90 场景) 统计 + mean 的 95% CI（t 分布）。

动机（统一比较表的结果）：SAA(训练分布) 均值好但尾部差，vision 点预测尾部极好但均值差
——两者互补。SAA-pred = 在视觉预测的置信邻域内对冲：θ_k ~ Dirichlet(α·θ̂)，D_k = D̂·LogNormal。

用法: python scripts/run_predictive_saa.py --seeds 11 22 33 --n-test 30 --k 5
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel  # noqa: E402
from src.models.composition_dfl import train_predictor, predict_supply, supply_dict  # noqa: E402
from src.models.saa_baseline import solve_saa_y  # noqa: E402
from run_composition_dfl import make_scenario, solve_y  # noqa: E402
from run_composition_dfl_vision import load_split  # noqa: E402
from run_baselines_compare import sample_site, N_SITES, N_IMGS, PURITY, D_SCALE  # noqa: E402

ALPHA, SIGMA_D = 100.0, 0.2  # Dirichlet 集中度（≈2-3% 组分噪声）与总量扰动


def t_ci95(x):
    """小样本 t 分布 95% CI（n 大时趋近 1.96σ/√n）；无 scipy 依赖的查表近似。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float(np.mean(x)), 0.0
    # t 临界值近似（df>=30 取 2.0 保守；n=90 时 1.987）
    t = 1.987 if n >= 60 else 2.023
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=r"D:\2026 cdw 0911 datasets\processed\predictive_saa_multi.json")
    args = ap.parse_args()

    P_tr, G_tr, _, LAB_tr = load_split("train")
    P_te, G_te, _, LAB_te = load_split("test")
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    cl_te = [np.where(LAB_te == c)[0] for c in range(4)]

    rng0 = np.random.default_rng(1)
    xs, ys = [], []
    for _ in range(500):
        th, D, feat = sample_site(P_tr, G_tr, cl_tr, rng0)
        xs.append(feat); ys.append(D * th)
    model = train_predictor(np.array(xs), np.array(ys), "mse", epochs=400, seed=0)

    methods = ["vision point", "SAA(train)", f"SAA-pred(a={ALPHA:.0f})", "oracle"]
    regs = {m: [] for m in methods}          # pooled over all seeds
    per_seed = {}
    for seed in args.seeds:
        seed_regs = {m: [] for m in methods}
        rng = np.random.default_rng(seed)    # 场景流：与 run_baselines_compare 完全一致
        rng_saa = np.random.default_rng(777)             # SAA(train) 采样流（独立）
        rng_pert = np.random.default_rng(10_000 + seed)  # SAA-pred 扰动流（独立，不污染场景流）
        for k in range(args.n_test):
            theta, D, feats = [], [], []
            for s in range(N_SITES):
                th, d, feat = sample_site(P_te, G_te, cl_te, rng)
                theta.append(th); D.append(d); feats.append(feat)
            theta = np.stack(theta); D = np.array(D)
            kwargs, S, T, _, _ = make_scenario(rng, np.eye(5), np.ones(1), n_sites=N_SITES,
                                               cap_factor=0.6, sites_override=(theta, D))
            true_sup = supply_dict(theta, D, S, T)
            _, z_star = solve_y(kwargs, true_sup)

            def regret(y_hat):
                rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                          lp_relaxation=True, **kwargs).solve(time_limit=30)
                return rec["objective"] - z_star if rec else np.nan

            pred = np.clip(predict_supply(model, np.array(feats)), 1e-6, None)
            theta_hat = pred / pred.sum(1, keepdims=True)
            D_hat = pred.sum(1)

            y_o, _ = solve_y(kwargs, true_sup)
            seed_regs["oracle"].append(regret(y_o))
            sup_v = supply_dict(theta_hat, D_hat, S, T)
            y_v, _ = solve_y(kwargs, sup_v)
            seed_regs["vision point"].append(regret(y_v))

            draws = []
            for kk in range(args.k):  # SAA(train)
                th_k, D_k = [], []
                for s in range(N_SITES):
                    t2, d2, _ = sample_site(P_tr, G_tr, cl_tr, rng_saa)
                    th_k.append(t2); D_k.append(d2)
                draws.append(supply_dict(np.stack(th_k), np.array(D_k), S, T))
            y_s = solve_saa_y(kwargs, draws, time_limit=30)
            seed_regs["SAA(train)"].append(
                regret({f: 1.0 if v > .5 else 0.0 for f, v in y_s.items()}) if y_s else np.nan)

            draws = []
            for kk in range(args.k):  # SAA-pred：预测中心 Dirichlet + 总量对数扰动
                th_k = np.stack([rng_pert.dirichlet(ALPHA * theta_hat[s]) for s in range(N_SITES)])
                D_k = D_hat * np.exp(rng_pert.normal(0, SIGMA_D, N_SITES))
                draws.append(supply_dict(th_k, D_k, S, T))
            y_p = solve_saa_y(kwargs, draws, time_limit=30)
            seed_regs[f"SAA-pred(a={ALPHA:.0f})"].append(
                regret({f: 1.0 if v > .5 else 0.0 for f, v in y_p.items()}) if y_p else np.nan)
        for m in methods:
            regs[m].extend(seed_regs[m])
        per_seed[seed] = {m: float(np.nanmean(v)) for m, v in seed_regs.items()}
        print(f"seed {seed} done: " + "  ".join(
            f"{m}={np.nanmean(v):.0f}" for m, v in seed_regs.items()), flush=True)

    print(f"\n=== SAA-pred 统一协议（{len(args.seeds)} seeds × {args.n_test} = {len(args.seeds)*args.n_test} 场景）===")
    print(f"{'method':20s} {'mean':>10s} {'95%CI':>12s} {'median':>10s} {'p90':>10s}")
    summary = {}
    for mth, r in regs.items():
        mean, half = t_ci95(r)
        arr = np.array(r, dtype=float)
        summary[mth] = {"mean": mean, "ci95_half": half,
                        "median": float(np.nanmedian(arr)),
                        "p90": float(np.nanquantile(arr, 0.9)),
                        "n": int((~np.isnan(arr)).sum())}
        s = summary[mth]
        print(f"{mth:20s} {s['mean']:10.1f} ±{s['ci95_half']:10.1f} {s['median']:10.1f} {s['p90']:10.1f}")
    json.dump({"config": vars(args) | {"alpha": ALPHA, "sigma_d": SIGMA_D},
               "per_seed_mean": per_seed, "pooled": summary}, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
