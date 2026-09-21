"""SAA-pred 变体实验（A1 校准自适应 α + A2 CVaR 风险度量 + A3 端点鲁棒）。

方法行：
- vision point            参考（点估计）
- SAA-pred                α=100 固定，期望目标（现有方法，复现）
- SAA-pred-adaptive       A1：每工地 α_s 由 帧离散度⊕校准区间 矩匹配反推
                          σ_g = sqrt(std_frame_g² + (q_rel_g·θ̂_g)²)，
                          α_s = median_g θ̂_g(1-θ̂_g)/σ_g² - 1，clip [10, 4000]
- CVaR-pred               A2：Rockafellar–Uryashev CVaR_0.8，K=10（期望→尾部定价）
- CVaR-pred-adaptive      A1+A2 组合
- robust-pred             A3：Bertsimas–Sim 式预算端点场景（Γ=2 组偏离 +
                          全低/全高，K=7），期望目标——"校准信息在随机-鲁棒谱系的使用"

协议：3 seeds × 30 场景，与其他比较表完全一致（同场景流、独立扰动流、全字典钉死）。
用法: python scripts/run_saapred_variants.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from src.models.cdw_lirp_multi import CDW_LIRP_MultiModel, MATERIALS  # noqa: E402
from src.models.composition_dfl import train_predictor, predict_supply, supply_dict  # noqa: E402
from src.models.saa_baseline import solve_saa_y  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402
from src.vision.site_scenario_bridge import CLASS2GROUP  # noqa: E402
from run_composition_dfl import make_scenario, solve_y  # noqa: E402
from run_composition_dfl_vision import load_split  # noqa: E402
from run_baselines_compare import sample_site, N_SITES, N_IMGS, PURITY, D_SCALE  # noqa: E402

FEAT = Path(r"D:\2026 cdw 0911 datasets\processed\area_features")
ALPHA_CONF = 0.10
K_MEAN, K_CVAR, BETA = 5, 10, 0.8
G2IDX = {g: i for i, g in enumerate(MATERIALS)}


def group_relative_widths():
    """val split 非零条件 conformal 的逐类 q̂ -> 组级相对半宽 q_rel_g（θ̂ 加权）。"""
    import numpy as np
    from src.vision.material_priors import CODD_PRIORS, CODD_METERS_PER_PIXEL
    ad = np.array([((CODD_PRIORS[c].density_lo + CODD_PRIORS[c].density_hi) / 2)
                   * ((CODD_PRIORS[c].thick_lo + CODD_PRIORS[c].thick_hi) / 2) for c in CATS])
    m2 = CODD_METERS_PER_PIXEL ** 2

    def theta(path, pred):
        df = pd.read_csv(path).set_index("image")
        return df.index, df[CATS].to_numpy() * (m2 if pred else 1.0) * ad[None, :]

    idx_p, MP = theta(FEAT / "pred_areas_val.csv", True)
    idx_g, MG = theta(FEAT / "gt_areas_val.csv", False)
    common = idx_p.intersection(idx_g)
    # 两 CSV 行序一致（同一次提取），直接按公共顺序对齐
    tp = MP / (MP.sum(1, keepdims=True) + 1e-12)
    tg = MG / (MG.sum(1, keepdims=True) + 1e-12)
    q_rel = np.zeros(len(MATERIALS))
    w_acc = np.zeros(len(MATERIALS))
    for j, c in enumerate(CATS):
        r = np.abs(tp[:, j] - tg[:, j])
        mask = tg[:, j] > 1e-6
        rr = r[mask] if mask.sum() >= 10 else r
        k = min(int(np.ceil((len(rr) + 1) * (1 - ALPHA_CONF))) - 1, len(rr) - 1)
        q = float(np.sort(rr)[max(k, 0)])
        th_mean = float(np.mean(tp[mask, j])) if mask.sum() else 0.05
        g = G2IDX[CLASS2GROUP[c]]
        q_rel[g] += q * th_mean
        w_acc[g] += th_mean
    q_abs = q_rel / np.maximum(w_acc, 1e-6)          # 组级绝对半宽（θ̂ 加权）
    # 相对端点偏离 δ_g = 组半宽 / 组典型份额（w_acc≈val 上该组期望份额），clip [0.05, 0.35]
    q_rel = np.clip(q_abs / np.maximum(w_acc, 0.05), 0.05, 0.35)
    return q_rel


def site_alpha(theta_hat5, feats10, q_rel5):
    """A1：帧离散度⊕校准宽度 -> 每工地 Dirichlet 浓度 α_s。"""
    g_sh = np.zeros((feats10.shape[0], len(MATERIALS)))
    for j, c in enumerate(CATS):
        g_sh[:, G2IDX[CLASS2GROUP[c]]] += feats10[:, j]
    g_sh = g_sh / (g_sh.sum(1, keepdims=True) + 1e-9)
    std_g = g_sh.std(0)  # 帧间标准差（每工地）
    sig = np.sqrt(std_g ** 2 + (q_rel5 * theta_hat5) ** 2)
    alphas = []
    for g in range(len(MATERIALS)):
        if theta_hat5[g] > 0.02 and sig[g] > 1e-4:
            alphas.append(theta_hat5[g] * (1 - theta_hat5[g]) / sig[g] ** 2 - 1)
    return float(np.clip(np.median(alphas) if alphas else 100.0, 10.0, 4000.0))


def endpoint_draws(theta_hat, D_hat, q_rel5, gamma=2):
    """A3：每工地端点场景——固定 7 模式（中点 + top-Γ 组各低/高 + 全低/全高）。"""
    delta = np.clip(q_rel5, 0.05, 0.35)
    score = delta * (theta_hat > 0.02)
    order = np.argsort(-(score + 1e-3 * theta_hat))  # 阈值不足时退化到 top-θ̂
    tops = list(order[:gamma])
    pats = [np.zeros(len(MATERIALS))]
    for g in tops:
        for sgn in (-1, 1):
            p = np.zeros(len(MATERIALS)); p[g] = sgn; pats.append(p)
    pats.append(-np.ones(len(MATERIALS))); pats.append(np.ones(len(MATERIALS)))
    assert len(pats) == 7
    out = []
    for p in pats:
        m = np.maximum(D_hat * theta_hat * (1 + delta * p), 0)
        out.append((m / m.sum(), float(m.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--out", default=r"D:\2026 cdw 0911 datasets\processed\saapred_variants.json")
    args = ap.parse_args()

    q_rel = group_relative_widths()
    print("组级端点偏离 δ_g =", np.round(q_rel, 3).tolist(), flush=True)

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

    methods = ["vision point", "SAA-pred(K10)", "SAA-pred(K20)",
               "SAA-pred-adaptive(K10)", "CVaR-pred-adaptive(K10)", "robust-pred"]
    regs = {m: [] for m in methods}
    raw = {m: [] for m in methods}      # 逐场景遗憾（尾部分析用）
    per_seed = {}
    for seed in args.seeds:
        seed_regs = {m: [] for m in methods}
        rng = np.random.default_rng(seed)
        rng_pert = np.random.default_rng(10_000 + seed)
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
            th_hat = pred / pred.sum(1, keepdims=True)
            D_hat = pred.sum(1)

            sup_v = supply_dict(th_hat, D_hat, S, T)
            y_v, _ = solve_y(kwargs, sup_v)
            seed_regs["vision point"].append(regret(y_v))

            def eval_y(y_sol, name):
                r = (regret({f: 1.0 if v > .5 else 0.0 for f, v in y_sol.items()})
                     if y_sol else np.nan)
                seed_regs[name].append(r)

            # 分块配对（先全部工地 Dirichlet、后整向量 normal）——与主表脚本约定一致；
            # 2026-09-16 发现交错 vs 分块配对在 K=5 下可造成 ~1.8x 均值差（纯 MC 噪声），
            # 故统一配对并把 K 提到 10/20 以压缩场景抽样噪声
            for name, K, adaptive, risk in [
                    ("SAA-pred(K10)", 10, False, "mean"),
                    ("SAA-pred(K20)", 20, False, "mean"),
                    ("SAA-pred-adaptive(K10)", 10, True, "mean"),
                    ("CVaR-pred-adaptive(K10)", 10, True, "cvar")]:
                alphas = [site_alpha(th_hat[s], feats[s], q_rel) if adaptive else 100.0
                          for s in range(N_SITES)]
                draws = []
                for _ in range(K):
                    th_k = np.stack([rng_pert.dirichlet(alphas[s] * th_hat[s])
                                     for s in range(N_SITES)])
                    D_k = D_hat * np.exp(rng_pert.normal(0, 0.2, N_SITES))
                    draws.append(supply_dict(th_k, D_k, S, T))
                y = solve_saa_y(kwargs, draws, time_limit=30, risk=risk, beta=BETA)
                eval_y(y, name)

            eps = [endpoint_draws(th_hat[s], D_hat[s], q_rel) for s in range(N_SITES)]
            draws = [supply_dict(np.stack([eps[s][j][0] for s in range(N_SITES)]),
                                 np.array([eps[s][j][1] for s in range(N_SITES)]), S, T)
                     for j in range(len(eps[0]))]
            y = solve_saa_y(kwargs, draws, time_limit=30)
            eval_y(y, "robust-pred")
        for m in methods:
            regs[m].extend(seed_regs[m]); raw[m].extend(seed_regs[m])
        per_seed[seed] = {m: float(np.nanmean(v)) for m, v in seed_regs.items()}
        print(f"seed {seed}: " + "  ".join(f"{m}={np.nanmean(v):.0f}"
                                           for m, v in seed_regs.items()), flush=True)

    def t_ci95(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        t = 1.987 if len(x) >= 60 else 2.023
        return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(len(x)))

    print(f"\n=== SAA-pred 变体（{len(args.seeds)} seeds × {args.n_test} 场景）===")
    print(f"{'method':22s} {'mean':>10s} {'95%CI':>11s} {'median':>9s} {'p90':>9s}")
    summary = {}
    for m in methods:
        mean, half = t_ci95(regs[m])
        a = np.array(regs[m], float)
        summary[m] = {"mean": mean, "ci95_half": half, "median": float(np.nanmedian(a)),
                      "p90": float(np.nanquantile(a, 0.9))}
        s = summary[m]
        print(f"{m:22s} {s['mean']:10.1f} ±{s['ci95_half']:9.1f} {s['median']:9.1f} {s['p90']:9.1f}")
    json.dump({"config": vars(args) | {"beta": BETA, "q_rel": q_rel.tolist()},
               "per_seed_mean": per_seed, "pooled": summary,
               "raw_regrets": {m: [None if (isinstance(v, float) and np.isnan(v)) else float(v)
                                   for v in vs] for m, vs in raw.items()}},
              open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
