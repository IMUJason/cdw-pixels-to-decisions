"""动作1（固定评测协议）+ 动作2a（恢复被丢弃的尺度通道）。

动作1 —— 协议固定：
  (a) 抽签调用顺序统一为**分块**（先全部工地 Dirichlet、后整向量 lognormal），
      消除"交错 vs 分块"配对造成的 1.8× 均值漂移（本轮已实测）；
  (b) R=3 个独立抽签库取平均，并报告跨库离散度作为残余噪声地板；
  (c) 主报 median / p90（尾部对单次抽样不敏感），均值仅作参考。

动作2a —— 尺度通道：
  现有特征 = 逐帧组成份额 P[i]/ΣP[i]（每帧和为1）→ **每帧绝对尺度被丢弃**，
  而目标是绝对吨位 D = D_SCALE · Σ_frames Σ_c G[f,c]·AD_MID[c]。
  变体特征追加两个逐帧标量：log1p(ΣP) 与 ΣP·AD_MID（部署方本已可算）。
  对照：F0=份额(10) vs F1=份额+尺度(12)，同协议、同场景、同抽签。

用法: python scripts/run_protocol_fixed.py
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
from run_composition_dfl_vision import load_split, AD_MID  # noqa: E402
from run_baselines_compare import N_SITES, N_IMGS, PURITY, D_SCALE  # noqa: E402
from run_saapred_variants import group_relative_widths, site_alpha  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402

AD_MID_ARR = np.array([AD_MID[c] for c in CATS])   # t/m²，逐类
K_MAIN, BETA, N_BANKS = 10, 0.8, 3


def feats_share(P_frames):
    """F0：逐帧组成份额 [N,10]（现状实现）。"""
    return P_frames / (P_frames.sum(1, keepdims=True) + 1e-9)


def feats_scaled(P_frames):
    """F1：份额 [N,10] + log1p(逐帧总面积) + 逐帧质量代理 [N,12]。"""
    sh = feats_share(P_frames)
    tot = P_frames.sum(1, keepdims=True)
    mpx = (P_frames * AD_MID_ARR[None, :]).sum(1, keepdims=True)
    return np.concatenate([sh, np.log1p(tot), mpx], axis=1)


def draw_site(P_pool, G_pool, clusters, rng):
    """按原型簇采样一个工地：返回 (theta, D, 帧面积矩阵)。"""
    cl = clusters[rng.integers(0, 4)]
    n_in = int(PURITY * N_IMGS)
    idx = np.concatenate([rng.choice(cl, n_in, replace=True),
                          rng.choice(len(P_pool), N_IMGS - n_in, replace=True)])
    gm = G_pool[idx].sum(0) * D_SCALE
    return gm / max(gm.sum(), 1e-9), float(gm.sum()), P_pool[idx]


def train_model(P_tr, G_tr, cl_tr, feat_fn):
    rng0 = np.random.default_rng(1)
    xs, ys = [], []
    for _ in range(500):
        th, D, Pf = draw_site(P_tr, G_tr, cl_tr, rng0)
        xs.append(feat_fn(Pf)); ys.append(D * th)
    ys = np.array(ys)
    model = train_predictor(np.array(xs), ys, "mse", epochs=400, seed=0)
    mix = ys.mean(0) / ys.mean(0).sum()
    return model, mix, float(ys.sum(1).mean()), None   # 第4项=可选标定因子 a


def fit_scale_factor(P_tr, G_tr, cl_tr):
    """在训练分布上拟合质量代理的标定因子 a：D ≈ a · Σ_frames mpx（逐帧可加量）。
    诊断证据：线性拟合后 D̂ 相对误差 6%（vs 网络 25%）——信息在，网络没用上。"""
    rng = np.random.default_rng(1)
    S, Dt = [], []
    for _ in range(500):
        th, D, Pf = draw_site(P_tr, G_tr, cl_tr, rng)
        mpx = (Pf * AD_MID_ARR[None, :]).sum(1)
        S.append(mpx.sum() * D_SCALE); Dt.append(D)
    S, Dt = np.array(S), np.array(Dt)
    return float((S @ Dt) / (S @ S))


def t_ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), 0.0
    t = 1.987 if len(x) >= 60 else 2.023
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--out",
                    default=r"D:\2026 cdw 0911 datasets\processed\protocol_fixed_2a.json")
    args = ap.parse_args()

    q_rel = group_relative_widths()
    P_tr, G_tr, _, LAB_tr = load_split("train")
    P_te, G_te, _, LAB_te = load_split("test")
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    cl_te = [np.where(LAB_te == c)[0] for c in range(4)]

    variants = {}
    for name, fn in (("F0_shares", feats_share), ("F1_shares+scale", feats_scaled)):
        print(f"[train] {name}", flush=True)
        variants[name] = train_model(P_tr, G_tr, cl_tr, fn)
    print("[train] F2 physics head (scale factor a)", flush=True)
    a_scale = fit_scale_factor(P_tr, G_tr, cl_tr)
    print(f"  a = {a_scale:.6g}")
    variants["F2_physhead"] = (variants["F0_shares"][0], variants["F0_shares"][1],
                               variants["F0_shares"][2], a_scale)

    planners = ["vision point", "SAA(train)", f"SAA-pred(K={K_MAIN})",
                f"CVaR-pred-adaptive(K={K_MAIN})"]
    regs = {(v, p): [] for v in variants for p in planners}
    d_err = {v: [] for v in variants}
    mix_err = {v: [] for v in variants}
    per_seed = {}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)             # 场景流
        rng_saa = np.random.default_rng(777)          # SAA(train) 固定流
        seed_regs = {(v, p): [] for v in variants for p in planners}
        for k in range(args.n_test):
            theta, D, frames = [], [], []
            for s in range(N_SITES):
                th, d, Pf = draw_site(P_te, G_te, cl_te, rng)
                theta.append(th); D.append(d); frames.append(Pf)
            theta = np.stack(theta); D = np.array(D)
            kwargs, S, T, _, _ = make_scenario(rng, np.eye(5), np.ones(1), n_sites=N_SITES,
                                               cap_factor=0.6, sites_override=(theta, D))
            true_sup = supply_dict(theta, D, S, T)
            _, z_star = solve_y(kwargs, true_sup)

            def regret(y_hat):
                rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                          lp_relaxation=True, **kwargs).solve(time_limit=30)
                return rec["objective"] - z_star if rec else np.nan

            # SAA(train)：与变体无关，场景级一次
            draws = []
            for _ in range(K_MAIN):
                th_k, D_k = [], []
                for s in range(N_SITES):
                    t2, d2, _ = draw_site(P_tr, G_tr, cl_tr, rng_saa)
                    th_k.append(t2); D_k.append(d2)
                draws.append(supply_dict(np.stack(th_k), np.array(D_k), S, T))
            y_tr = solve_saa_y(kwargs, draws, time_limit=30)
            r_tr = (regret({f: 1.0 if v > .5 else 0.0 for f, v in y_tr.items()})
                    if y_tr else np.nan)

            for vname, (model, _, _, a_s) in variants.items():
                F = np.stack([(feats_scaled(frames[s]) if vname.startswith("F1")
                               else feats_share(frames[s])) for s in range(N_SITES)])
                pred = np.clip(predict_supply(model, F), 1e-6, None)
                th_hat = pred / pred.sum(1, keepdims=True)
                if a_s is not None:
                    # F2：组成取自网络，总量用可加物理头（逐帧质量代理求和后标定）
                    S_sum = np.array([(frames[s] * AD_MID_ARR[None, :]).sum() * D_SCALE
                                      for s in range(N_SITES)])
                    D_hat = a_s * S_sum
                else:
                    D_hat = pred.sum(1)
                d_err[vname].append(float(np.mean(np.abs(D_hat - D) / np.maximum(D, 1e-9))))
                mix_err[vname].append(float(np.mean(np.abs(th_hat - theta))))

                y_v, _ = solve_y(kwargs, supply_dict(th_hat, D_hat, S, T))
                seed_regs[(vname, "vision point")].append(regret(y_v))
                seed_regs[(vname, "SAA(train)")].append(r_tr)

                for r in range(N_BANKS):
                    rng_p = np.random.default_rng(1000 * (r + 1) + seed)
                    alphas = [site_alpha(th_hat[s], F[s][:, :10], q_rel)
                              for s in range(N_SITES)]
                    # 分块调用：先全部工地 Dirichlet，再整向量 lognormal
                    draws = []
                    for _ in range(K_MAIN):
                        th_k = np.stack([rng_p.dirichlet(100.0 * th_hat[s])
                                         for s in range(N_SITES)])
                        D_k = D_hat * np.exp(rng_p.normal(0, 0.2, N_SITES))
                        draws.append(supply_dict(th_k, D_k, S, T))
                    y_p = solve_saa_y(kwargs, draws, time_limit=30, risk="mean")
                    seed_regs[(vname, f"SAA-pred(K={K_MAIN})")].append(
                        regret({f: 1.0 if v > .5 else 0.0 for f, v in y_p.items()})
                        if y_p else np.nan)

                    draws = []
                    for _ in range(K_MAIN):
                        th_k = np.stack([rng_p.dirichlet(alphas[s] * th_hat[s])
                                         for s in range(N_SITES)])
                        D_k = D_hat * np.exp(rng_p.normal(0, 0.2, N_SITES))
                        draws.append(supply_dict(th_k, D_k, S, T))
                    y_c = solve_saa_y(kwargs, draws, time_limit=30, risk="cvar", beta=BETA)
                    seed_regs[(vname, f"CVaR-pred-adaptive(K={K_MAIN})")].append(
                        regret({f: 1.0 if v > .5 else 0.0 for f, v in y_c.items()})
                        if y_c else np.nan)
        for key in seed_regs:
            regs[key].extend(seed_regs[key])
        per_seed[seed] = {f"{v}|{p}": float(np.nanmean(seed_regs[(v, p)]))
                          for v in variants for p in planners}
        print(f"seed {seed}: " + "  ".join(
            f"{v.split('_')[0]}/{p.split('(')[0].split()[0]}={np.nanmean(seed_regs[(v,p)]):.0f}"
            for v in variants for p in planners), flush=True)

    print(f"\n=== 动作1+2a（{len(args.seeds)} seeds × {args.n_test} 场景 × {N_BANKS} banks）===")
    print(f"{'variant':18s} {'planner':26s} {'mean':>9s} {'±CI':>8s} {'median':>9s} {'p90':>9s}")
    summary = {}
    for v in variants:
        for p in planners:
            arr = np.array(regs[(v, p)], float)
            mean, half = t_ci95(arr)
            summary[f"{v}|{p}"] = {"mean": mean, "ci95_half": half,
                                   "median": float(np.nanmedian(arr)),
                                   "p90": float(np.nanquantile(arr, 0.9))}
            s = summary[f"{v}|{p}"]
            print(f"{v:18s} {p:26s} {s['mean']:9.1f} ±{s['ci95_half']:7.1f} "
                  f"{s['median']:9.1f} {s['p90']:9.1f}")
        print(f"  → {v}: D̂相对误差={np.mean(d_err[v]):.4f}  θ̂MAE={np.mean(mix_err[v]):.4f}")

    json.dump({"config": vars(args) | {"K": K_MAIN, "beta": BETA, "n_banks": N_BANKS},
               "per_seed_mean": per_seed, "summary": summary,
               "d_rel_err": {v: float(np.mean(d_err[v])) for v in variants},
               "mix_mae": {v: float(np.mean(mix_err[v])) for v in variants}},
              open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
