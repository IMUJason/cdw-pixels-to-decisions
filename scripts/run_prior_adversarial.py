"""密度先验对抗实验（"虚拟称重"）：检验组成估计与决策价值对先验误差的鲁棒性。

动机：现协议中真值 θ 与预测 θ̂ 共享同一套密度先验（AD_MID），存在循环性——
θMAE 只反映掩码面积误差，不反映先验本身错误时的组成误差。本实验直接对抗：
- 真值世界 B：GT 面积 × **扰动先验**（模拟"我们以为的先验是错的"）；
- 预测世界 A：预测器仍在 AD_MID 世界训练与推理（部署不变）；
- 评估 B 世界中的遗憾与方法排序。若排序稳定 ⇒ 决策价值对先验误差鲁棒，
  可作为缺少称重真值时的替代证据（WEEE 不批复预案的核心实验）。

扰动水平：
- mild：每类面密度在 [lo·thick_lo, hi·thick_hi] 内独立均匀采样（先验区间内的随机真值）；
- severe：端点对抗——重料（concrete/brick/tile/stone/pipes）取区间高端、
  轻料（wood/gypsum/foam/plastic/general_w）取低端，最大化质量份额失真。

用法: python scripts/run_prior_adversarial.py --n-test 20
"""
import argparse
import json
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
from src.vision.material_priors import CODD_PRIORS  # noqa: E402
from src.vision.site_scenario_bridge import CLASS2GROUP  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402
from run_composition_dfl import make_scenario, solve_y  # noqa: E402
from run_composition_dfl_vision import load_split  # noqa: E402
from run_baselines_compare import sample_site, N_SITES, K_SAA  # noqa: E402

FEAT_DIR = Path(r"D:\2026 cdw 0911 datasets\processed\area_features")
HEAVY = {"concrete", "brick", "tile", "stone", "pipes"}


def ad_table(mode: str, rng: np.random.Generator) -> dict:
    """每类面密度（t/m²）扰动表。"""
    out = {}
    for c in CATS:
        p = CODD_PRIORS[c]
        lo, hi = p.density_lo * p.thick_lo, p.density_hi * p.thick_hi
        if mode == "mid":
            out[c] = (p.density_lo + p.density_hi) / 2 * (p.thick_lo + p.thick_hi) / 2
        elif mode == "mild":
            out[c] = rng.uniform(lo, hi)
        elif mode == "severe":
            out[c] = hi if c in HEAVY else lo
        else:
            raise ValueError(mode)
    return out


def gt_mass_world(split: str, ad: dict) -> tuple[np.ndarray, list]:
    """用给定面密度表把 GT 面积映射为 5 组质量（真值世界）。"""
    gt = pd.read_csv(FEAT_DIR / f"gt_areas_{split}.csv").set_index("image")
    mass = np.zeros((len(gt), len(MATERIALS)))
    for j, c in enumerate(CATS):
        mass[:, MATERIALS.index(CLASS2GROUP[c])] += gt[CATS].to_numpy()[:, j] * ad[c]
    return mass, list(gt.index)


def run_world(mode: str, n_test: int, seed: int, model, P_te, cl_te, P_tr, G_tr_mid, cl_tr):
    """在扰动真值世界上评估 4 方法遗憾。"""
    rng = np.random.default_rng(seed)
    ad = ad_table(mode, rng)
    G_te_B, _ = gt_mass_world("test", ad)
    # 与 load_split 的 common 对齐（test split 的 P/G 索引一致，直接用全部）
    methods = ["no-vision", "vision", "SAA(train)", "SAA-pred", "oracle"]
    regs = {m: [] for m in methods}
    rng_saa = np.random.default_rng(777)
    rng_pert = np.random.default_rng(10_000 + seed)
    # no-vision 参考：A 世界（部署方认知）的训练池均值
    mean_mix = None
    rng_nv = np.random.default_rng(1)
    xs_nv = []
    for _ in range(500):
        th, d, _ = sample_site(P_tr, G_tr_mid, cl_tr, rng_nv)
        xs_nv.append(d * th)
    xs_nv = np.array(xs_nv)
    mean_mix = xs_nv.mean(0) / xs_nv.mean(0).sum()
    mean_tot = float(xs_nv.sum(1).mean())

    for k in range(n_test):
        theta, D, feats = [], [], []
        for s in range(N_SITES):
            th, d, feat = sample_site(P_te, G_te_B, cl_te, rng)  # 真值来自 B 世界
            theta.append(th); D.append(d); feats.append(feat)
        theta = np.stack(theta); D = np.array(D)
        kwargs, S, T, _, _ = make_scenario(rng, np.eye(5), np.ones(1), n_sites=N_SITES,
                                           cap_factor=0.6, sites_override=(theta, D))
        true_sup = supply_dict(theta, D, S, T)
        y_o, z_star = solve_y(kwargs, true_sup)

        def regret(y_hat):
            rec = CDW_LIRP_MultiModel(supply=true_sup, y_fixed=y_hat,
                                      lp_relaxation=True, **kwargs).solve(time_limit=30)
            return rec["objective"] - z_star if rec else np.nan

        regs["oracle"].append(regret(y_o))
        th_n = np.tile(mean_mix, (N_SITES, 1))
        y_n, _ = solve_y(kwargs, supply_dict(th_n, np.full(N_SITES, mean_tot / N_SITES), S, T))
        regs["no-vision"].append(regret(y_n))
        pred = np.clip(predict_supply(model, np.array(feats)), 1e-6, None)
        th_hat = pred / pred.sum(1, keepdims=True)
        y_v, _ = solve_y(kwargs, supply_dict(th_hat, pred.sum(1), S, T))
        regs["vision"].append(regret(y_v))
        draws = []
        for _ in range(K_SAA):
            th_k, D_k = [], []
            for s in range(N_SITES):
                t2, d2, _ = sample_site(P_tr, G_tr_mid, cl_tr, rng_saa)
                th_k.append(t2); D_k.append(d2)
            draws.append(supply_dict(np.stack(th_k), np.array(D_k), S, T))
        y_s = solve_saa_y(kwargs, draws, time_limit=30)
        regs["SAA(train)"].append(
            regret({f: 1.0 if v > .5 else 0.0 for f, v in y_s.items()}) if y_s else np.nan)
        # SAA-pred：以（A 世界训练的）预测为中心对冲，在 B 世界评估
        draws = []
        for _ in range(K_SAA):
            th_k = np.stack([rng_pert.dirichlet(100.0 * th_hat[s]) for s in range(N_SITES)])
            D_k = pred.sum(1) * np.exp(rng_pert.normal(0, 0.2, N_SITES))
            draws.append(supply_dict(th_k, D_k, S, T))
        y_p = solve_saa_y(kwargs, draws, time_limit=30)
        regs["SAA-pred"].append(
            regret({f: 1.0 if v > .5 else 0.0 for f, v in y_p.items()}) if y_p else np.nan)
    return {m: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                "p90": float(np.nanquantile(v, 0.9))} for m, v in regs.items()}, ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    P_tr, G_tr_mid, _, LAB_tr = load_split("train")   # A 世界（部署方认知）
    P_te, G_te_mid, _, LAB_te = load_split("test")
    cl_tr = [np.where(LAB_tr == c)[0] for c in range(4)]
    cl_te = [np.where(LAB_te == c)[0] for c in range(4)]

    # 预测器在 A 世界（AD_MID）训练——模拟"用错误先验训练的模型部署到真实世界"
    rng0 = np.random.default_rng(1)
    xs, ys = [], []
    for _ in range(500):
        th, D, feat = sample_site(P_tr, G_tr_mid, cl_tr, rng0)
        xs.append(feat); ys.append(D * th)
    model = train_predictor(np.array(xs), np.array(ys), "mse", epochs=400, seed=0)

    results = {}
    for mode in ("mid", "mild", "severe"):
        summ, ad = run_world(mode, args.n_test, args.seed, model,
                             P_te, cl_te, P_tr, G_tr_mid, cl_tr)
        results[mode] = summ
        print(f"\n=== 先验世界 {mode}（seed {args.seed} × {args.n_test} 场景）===")
        print(f"{'method':12s} {'mean':>10s} {'median':>10s} {'p90':>10s}")
        for m, s in summ.items():
            print(f"{m:12s} {s['mean']:10.1f} {s['median']:10.1f} {s['p90']:10.1f}")
        if mode != "mid":
            print("  扰动面密度示例: " + ", ".join(
                f"{c}={ad[c]:.3f}" for c in ("concrete", "wood", "plastic")))

    out = Path(r"D:\2026 cdw 0911 datasets\processed\prior_adversarial.json")
    json.dump({"config": vars(args), "results": results}, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
