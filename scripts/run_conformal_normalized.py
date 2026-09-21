"""归一化 conformal 分数（诊断5）：|θ̂-θ|/σ̂ 自适应宽度 vs 同方差分数。

σ̂ 来源：每图 prior-MC——对每类面密度在先验区间内均匀采样 R=200 次，
与该图（预测）面积叉乘得到 θ 的后验样本，σ̂ = 逐类标准差。
同批计算 未归一化/归一化 两种分数的 val 校准 -> test 条件覆盖/宽度，
保证对比在同一约定下（apples-to-apples）。

用法: python scripts/run_conformal_normalized.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.codd_to_coco import CATS  # noqa: E402
from src.vision.material_priors import CODD_PRIORS, CODD_METERS_PER_PIXEL  # noqa: E402

FEAT = Path(r"D:\2026 cdw 0911 datasets\processed\area_features")
ALPHA, R = 0.10, 200
rng = np.random.default_rng(7)
AD_LO = np.array([CODD_PRIORS[c].density_lo * CODD_PRIORS[c].thick_lo for c in CATS])
AD_HI = np.array([CODD_PRIORS[c].density_hi * CODD_PRIORS[c].thick_hi for c in CATS])
M2 = CODD_METERS_PER_PIXEL ** 2


def load_theta(split):
    p = pd.read_csv(FEAT / f"pred_areas_{split}.csv").set_index("image")
    g = pd.read_csv(FEAT / f"gt_areas_{split}.csv").set_index("image")
    common = p.index.intersection(g.index)
    A_pred = p.loc[common, CATS].to_numpy() * M2
    A_gt = g.loc[common, CATS].to_numpy()
    return common, A_pred, A_gt


def theta_det(A):
    ad_mid = (AD_LO + AD_HI) / 2
    M = A * ad_mid[None, :]
    return M / (M.sum(1, keepdims=True) + 1e-12)


def theta_sigma(A):
    """prior-MC：面密度在 [lo, hi] 均匀采样 -> 每图逐类 σ̂。"""
    draws = rng.uniform(AD_LO[None, :], AD_HI[None, :], size=(R, len(AD_LO)))  # [R, C]
    M = A[None, :, :] * draws[:, None, :]                                      # [R, n, C]
    T = M / (M.sum(2, keepdims=True) + 1e-12)
    return T.std(0)                                                            # [n, C]


def calibrate(scores, mask, normalized):
    qs = {}
    for j, c in enumerate(CATS):
        r = scores[:, j]
        m = mask[:, j] > 1e-6
        rr = r[m] if m.sum() >= 10 else r
        k = min(int(np.ceil((len(rr) + 1) * (1 - ALPHA))) - 1, len(rr) - 1)
        qs[c] = float(np.sort(rr)[max(k, 0)])
    return qs


def evaluate(qs, theta_p, theta_g, sigma):
    cov, wid = [], []
    for j, c in enumerate(CATS):
        p, g, s = theta_p[:, j], theta_g[:, j], sigma[:, j]
        m = g > 1e-6
        half = qs[c] * (s[m] if s is not None else 1.0)
        lo = np.maximum(0, p[m] - half)
        hi = np.minimum(1, p[m] + half)
        cov += list((g[m] >= lo) & (g[m] <= hi))
        wid += list(hi - lo)
    return float(np.mean(cov)), float(np.mean(wid))


def main():
    # 校准：val（held-out，同管线）
    _, Ap_v, Ag_v = load_theta("val")
    tp_v, tg_v = theta_det(Ap_v), theta_det(Ag_v)
    sg_v = theta_sigma(Ap_v)
    # 正则化量下限：类级平均绝对误差的一半——漏检类（MCσ̂≈0）不再产生零宽度点区间，
    # 也不会因个别大比率把 q 推爆（2026-09-16 病态修复）
    cmean = np.array([np.mean(np.abs(tp_v[tg_v[:, j] > 1e-6, j] - tg_v[tg_v[:, j] > 1e-6, j]))
                      if (tg_v[:, j] > 1e-6).sum() >= 10 else 0.05 for j in range(len(CATS))])
    sv = np.maximum(sg_v, 0.5 * cmean[None, :])
    q_plain = calibrate(np.abs(tp_v - tg_v), tg_v, False)
    q_norm = calibrate(np.abs(tp_v - tg_v) / sv, tg_v, True)

    # 测试
    _, Ap_t, Ag_t = load_theta("test")
    tp_t, tg_t = theta_det(Ap_t), theta_det(Ag_t)
    st = np.maximum(theta_sigma(Ap_t), 0.5 * cmean[None, :])

    cov_p, wid_p = evaluate(q_plain, tp_t, tg_t, np.ones_like(tp_t))
    cov_n, wid_n = evaluate(q_norm, tp_t, tg_t, st)
    print(f"同方差分数  : conditional coverage={cov_p:.3f}  mean_width={wid_p:.4f}")
    print(f"归一化分数  : conditional coverage={cov_n:.3f}  mean_width={wid_n:.4f}  "
          f"({(1 - wid_n / wid_p) * 100:.1f}% tighter)")
    # 宽度分位数：自适应是否区分难易图
    w_n = []
    for j in range(len(CATS)):
        m = tg_t[:, j] > 1e-6
        w_n += list(2 * q_norm[CATS[j]] * st[m, j])
    w_n = np.array(w_n)
    print(f"归一化宽度分位: p10={np.quantile(w_n, .1):.4f}  p50={np.quantile(w_n, .5):.4f}  "
          f"p90={np.quantile(w_n, .9):.4f}（自适应带宽）")


if __name__ == "__main__":
    main()
