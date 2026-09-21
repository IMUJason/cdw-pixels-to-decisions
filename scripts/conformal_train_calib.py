"""A1：用训练集 1,984 图做 conformal 校准集，评估测试集条件覆盖/宽度。

θ 从缓存的面积特征（pred_areas_*.csv / gt_areas_*.csv）用面密度中点确定性折算。
对比：校准集=test-573（旧） vs train-1984（新），均用非零组分条件校准。

用法: python scripts/conformal_train_calib.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.codd_to_coco import CATS  # noqa: E402
from src.vision.material_priors import CODD_PRIORS, CODD_METERS_PER_PIXEL  # noqa: E402

FEAT = Path(r"D:\2026 cdw 0911 datasets\processed\area_features")
AD_MID = np.array([((CODD_PRIORS[c].density_lo + CODD_PRIORS[c].density_hi) / 2)
                   * ((CODD_PRIORS[c].thick_lo + CODD_PRIORS[c].thick_hi) / 2) for c in CATS])
M2PP2 = CODD_METERS_PER_PIXEL ** 2
ALPHA = 0.10


def theta_from_areas(csv_path, pred: bool):
    df = pd.read_csv(csv_path).set_index("image")
    A = df[CATS].to_numpy() * (M2PP2 if pred else 1.0)  # pred 为 px²
    M = A * AD_MID[None, :]
    s = M.sum(1, keepdims=True)
    theta = M / (s + 1e-12)
    return pd.DataFrame(theta, index=df.index, columns=CATS)


def nonzero_q(theta_p: pd.DataFrame, theta_g: pd.DataFrame):
    qs = {}
    for c in CATS:
        r = np.abs(theta_p[c].to_numpy() - theta_g[c].to_numpy())
        mask = theta_g[c].to_numpy() > 1e-6
        rr = r[mask] if mask.sum() >= 10 else r
        k = min(int(np.ceil((len(rr) + 1) * (1 - ALPHA))) - 1, len(rr) - 1)
        qs[c] = float(np.sort(rr)[max(k, 0)])
    return qs


def cond_eval(qs, theta_p, theta_g, title):
    cov, wid = [], []
    for c in CATS:
        p = theta_p[c].to_numpy(); g = theta_g[c].to_numpy()
        m = g > 1e-6
        lo = np.maximum(0, p[m] - qs[c]); hi = np.minimum(1, p[m] + qs[c])
        cov += list((g[m] >= lo) & (g[m] <= hi))
        wid += list(hi - lo)
    print(f"{title}: n={len(cov)}, coverage={np.mean(cov):.3f}, mean_width={np.mean(wid):.4f}")


def main():
    # 测试端 θ：与校准端同管线（面积确定性折算），保证 q 可迁移
    theta_p_te = theta_from_areas(FEAT / "pred_areas_test.csv", pred=True)
    theta_g_te = theta_from_areas(FEAT / "gt_areas_test.csv", pred=False)
    common = theta_p_te.index.intersection(theta_g_te.index)
    theta_p_te, theta_g_te = theta_p_te.loc[common], theta_g_te.loc[common]

    # 校准集 A：测试半区（复现旧结果）
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(common))
    half = len(common) // 2
    qA = nonzero_q(theta_p_te.iloc[idx[:half]], theta_g_te.iloc[idx[:half]])
    cond_eval(qA, theta_p_te.iloc[idx[half:]], theta_g_te.iloc[idx[half:]],
              "校准集=test半区(286)")

    # 校准集 B：训练集 1,984 图（θ 由缓存面积确定性折算）
    theta_p_tr = theta_from_areas(FEAT / "pred_areas_train.csv", pred=True)
    theta_g_tr = theta_from_areas(FEAT / "gt_areas_train.csv", pred=False)
    common_tr = theta_p_tr.index.intersection(theta_g_tr.index)
    qB = nonzero_q(theta_p_tr.loc[common_tr], theta_g_tr.loc[common_tr])
    cond_eval(qB, theta_p_te, theta_g_te, "校准集=train全集(1984)")
    print("qB:", {c: round(v, 4) for c, v in qB.items()})


if __name__ == "__main__":
    main()
