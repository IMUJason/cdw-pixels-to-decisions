"""动作2a 失败诊断：为什么给了质量代理特征，D̂ 误差仍高达 25%？

假设检验：
  H1（学习问题）  尺度信号是信息性的，但 MLP 没学会 -> 换特征编码/加样本可救
  H2（感知问题）  预测掩码的**绝对面积**本身误差大（漏检+掩码偏小），
                  而组成份额经归一化后误差相消 -> 任何特征工程都救不了，
                  必须改感知或引入称重标定

诊断（纯 numpy）：
  A. 逐帧质量代理 mpx = Σ_c P[c]·AD_MID[c] vs 逐帧真值质量 gm：
     比值分布（偏差与离散度）——H2 的直接证据
  B. 工地级：Σ_frames mpx 与 D_true 的相关性、最优线性拟合后的残余误差
  C. 参考下界：常数 D（no-vision 口径）的误差
  若 (B) 拟合后残余仍 ≈25% ⇒ H2 成立，2a 的零结果由感知绝对面积误差解释。

用法: python scripts/diag_scale_channel.py
"""
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from run_composition_dfl_vision import load_split, AD_MID  # noqa: E402
from run_baselines_compare import N_SITES, N_IMGS, PURITY, D_SCALE  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402

AD = np.array([AD_MID[c] for c in CATS])


def main():
    for split in ("val", "test"):
        P, G_mass, _, LAB = load_split(split)
        cl = [np.where(LAB == c)[0] for c in range(4)]
        # ---- A. 逐帧 ----
        mpx_frame = (P * AD[None, :]).sum(1)          # 预测质量代理 [n]
        gm_frame = G_mass.sum(1)                      # 真值质量 [n]
        r = mpx_frame / np.maximum(gm_frame, 1e-12)
        lr = np.log10(r)
        print(f"\n=== {split}：逐帧质量代理/真值（n={len(P)}）===")
        print(f"  比值 中位={np.median(r):.3f}  p10={np.percentile(r,10):.3f}  "
              f"p90={np.percentile(r,90):.3f}  log10 标准差={lr.std():.3f} dex")
        print(f"  ⇒ 单帧质量估计相对离散度 ≈ {10**lr.std():.2f}×")

        # ---- B/C. 工地级 ----
        rng = np.random.default_rng(11)
        S_mpx, D_true = [], []
        for _ in range(300):
            Ds, sm = [], []
            for s in range(N_SITES):
                c = cl[rng.integers(0, 4)]
                n_in = int(PURITY * N_IMGS)
                idx = np.concatenate([rng.choice(c, n_in, replace=True),
                                      rng.choice(len(P), N_IMGS - n_in, replace=True)])
                gm = G_mass[idx].sum(0) * D_SCALE
                Ds.append(gm.sum())
                sm.append(mpx_frame[idx].sum() * D_SCALE)
            if len(Ds) == N_SITES:
                D_true.append(np.mean(Ds)); S_mpx.append(np.mean(sm))
        D_true = np.array(D_true); S_mpx = np.array(S_mpx)
        corr = np.corrcoef(S_mpx, D_true)[0, 1]
        # 常数基线
        const = D_true.mean()
        e_const = np.mean(np.abs(const - D_true) / D_true)
        # 最优线性（含过原点与带截距）
        a1 = float((S_mpx @ D_true) / (S_mpx @ S_mpx))
        e_lin0 = np.mean(np.abs(a1 * S_mpx - D_true) / D_true)
        A = np.vstack([S_mpx, np.ones_like(S_mpx)]).T
        coef, *_ = np.linalg.lstsq(A, D_true, rcond=None)
        e_lin1 = np.mean(np.abs(A @ coef - D_true) / D_true)
        print(f"  工地级：corr(Σmpx, D_true)={corr:.3f}")
        print(f"  D̂ 相对误差 —— 常数(no-vision)={e_const:.4f} | "
              f"最优比例 a={a1:.4g} -> {e_lin0:.4f} | 线性+截距 -> {e_lin1:.4f}")
        print(f"  ⇒ 若线性拟合后仍 ≈0.2+，则尺度信号本身噪声主导（H2）")


if __name__ == "__main__":
    main()
