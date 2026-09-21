"""RAMSES 质量级外部验证（WEEE 替代方案）：用 85,800 个带逐实例质量的骨料颗粒，
检验"面积×面密度先验 → 质量"链条的误差结构——回应伪真值循环性质疑。

数据：Recherche Data Gouv doi:10.57745/KC4EA2（Lux 2025, etalab 2.0），
仅使用 annotations.tab（89,600 实例：area px²、mass g、class、baseimg），无需图像。

三个任务：
A. 逐类经验面密度（mass/area）统计：分布、p5-p95 区间、与 CODD 先验区间的对照；
B. 组成误差：同图（baseimg）内"面积份额" vs "质量份额"的逐类偏差与 θMAE——
   直接量化"把面积当质量"在组成层面的误差（CDW 伪真值协议的核心假设检验）；
C. 面积→质量的回归校准可行性：log-log 线性（mass ≈ a·area^b·ρ_class）的 R²，
   说明先验+一次标定可把组成误差压到什么量级。

用法: python scripts/run_ramses_masscheck.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAMSES = Path(r"D:\2026 cdw 0911 datasets\ramses")
OUT = Path(r"D:\2026 cdw 0911 datasets\processed\ramses_masscheck.json")

# RAMSES 类 -> 与 CODD 对齐的物料族（Ra/Rc/Rb/Ru* = 矿物/骨料；Rg=石膏；Pl=塑料；
# X*=杂类；Coin/SHELLS/UNKNOWN 忽略）
FAMILY = {"Rc": "mineral", "Ra": "mineral", "Rb01": "mineral", "Rb02": "mineral",
          "Ru01": "mineral", "Ru02": "mineral", "Ru03": "mineral", "Ru04": "mineral",
          "Ru05": "mineral", "Ru06": "mineral", "Rcu01": "mineral",
          "Rg": "gypsum", "Pl": "plastic_mix",
          "X01": "general", "X02": "general", "X03": "general", "X04": "general"}


def main():
    df = pd.read_csv(RAMSES / "annotations.tab", sep="\t", low_memory=False)
    df = df[df["class"].isin(FAMILY)].copy()
    df = df[df["mass"].notna() & (df["mass"] > 0) & (df["area"] > 0)]
    df["family"] = df["class"].map(FAMILY)
    print(f"有效实例: {len(df)} / 89,600")

    # --- 任务 A：逐类经验面密度（g/px²），分布与区间 ---
    df["ad"] = df["mass"] / df["area"]
    print("\n=== A. 逐类经验面密度（g/px²，log10 中位数 [p5, p95]）===")
    ad_stats = {}
    for fam, g in df.groupby("family"):
        la = np.log10(g["ad"])
        med, p5, p95 = float(la.median()), float(la.quantile(.05)), float(la.quantile(.95))
        ad_stats[fam] = {"n": len(g), "log10_med": med, "log10_p5": p5, "log10_p95": p95,
                         "span_dex": p95 - p5}
        print(f"{fam:12s} n={len(g):6d}  10^{med:.3f}  [10^{p5:.3f}, 10^{p95:.3f}]  "
              f"跨度 {p95-p5:.2f} dex")

    # --- 任务 B：同图组成——面积份额 vs 质量份额 ---
    print("\n=== B. 组成误差（面积份额 vs 质量份额，按 baseimg）===")
    fams = sorted(df["family"].unique())
    recs = []
    for img, g in df.groupby("baseimg"):
        if len(g) < 5:
            continue
        a = g.groupby("family")["area"].sum().reindex(fams, fill_value=0.0).to_numpy()
        m = g.groupby("family")["mass"].sum().reindex(fams, fill_value=0.0).to_numpy()
        if a.sum() <= 0 or m.sum() <= 0:
            continue
        recs.append((a / a.sum(), m / m.sum()))
    print(f"有效图（≥5 实例）: {len(recs)}")
    mae_area = float(np.mean([np.abs(a - m).mean() for a, m in recs]))
    cos_area = float(np.mean([a @ m / (np.linalg.norm(a) * np.linalg.norm(m) + 1e-12)
                              for a, m in recs]))
    # 基线对照：面积×逐类中位面密度（一次标定）后的组成
    med_ad = {f: 10 ** ad_stats[f]["log10_med"] for f in fams}
    mae_cal = float(np.mean([
        np.abs((a * np.array([med_ad[f] for f in fams]) / (a * np.array([med_ad[f] for f in fams])).sum()) - m).mean()
        if (a * np.array([med_ad[f] for f in fams])).sum() > 0 else np.nan
        for a, m in recs]))
    print(f"面积份额直接当质量份额: θMAE={mae_area:.4f}  cosine={cos_area:.4f}")
    print(f"面积×类中位面密度标定:  θMAE={mae_cal:.4f}")

    # --- 任务 C：log-log 回归 mass ~ area（逐族） ---
    print("\n=== C. log10(mass) ~ log10(area) 回归（逐族 R²）===")
    reg = {}
    for fam, g in df.groupby("family"):
        x, y = np.log10(g["area"]), np.log10(g["mass"])
        b, a0 = np.polyfit(x, y, 1)
        r2 = 1 - float(np.var(y - (b * x + a0)) / np.var(y))
        reg[fam] = {"slope_b": float(b), "intercept": float(a0), "R2": r2}
        print(f"{fam:12s} b={b:.3f}  R²={r2:.3f}")

    json.dump({"n_instances": len(df), "ad_stats": ad_stats,
               "composition": {"thetaMAE_area": mae_area, "cosine_area": cos_area,
                               "thetaMAE_calibrated": mae_cal, "n_images": len(recs)},
               "regression": reg}, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
