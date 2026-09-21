"""动作2b：用称重数据定量化不可约的面密度变异，检验手设先验区间是否过窄。

动机链（今日已建立）：
  尺度通道（质量代理 mpx = Σ_c P·AD_MID）被证明是**决策关键路径**
  （可加物理头使 D̂ 误差 0.266→0.094、p90 降 33-40×），而 mpx 依赖手设密度先验
  AD_MID = ρ·τ。若先验区间过窄，则传播出的不确定性被低估——而 conformal 层
  会"默默补偿"，掩盖了先验本身的错误。这正是伪真值循环性质疑的实质。

方法：
  用 RAMSES 85,424 个**称重**颗粒实例（etalab 2.0，仅需标注表）：
  1. 逐类面密度 AD = mass/area（g/px²），**按颗粒尺寸类分层**以分离尺寸效应；
  2. 同尺寸类内的剩余离散 = 不可约的材料/形状变异（任何图像都预测不了
     "这一颗粒是否更厚"）；
  3. 与 CODD 先验区间的 log10 宽度对比 -> 先验低估倍数。

用法: python scripts/run_ramses_prior_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.vision.material_priors import CODD_PRIORS  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402

RAMSES = Path(r"D:\2026 cdw 0911 datasets\ramses\annotations.tab")
OUT = Path(r"D:\2026 cdw 0911 datasets\processed\ramses_prior_audit.json")

# RAMSES 类 -> CODD 类（仅矿物族有可比对象；轻料样本极少，如实标注为不足）
CLASS_MAP = {"Rc": "concrete", "Ra": "stone", "Rb01": "brick", "Rb02": "brick",
             "Ru01": "concrete", "Ru02": "concrete", "Ru03": "concrete",
             "Ru04": "concrete", "Ru05": "concrete", "Ru06": "concrete",
             "Rcu01": "concrete", "Rg": "gypsum_board", "Pl": "plastic"}


def prior_log_width(cat):
    p = CODD_PRIORS[cat]
    lo = p.density_lo * p.thick_lo
    hi = p.density_hi * p.thick_hi
    return float(np.log10(hi / lo))


def main():
    df = pd.read_csv(RAMSES, sep="\t", low_memory=False)
    df = df[df["mass"].notna() & (df["mass"] > 0) & (df["area"] > 0)].copy()
    df = df[df["class"].isin(CLASS_MAP)]
    df["codd"] = df["class"].map(CLASS_MAP)
    df["logad"] = np.log10(df["mass"] / df["area"])
    # 尺寸分层：res 是颗粒尺寸类（mm）；归到最近的 12.5 / 25 / 50 档
    df["size"] = pd.cut(df["res"], bins=[0, 17, 35, 100], labels=["12.5", "25", "50"])

    print(f"有效实例 {len(df)}；尺寸分层样本数：")
    print(df["size"].value_counts().to_string())

    rows = []
    for (cat, size), g in df.groupby(["codd", "size"], observed=True):
        if len(g) < 30:
            continue
        p2, p98 = np.percentile(g["logad"], [2.5, 97.5])
        rows.append({"codd": cat, "size": str(size), "n": len(g),
                     "log10_med": float(g["logad"].median()),
                     "span_dex": float(p98 - p2),
                     "sd_dex": float(g["logad"].std()),
                     "prior_width_dex": prior_log_width(cat)})
    tab = pd.DataFrame(rows).sort_values(["codd", "size"])
    tab["prior_over_emp"] = tab["prior_width_dex"] / tab["span_dex"]

    print("\n=== 同尺寸类内的面密度离散（不可约）vs 先验区间宽度 ===")
    print(f"{'CODD类':14s} {'尺寸':5s} {'n':>7s} {'log10中位':>10s} "
          f"{'经验span':>9s} {'经验sd':>8s} {'先验宽度':>9s} {'先验/经验':>9s}")
    for _, r in tab.iterrows():
        print(f"{r['codd']:14s} {r['size']:5s} {int(r['n']):7d} {r['log10_med']:10.3f} "
              f"{r['span_dex']:9.2f} {r['sd_dex']:8.2f} {r['prior_width_dex']:9.2f} "
              f"{r['prior_over_emp']:9.2f}")

    # 矿物族（占吨位主导）汇总
    min_ = tab[tab["codd"].isin(["concrete", "brick", "stone"])]
    if len(min_):
        emp = float(np.average(min_["span_dex"], weights=min_["n"]))
        pri = float(np.average(min_["prior_width_dex"], weights=min_["n"]))
        print(f"\n矿物族（按样本加权）：经验 span={emp:.2f} dex，"
              f"先验宽度={pri:.2f} dex ⇒ 先验低估 {emp / pri:.1f}×")

    # 决策含义：若密度变异按经验值放大，prior-MC 传播的组成不确定性相应放大
    print("\n=== 决策含义 ===")
    print("先验区间过窄 ⇒ prior-MC 传播的组成σ̂ 被低估 ⇒ 校准层被迫在不匹配的")
    print("先验上补偿（E2 中 conformal 条件宽度 0.278 vs prior-MC 0.109 即此现象）。")
    print("同时 mpx 质量代理的**不可约误差地板**由经验 sd 决定（无图像可消除）。")

    json.dump({"table": tab.to_dict(orient="records"),
               "mineral_family": {"empirical_span_dex": emp, "prior_width_dex": pri,
                                  "understated_x": emp / pri} if len(min_) else {}},
              open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
