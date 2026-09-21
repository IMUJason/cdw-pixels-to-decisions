"""伪真值管线 v1：CODD 多边形 -> 每类质量 -> 组成向量 θ 与总量 D（带不确定性）。

物理链：多边形像素面积 ×(m/px)²→ 平面面积 ×面密度(密度×厚度先验采样)→ 每类质量
组成 θ = 质量归一化；总量 D = Σ质量。不确定性来自密度/厚度先验区间（Monte Carlo）。

用法:
    python -m src.vision.composition_estimator --codd-root "D:/.../codd/..." --split testing \
        --n-mc 200 --out data/processed/composition_testing.csv
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.codd_to_coco import CATS, parse_polygons  # noqa: E402
from src.vision.material_priors import CODD_PRIORS, CODD_METERS_PER_PIXEL  # noqa: E402


def shoelace(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0  # px²


def image_class_areas(xml_path: str):
    """返回 {class: 平面面积 m²}（用多边形，而非 bbox——更贴近真实可见面积）。"""
    _, objs = parse_polygons(xml_path)
    m2_per_px2 = CODD_METERS_PER_PIXEL ** 2
    areas = defaultdict(float)
    for o in objs:
        if o["label"] in CODD_PRIORS:
            areas[o["label"]] += shoelace(o["polygon"]) * m2_per_px2
    return areas


def estimate_image(xml_path: str, n_mc: int, rng):
    """MC 采样返回 (组成 θ 样本 [n_mc, K], 总量 D 样本 [n_mc])。"""
    areas = image_class_areas(xml_path)
    present = [c for c in CATS if areas.get(c, 0) > 0]
    if not present:
        return np.zeros((n_mc, len(CATS))), np.zeros(n_mc)
    A = np.array([areas[c] for c in present])          # [k] m²
    ad = np.array([CODD_PRIORS[c].sample(rng) for c in present])  # [k] t/m²（先验中心）
    # MC：面密度在区间内再扰动（先验采样独立性）
    mass = np.zeros((n_mc, len(present)))
    for j, c in enumerate(present):
        p = CODD_PRIORS[c]
        lo, hi = p.density_lo * p.thick_lo, p.density_hi * p.thick_hi
        mass[:, j] = A[j] * rng.uniform(lo, hi, size=n_mc)
    theta = mass / mass.sum(axis=1, keepdims=True)
    full_theta = np.zeros((n_mc, len(CATS)))
    for j, c in enumerate(present):
        full_theta[:, CATS.index(c)] = theta[:, j]
    return full_theta, mass.sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codd-root", required=True)
    ap.add_argument("--split", default="testing")
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for xp in sorted(glob.glob(os.path.join(args.codd_root, args.split, "*.xml"))):
        theta, D = estimate_image(xp, args.n_mc, rng)
        q = np.quantile(D, [0.05, 0.5, 0.95])
        row = {"image": Path(xp).stem,
               "D_lo": q[0], "D_med": q[1], "D_hi": q[2]}  # 单位 t/图（可见视场内）
        for i, c in enumerate(CATS):
            row[f"theta_med_{c}"] = float(np.median(theta[:, i]))
            row[f"theta_p05_{c}"] = float(np.quantile(theta[:, i], 0.05))
            row[f"theta_p95_{c}"] = float(np.quantile(theta[:, i], 0.95))
        rows.append(row)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{len(df)} images -> {out}")
    print(df[[c for c in df.columns if c.startswith('theta_med_')]].mean().round(3))


if __name__ == "__main__":
    main()
