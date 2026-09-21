"""预测掩码 -> 组成/总量分布 的端到端推理管线（研究主链路）。

与 composition_estimator.py（真值标注驱动，用于伪真值）不同，本模块用训练好的
分割模型在原始图像上预测掩码，再走同一套物理先验（material_priors）输出
组成 θ 与总量 D 的 Monte Carlo 分布——即"感知层+不确定性层"的推理实现。

不确定性来源（可组合）：
1. 密度/厚度先验区间（同伪真值）
2. 分割掩码的像素面积噪声（乘性对数正态，σ 可调——由 E1 的面积误差标定）

用法:
    python -m src.vision.vision_composition --weights <best.pt> --source <img dir> --out csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.material_priors import CODD_PRIORS, CODD_METERS_PER_PIXEL  # noqa: E402
from src.vision.codd_to_coco import CATS  # noqa: E402


def predict_class_areas(model, img_path: str, imgsz: int, conf: float):
    """返回 {class: 掩码像素面积(原图尺度)}。"""
    r = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False, device=0)[0]
    areas = {}
    if r.masks is None:
        return areas, 0
    masks = r.masks.data.cpu().numpy()            # [N, h, w]（letterbox 后）
    scale = r.orig_shape[0] / masks.shape[1]      # 还原到原图的尺度因子
    for i, cid in enumerate(r.boxes.cls.cpu().numpy().astype(int)):
        name = r.names[cid]
        areas[name] = areas.get(name, 0.0) + masks[i].sum() * scale * scale
    return areas, len(masks)


def areas_to_distribution(areas_px: dict, n_mc: int, rng, mask_sigma: float = 0.15):
    """面积(px²) -> (θ 样本 [n_mc,K], D 样本 [n_mc])，掩码面积乘性噪声 + 先验 MC。"""
    m2pp = CODD_METERS_PER_PIXEL
    present = [c for c in CATS if areas_px.get(c, 0) > 0]
    if not present:
        return np.zeros((n_mc, len(CATS))), np.zeros(n_mc)
    A_px = np.array([areas_px[c] for c in present])
    # 掩码面积不确定性：乘性对数正态
    A = A_px[None, :] * np.exp(rng.normal(0, mask_sigma, size=(n_mc, len(present))))
    A = A * (m2pp ** 2)
    mass = np.zeros((n_mc, len(present)))
    for j, c in enumerate(present):
        p = CODD_PRIORS[c]
        lo, hi = p.density_lo * p.thick_lo, p.density_hi * p.thick_hi
        mass[:, j] = A[:, j] * rng.uniform(lo, hi, size=n_mc)
    theta = mass / mass.sum(axis=1, keepdims=True)
    full = np.zeros((n_mc, len(CATS)))
    for j, c in enumerate(present):
        full[:, CATS.index(c)] = theta[:, j]
    return full, mass.sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="图像目录")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--mask-sigma", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    rng = np.random.default_rng(args.seed)
    rows = []
    imgs = sorted(str(p) for p in Path(args.source).glob("*.jpg"))
    for k, img in enumerate(imgs):
        areas, n_det = predict_class_areas(model, img, args.imgsz, args.conf)
        theta, D = areas_to_distribution(areas, args.n_mc, rng, args.mask_sigma)
        row = {"image": Path(img).stem, "n_det": n_det,
               "D_med": float(np.median(D)),
               "D_lo": float(np.quantile(D, 0.05)),
               "D_hi": float(np.quantile(D, 0.95))}
        for i, c in enumerate(CATS):
            row[f"theta_med_{c}"] = float(np.median(theta[:, i]))
            row[f"theta_p05_{c}"] = float(np.quantile(theta[:, i], 0.05))
            row[f"theta_p95_{c}"] = float(np.quantile(theta[:, i], 0.95))
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(imgs)}", flush=True)
    import pandas as pd
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} images -> {out}")


if __name__ == "__main__":
    main()
