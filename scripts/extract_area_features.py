"""预计算逐图逐类面积特征（预测 + 真值）——E3 视觉特征接入用。

输出两个 CSV（行=图像）：
- pred_areas_{split}.csv：s@640 预测掩码的逐类像素面积（10 列）
- gt_areas_{split}.csv：真值多边形的逐类像素面积（10 列，shoelace）

预测面积与真值面积的比率结构 = 感知噪声的真实分布（替代此前的对数正态模拟）。

用法:
    python scripts/extract_area_features.py --weights <best.pt> --imgsz 640
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.codd_to_coco import CATS  # noqa: E402
from src.vision.composition_estimator import image_class_areas  # noqa: E402

IMG_ROOT = Path(r"D:\2026 cdw 0911 datasets\processed\codd_yolo\images")
OUT_DIR = Path(r"D:\2026 cdw 0911 datasets\processed\area_features")
XML_ROOT = Path(r"D:\2026 cdw 0911 datasets\codd\Construction and Demolition Waste Object Detection Dataset  (CODD)")


def gt_areas(split: str) -> pd.DataFrame:
    import glob
    rows = []
    for xp in sorted(glob.glob(str(XML_ROOT / split / "*.xml"))):
        areas = image_class_areas(xp)  # m²（已乘标定）
        rows.append({"image": Path(xp).stem,
                     **{c: areas.get(c, 0.0) for c in CATS}})
    return pd.DataFrame(rows)


def pred_areas(weights: str, split: str, imgsz: int, conf: float) -> pd.DataFrame:
    from ultralytics import YOLO
    model = YOLO(weights)
    rows = []
    imgs = sorted(str(p) for p in (IMG_ROOT / split).glob("*.jpg"))
    for k, img in enumerate(imgs):
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=0)[0]
        areas = {c: 0.0 for c in CATS}
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            scale = r.orig_shape[0] / masks.shape[1]
            for i, cid in enumerate(r.boxes.cls.cpu().numpy().astype(int)):
                areas[r.names[cid]] += float(masks[i].sum() * scale * scale)
        rows.append({"image": Path(img).stem, **areas})
        if (k + 1) % 200 == 0:
            print(f"{split}: {k+1}/{len(imgs)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r"D:\2026 cdw 0911 datasets\runs\e1\upgrade_s640\weights\best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        src_split = {"train": "training", "test": "testing"}[split]
        if not (OUT_DIR / f"gt_areas_{split}.csv").exists():
            gt = gt_areas(src_split)
            gt.to_csv(OUT_DIR / f"gt_areas_{split}.csv", index=False)
            print(f"{split}: gt areas {len(gt)} rows")
        pr = pred_areas(args.weights, split, args.imgsz, args.conf)  # YOLO 目录名
        pr.to_csv(OUT_DIR / f"pred_areas_{split}.csv", index=False)
        print(f"{split}: pred areas {len(pr)} rows")


if __name__ == "__main__":
    main()
