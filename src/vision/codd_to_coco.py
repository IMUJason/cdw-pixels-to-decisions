"""CODD (VOC XML + LabelMe-style polygons) -> COCO json 转换器。

CODD 特殊性：
- 多边形存于 <object><polygon><x1><y1>...<xN><yN></polygon>（非 VOC 标准 <segmentation>）
- <segmented>0</segmented> 字段不可信（实际 100% 实例有多边形）
- 无多边形的实例退化为 bbox 四点多边形

用法:
    python -m src.vision.codd_to_coco --codd-root "D:/2026 cdw 0911 datasets/codd/..." --out-dir data/processed/codd_coco
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

CATS = [
    "brick", "concrete", "tile", "wood", "gypsum_board",
    "foam", "general_w", "stone", "plastic", "pipes",
]  # 与 CODD 官方 XML name 一致（字母序无关，id 从 1 起）


def parse_polygons(xml_path: str):
    """返回 (filename, [ {label, polygon:[[x,y],...], bbox:[x,y,w,h]} ])"""
    root = ET.parse(xml_path).getroot()
    fname = root.findtext("filename") or Path(xml_path).with_suffix(".jpg").name
    objs = []
    for o in root.findall("object"):
        label = o.findtext("name")
        if label is None or not label.strip():
            continue  # CODD 偶发空 object（Demetriou 同款噪声，过滤）
        label = label.strip()
        bb = o.find("bndbox")
        bbox = None
        if bb is not None:
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            bbox = [x1, y1, x2 - x1, y2 - y1]
        pg = o.find("polygon")
        pts = []
        if pg is not None:
            i = 1
            while True:
                xv, yv = pg.findtext(f"x{i}"), pg.findtext(f"y{i}")
                if xv is None or yv is None:
                    break
                pts.append([float(xv), float(yv)])
                i += 1
        if len(pts) < 3 and bbox is not None:
            # 退化：用 bbox 四角
            x, y, w, h = bbox
            pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        if len(pts) >= 3:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if bbox is None:
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            objs.append({"label": label, "polygon": pts, "bbox": bbox})
    return fname, objs


def build_split(img_dir: str, split: str, out_path: str, cat2id: dict):
    images, annotations = [], []
    ann_id = 1
    xmls = sorted(glob.glob(os.path.join(img_dir, "*.xml")))
    for img_id, xp in enumerate(xmls, start=1):
        fname, objs = parse_polygons(xp)
        # 统一尺寸 1920x1200（CODD 全部一致；仍从 XML size 读取以防例外）
        r = ET.parse(xp).getroot()
        w = int(float(r.findtext(".//width"))); h = int(float(r.findtext(".//height")))
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for o in objs:
            flat = [c for p in o["polygon"] for c in p]
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat2id[o["label"]],
                "segmentation": [flat],
                "bbox": [round(v, 2) for v in o["bbox"]],
                "area": o["bbox"][2] * o["bbox"][3],  # bbox 面积（与 YOLO-seg 官方评测口径一致）
                "iscrowd": 0,
            })
            ann_id += 1
    coco = {
        "info": {"description": f"CODD {split} (converted from VOC polygons)"},
        "licenses": [{"name": "CC BY 4.0"}],
        "categories": [{"id": i, "name": c} for c, i in cat2id.items()],
        "images": images, "annotations": annotations,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out_path, "w", encoding="utf-8"))
    print(f"{split}: {len(images)} images, {len(annotations)} instances -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codd-root", required=True, help="CODD 根目录（含 training/validation/testing）")
    ap.add_argument("--out-dir", default="data/processed/codd_coco")
    args = ap.parse_args()
    cat2id = {c: i + 1 for i, c in enumerate(CATS)}
    for split in ["training", "validation", "testing"]:
        build_split(os.path.join(args.codd_root, split), split,
                    os.path.join(args.out_dir, f"{split}.json" if split != "validation" else "val.json"),
                    cat2id)
    # ultralytics 需要 {split}/images + labels，另行生成；先输出 COCO 供评测/通用管线


if __name__ == "__main__":
    main()
