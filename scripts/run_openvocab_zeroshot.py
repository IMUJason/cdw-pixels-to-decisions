"""B1：开放词汇零样本迁移（CODD→CDW-Seg 的语义因素解耦）。

闭集 YOLOv8 零样本崩溃（mAP50=0.008）的主因之一是类别错配（CODD 的
brick/tile/gypsum vs CDW-Seg 的 timber/fill dirt/steel/fabric）。
用 YOLO-World（ultralytics 内置，文本提示即类别）在同一测试集上评估：
若开放词汇显著高于闭集崩溃值 => 语义因素被证实可解，几何/光度域隙剩余。

评估：box AP50（自实现 IoU 贪心匹配 + 连续 AP），GT 取 LabelMe json 全 9 类。

用法: python scripts/run_openvocab_zeroshot.py
"""
import json
from pathlib import Path

import numpy as np

CDWSEG = Path(r"D:\2026 cdw 0911 datasets\cdw_seg\Original_Images_and_Annotation_Files")
TEST_IMG_DIR = Path(r"D:\2026 cdw 0911 datasets\processed\cdwseg_fs50\images\test")

CLASSES = ["CP", "FD", "WT", "HP", "SP", "ST", "FB", "CB", "PB"]
PROMPTS = {
    "CP": "concrete chunk", "FD": "dirt soil pile", "WT": "wood timber plank",
    "HP": "hard plastic piece", "SP": "plastic bag", "ST": "steel rebar metal",
    "FB": "fabric textile cloth", "CB": "cardboard box", "PB": "plasterboard drywall sheet",
}


def gt_boxes(img_stem):
    """GT 框。实测：ultralytics predict 输出已应用 EXIF（竖构图坐标系），
    与 LabelMe 标注坐标系一致——直接使用原坐标，无需旋转（2026-09-16 两次
    实验确认：旋转/不旋转 AP 均≈0，瓶颈在模型而非坐标系）。"""
    d = json.load(open(CDWSEG / f"{img_stem}.json"))
    out = []
    for s in d["shapes"]:
        if s["label"] not in PROMPTS or s["shape_type"] not in ("polygon", None):
            continue
        p = np.asarray(s["points"], float)
        out.append((s["label"], float(p[:, 0].min()), float(p[:, 1].min()),
                    float(p[:, 0].max()), float(p[:, 1].max())))
    return out


def iou(a, b):
    x1, y1 = max(a[1], b[1]), max(a[2], b[2])
    x2, y2 = min(a[3], b[3]), min(a[4], b[4])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(scores, n_gt):
    """连续 AP（按分数降序，IoU≥0.5 判 TP，贪心）。"""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, float)[:, 0])
    tp = np.zeros(len(order)); fp = np.zeros(len(order))
    for rank, i in enumerate(order):
        tp[rank] = 1.0 if scores[i][1] else 0.0
        fp[rank] = 1.0 - tp[rank]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def main():
    from ultralytics import YOLOWorld
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes([PROMPTS[c] for c in CLASSES])
    stems = sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpg"))
    print(f"test images: {len(stems)}")
    per_class = {c: [] for c in CLASSES}   # (score, is_tp)
    n_gt = {c: 0 for c in CLASSES}
    for stem in stems:
        gts = gt_boxes(stem)
        r = model(str(CDWSEG / f"{stem}.jpg"), conf=0.02, imgsz=1280, verbose=False)[0]
        preds = []
        if r.boxes is not None:
            for bb, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                preds.append((CLASSES[int(cls)], float(cf),
                              float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])))
        for c in CLASSES:
            n_gt[c] += sum(1 for g in gts if g[0] == c)
        used = set()
        for p in sorted([p for p in preds], key=lambda t: -t[1]):
            best, bi = 0.0, -1
            for gi, g in enumerate(gts):
                if gi in used or g[0] != p[0]:
                    continue
                v = iou((0, p[2], p[3], p[4], p[5]), g)
                if v > best:
                    best, bi = v, gi
            if best >= 0.5:
                used.add(bi)
                per_class[p[0]].append((p[1], True))
            else:
                per_class[p[0]].append((p[1], False))
    print("\n=== YOLO-World zero-shot (box AP50, 9 类) ===")
    aps = []
    for c in CLASSES:
        a = ap50(per_class[c], n_gt[c])
        aps.append(a)
        print(f"{c:3s} {PROMPTS[c]:24s} nGT={n_gt[c]:4d}  AP50={a:.3f}")
    valid = [a for a in aps if not np.isnan(a)]
    print(f"mean AP50 (present classes) = {np.mean(valid):.4f}")


if __name__ == "__main__":
    main()
