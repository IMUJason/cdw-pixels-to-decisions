"""CDW-Seg COCO -> YOLO-seg 标签（重叠类映射到 CODD 类名，用于域偏移评估）。

映射（仅保留与 CODD 重叠的 4 个类，其余类标注剔除）：
  CP -> concrete ; WT -> wood ; PB -> gypsum_board ; HP/SP -> plastic

用法:
    python scripts/cdwseg_to_yolo.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.codd_to_coco import CATS  # noqa: E402

ROOT = Path(r"D:\2026 cdw 0911 datasets\cdw_seg")
IMG_SRC = ROOT / "Ground_Truths_VOC_Format"  # COCO file_name 已含 JPEGImages/ 前缀
COCO = ROOT / "Ground_Truths_COCO_Format" / "annotations.json"
OUT = Path(r"D:\2026 cdw 0911 datasets\processed\cdwseg_yolo")

MAP = {"CP": "concrete", "WT": "wood", "PB": "gypsum_board",
       "HP": "plastic", "SP": "plastic"}


def poly_to_yolo(seg, w, h):
    pts = [(seg[i] / w, seg[i + 1] / h) for i in range(0, len(seg), 2)]
    return " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)


def main(split_seed=0):
    d = json.load(open(COCO, encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    ann_by_img = defaultdict(list)
    kept = dropped = 0
    for a in d["annotations"]:
        cname = next(c["name"] for c in d["categories"] if c["id"] == a["category_id"])
        if cname in MAP:
            ann_by_img[a["image_id"]].append((MAP[cname], a["segmentation"]))
            kept += 1
        else:
            dropped += 1
    print(f"annotations: kept(overlap)={kept}, dropped(non-overlap)={dropped}")

    # 划分：train 350 / val 40 / test 40（按文件名排序后随机）
    import random
    random.seed(split_seed)
    files = sorted(imgs.keys())
    random.shuffle(files)
    n = len(files)
    splits = {"train": files[: int(0.8 * n)], "val": files[int(0.8 * n): int(0.9 * n)],
              "test": files[int(0.9 * n):]}
    import shutil
    for sp, ids in splits.items():
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        for iid in ids:
            im = imgs[iid]
            lines = []
            for cname, segs in ann_by_img.get(iid, []):
                for seg in segs:  # COCO polygon 列表
                    if len(seg) >= 6:
                        lines.append(f"{CATS.index(cname)} {poly_to_yolo(seg, im['width'], im['height'])}")
            (OUT / "labels" / sp / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
            dst = OUT / "images" / sp / Path(im["file_name"]).name
            if not dst.exists():
                shutil.copy(IMG_SRC / im["file_name"], dst)
        print(f"{sp}: {len(ids)} images")
    yaml = OUT / "cdwseg_overlap.yaml"
    yaml.write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CATS)) + "\n")
    print("yaml ->", yaml)


if __name__ == "__main__":
    main()
