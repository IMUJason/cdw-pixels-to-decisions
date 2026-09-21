"""CODD -> YOLO-seg 格式导出（零拷贝：junction 引用原图目录，仅生成标签 txt + yaml）。

用法:
    python -m src.vision.codd_to_yolo --codd-root "D:/2026 cdw 0911 datasets/codd/..." --out-root "D:/2026 cdw 0911 datasets/processed/codd_yolo"
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vision.codd_to_coco import CATS, parse_polygons  # noqa: E402

SPLITS = {"training": "train", "validation": "val", "testing": "test"}


def export_labels(xml_dir: str, lbl_dir: str):
    lbl_dir = Path(lbl_dir); lbl_dir.mkdir(parents=True, exist_ok=True)
    import glob
    n_img, n_obj = 0, 0
    for xp in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        fname, objs = parse_polygons(xp)
        w = h = None
        import xml.etree.ElementTree as ET
        r = ET.parse(xp).getroot()
        w = float(r.findtext(".//width")); h = float(r.findtext(".//height"))
        lines = []
        for o in objs:
            if o["label"] not in CATS:
                continue
            cid = CATS.index(o["label"])
            pts = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in o["polygon"])
            lines.append(f"{cid} {pts}")
        (lbl_dir / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
        n_img += 1; n_obj += len(lines)
    print(f"  {lbl_dir}: {n_img} labels, {n_obj} instances")


def make_junction(src: str, dst: str):
    dst = Path(dst)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(Path(src).resolve())],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mklink /J failed: {r.stderr or r.stdout}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codd-root", required=True)
    ap.add_argument("--out-root", default="D:/2026 cdw 0911 datasets/processed/codd_yolo")
    args = ap.parse_args()
    root = Path(args.codd_root)
    out = Path(args.out_root)
    for src_split, yolo_split in SPLITS.items():
        img_src = root / src_split
        make_junction(img_src, out / "images" / yolo_split)
        export_labels(str(img_src), str(out / "labels" / yolo_split))
    yaml = out / "codd.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CATS)) + "\n",
        encoding="utf-8")
    print("yaml ->", yaml)


if __name__ == "__main__":
    main()
