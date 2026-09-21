"""B1：CDW-Seg 少样本微调——域偏移可恢复性曲线。

从 CODD 预训练权重出发，分别用 N=20/50 张 CDW-Seg 训练图微调，
固定同一 43 图测试集评估。零样本基线：MASK mAP50=0.008。

用法: python scripts/cdwseg_fewshot.py --n-shot 20
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

SRC = Path(r"D:\2026 cdw 0911 datasets\processed\cdwseg_yolo")
WEIGHTS = r"D:\2026 cdw 0911 datasets\runs\e1\upgrade_s640\weights\best.pt"
RUNS = Path(r"D:\2026 cdw 0911 datasets\runs\cdwseg_fewshot")


def make_subset(n: int) -> Path:
    out = SRC.parent / f"cdwseg_fs{n}"
    if out.exists():
        return out
    for sp in ("train", "val", "test"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    train_imgs = sorted((SRC / "images" / "train").glob("*.jpg"))
    picked = train_imgs[:n]
    for p in picked:
        shutil.copy(p, out / "images" / "train" / p.name)
        lp = SRC / "labels" / "train" / (p.stem + ".txt")
        if lp.exists():
            shutil.copy(lp, out / "labels" / "train" / lp.name)
    for sp in ("val", "test"):
        for p in (SRC / "images" / sp).glob("*.jpg"):
            shutil.copy(p, out / "images" / sp / p.name)
        for lp in (SRC / "labels" / sp).glob("*.txt"):
            shutil.copy(lp, out / "labels" / sp / lp.name)
    names = (SRC / "cdwseg_overlap.yaml").read_text().split("names:")[1]
    (out / "cdwseg_fs.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:{names}")
    print(f"subset fs{n}: {n} train imgs")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shot", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    out = make_subset(args.n_shot)
    RUNS.mkdir(parents=True, exist_ok=True)
    tag = f"fs{args.n_shot}" + (f"_i{args.imgsz}" if args.imgsz != 640 else "")
    m = YOLO(WEIGHTS)
    m.train(data=str(out / "cdwseg_fs.yaml"), imgsz=args.imgsz, epochs=args.epochs,
            batch=4, workers=0, amp=False, seed=0, project=str(RUNS),
            name=tag, exist_ok=True, patience=30)
    r = YOLO(str(RUNS / tag / "weights" / "best.pt")).val(
        data=str(out / "cdwseg_fs.yaml"), split="test", imgsz=args.imgsz, batch=4, workers=0)
    print(f"=== CDW-Seg few-shot N={args.n_shot} imgsz={args.imgsz} TEST ===")
    print(f"BOX mAP50={r.box.map50:.4f} | MASK mAP50={r.seg.map50:.4f} mAP50-95={r.seg.map:.4f}")
    for i, ap_ in enumerate(r.seg.maps):
        if r.names[i] in ("concrete", "wood", "gypsum_board", "plastic"):
            print(f"  {r.names[i]:14s} mask mAP50-95={ap_:.4f}")


if __name__ == "__main__":
    main()
