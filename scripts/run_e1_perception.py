"""E1 感知基准：CODD 实例分割训练（复现官方基线 + 升级配置）。

官方基线（CODD 自带 PDF）：YOLOv8n-seg @320 —— 测试集 BOX mAP50=0.860 / MASK mAP50=0.847
（MASK mAP50:95=0.633）。复现成功 = 我们的数据管线与评测协议与作者对齐。

用法:
    python scripts/run_e1_perception.py --model n --imgsz 320 --epochs 150 --tag repro_n320
    python scripts/run_e1_perception.py --model s --imgsz 640 --epochs 150 --tag upgrade_s640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(r"D:\2026 cdw 0911 datasets\processed\codd_yolo\codd.yaml")
RUNS_DIR = Path(r"D:\2026 cdw 0911 datasets\runs\e1")
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"  # plan2/weights/


def load_yolo(model: str) -> YOLO:
    w = WEIGHTS_DIR / f"yolov8{model}-seg.pt"
    return YOLO(str(w) if w.exists() else f"yolov8{model}-seg.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="n", choices=["n", "s", "m"], help="YOLO 规模")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default=None, help="运行名后缀")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--amp", action="store_true", help="启用 AMP（默认关闭以跳过需联网的 amp check）")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.tag or f"yolov8{args.model}seg_{args.imgsz}"
    model = load_yolo(args.model)  # 优先 plan2/weights/ 本地权重

    model.train(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        amp=args.amp,
        seed=0,
        workers=4,
        val=True,
    )

    # 训练完自动在测试集评估（best.pt）
    best = RUNS_DIR / name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(DATA_YAML), split="test", imgsz=args.imgsz)
    print(f"\n=== {name} TEST: box mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
          f"| mask mAP50={metrics.seg.map50:.4f} mAP50-95={metrics.seg.map:.4f} ===")
    per_cls = metrics.seg.maps  # 每类 mask mAP50-95
    names = metrics.names
    for i, apv in enumerate(per_cls):
        print(f"  {names[i]:14s} mask mAP50-95 = {apv:.4f}")


if __name__ == "__main__":
    main()
