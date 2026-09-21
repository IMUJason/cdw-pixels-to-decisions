"""E4 补充：零样本 CODD→CDW-Seg 在不同推理分辨率下的掩码 mAP50。

12MP 图在 640 推理时有效分辨率损失 ~4.7×，细薄物料（织物/软塑料）被抹除。
对比 imgsz ∈ {640, 1280}，其余协议与 E4 主实验一致（同一 43 图测试集）。

用法: python scripts/run_zeroshot_resolution.py
"""
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = r"D:\2026 cdw 0911 datasets\runs\e1\upgrade_s640\weights\best.pt"
DATA = r"D:\2026 cdw 0911 datasets\processed\cdwseg_fs50\cdwseg_fs.yaml"

for imgsz in (640, 1280):
    r = YOLO(WEIGHTS).val(data=DATA, split="test", imgsz=imgsz, batch=2, workers=0)
    print(f"=== zero-shot CODD->CDW-Seg @ imgsz={imgsz} ===")
    print(f"BOX mAP50={r.box.map50:.4f} | MASK mAP50={r.seg.map50:.4f} "
          f"mAP50-95={r.seg.map:.4f}")
    for i, ap_ in enumerate(r.seg.maps):
        if r.names[i] in ("concrete", "wood", "gypsum_board", "plastic"):
            print(f"  {r.names[i]:14s} mask mAP50-95={ap_:.4f}")
