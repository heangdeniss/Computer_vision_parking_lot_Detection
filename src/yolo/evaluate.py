#!/usr/bin/env python3
"""Evaluate a trained YOLOv11 checkpoint on the test (or val) split.

Example:
    .venv/bin/python src/yolo/evaluate.py --weights runs/detect/parking_yolo11s/weights/best.pt --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(ROOT / "runs/detect/parking_yolo11s/weights/best.pt"))
    p.add_argument("--data", type=Path, default=ROOT / "parking-space.v1i.yolov11/data.yaml")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device")
    p.add_argument("--conf", type=float, default=0.001, help="confidence threshold (0.001 = standard mAP)")
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--project", default=str(ROOT / "runs/detect/eval"))
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                        device=args.device, conf=args.conf, iou=args.iou,
                        project=args.project, name=args.split, plots=True)

    names = metrics.names
    print("\n" + "=" * 78)
    print(f"Evaluation (YOLOv11): {args.weights} | split={args.split}")
    print("=" * 78)
    print(f"Overall  mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
          f"precision={metrics.box.mp:.4f}  recall={metrics.box.mr:.4f}")

    print("\nPer-class metrics (0 = empty, 1 = occupied)")
    print(f"{'id':>2}  {'class':<12} {'P':>7} {'R':>7} {'mAP50':>8} {'mAP50-95':>9} {'CM recall':>10} {'instances':>10}")
    print("-" * 78)
    cm = getattr(metrics.confusion_matrix, "matrix", None)
    for i, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        precision, recall, ap50, ap50_95 = metrics.box.class_result(i)
        cm_recall = "n/a"
        if cm is not None and class_id < cm.shape[0] - 1:
            col = cm[:-1, class_id].sum()
            cm_recall = f"{cm[class_id, class_id] / col:.3f}" if col else "n/a"
        n = int(metrics.nt_per_class[class_id]) if metrics.nt_per_class is not None else -1
        print(f"{class_id:>2}  {names.get(class_id, '?'):<12} {precision:>7.4f} {recall:>7.4f} "
              f"{ap50:>8.4f} {ap50_95:>9.4f} {cm_recall:>10} {n:>10}")


if __name__ == "__main__":
    main()
