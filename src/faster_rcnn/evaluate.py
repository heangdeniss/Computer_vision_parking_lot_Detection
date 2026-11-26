#!/usr/bin/env python3
"""Evaluate a trained Faster R-CNN checkpoint on test or val split.

Example:
    .venv/bin/python src/faster_rcnn/evaluate.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data import YoloParkingDataset, detection_collate, load_data_yaml
from src.faster_rcnn.model import load_faster_rcnn_checkpoint
from src.metrics import evaluate_detector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "runs/faster_rcnn/parking_frcnn/weights/best.pt")
    p.add_argument("--data", type=Path, default=ROOT / "parking-space.v1i.yolov11/data.yaml")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--conf", type=float, default=0.001, help="confidence threshold (0.001 = standard mAP)")
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--project", type=Path, default=ROOT / "runs/faster_rcnn/eval")
    args = p.parse_args()

    data = load_data_yaml(args.data)
    device = torch.device(args.device)
    model, meta = load_faster_rcnn_checkpoint(
        args.weights, device=device, box_score_thresh=args.conf, box_nms_thresh=args.iou
    )
    ds = YoloParkingDataset(data[args.split], names=data["names"], augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"), collate_fn=detection_collate,
    )

    out_dir = args.project / args.split
    metrics = evaluate_detector(
        model, loader, device=device, names=data["names"], conf_eval=args.conf, plots_dir=out_dir
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"Evaluation (Faster R-CNN {meta['backbone']}): {args.weights} | split={args.split}")
    print("=" * 78)
    print(f"Overall  mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map50_95']:.4f}  "
          f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
          f"speed={metrics['ms_per_image']:.1f} ms/img")

    print("\nPer-class metrics (0 = empty, 1 = occupied)")
    print(f"{'id':>2}  {'class':<12} {'P':>7} {'R':>7} {'mAP50':>8} {'mAP50-95':>9} {'CM recall':>10} {'instances':>10}")
    print("-" * 78)
    for cls_name in data["names"]:
        row = metrics["per_class"].get(cls_name)
        if not row:
            continue
        cm_r = f"{row['cm_recall']:.3f}" if row["cm_recall"] is not None else "n/a"
        print(f"{row['class_id']:>2}  {cls_name:<12} {row['precision']:>7.4f} {row['recall']:>7.4f} "
              f"{row['map50']:>8.4f} {row['map50_95']:>9.4f} {cm_r:>10} {row['instances']:>10}")
    print(f"\nCross-class swaps @ conf=0.25: {metrics['cross_class_swaps']} | saved to {out_dir}")


if __name__ == "__main__":
    main()
