#!/usr/bin/env python3
"""Fine-tune YOLOv11 on the parking-space dataset.

Example:
    python src/train.py --weights yolo11s.pt

Weights are transferred from the pretrained checkpoint; the class-score branch is
re-initialized for the 2 parking classes. Nothing runs unless you execute this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "parking-space.v1i.yolov11/data.yaml")
    p.add_argument("--config", type=Path, default=ROOT / "configs/yolo11_parking.yaml")
    p.add_argument("--hyp", type=Path, default=ROOT / "configs/train_hyp.yaml")
    p.add_argument("--weights", default="yolo11s.pt", help="pretrained checkpoint (yolo11s.pt / yolo11m.pt / local .pt)")
    p.add_argument("--project", default=str(ROOT / "runs/detect"))
    p.add_argument("--name", default="parking_yolo11s")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device")
    p.add_argument("--dry-run", action="store_true", help="validate configs/weights and print the plan, do not train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data, cfg, hyp = YAML.load(args.data), YAML.load(args.config), YAML.load(args.hyp)

    if cfg["nc"] != data["nc"]:
        raise SystemExit(f"nc mismatch: model config has {cfg['nc']}, dataset has {data['nc']}")
    print(f"Dataset: {args.data}")
    print(f"  nc={data['nc']} names={data['names']} (0=empty, 1=occupied)")
    print(f"Architecture: {args.config} (nc={cfg['nc']}, scale=s)")
    print(f"Hyperparameters: {args.hyp} ({len(hyp)} overrides)")
    print(f"Pretrained weights: {args.weights}")

    model = YOLO(str(args.config), verbose=False)
    model.load(args.weights)
    model.model.names = data["names"]  # keep class names consistent for plots and cls-head remap

    if args.dry_run:
        print("\nDry run: model built and weights transferred successfully. Training not started.")
        return

    overrides = {k: v for k, v in vars(args).items()
                 if k in {"epochs", "batch", "imgsz", "device"} and v is not None}
    model.train(cfg=str(args.hyp), data=str(args.data), project=args.project, name=args.name,
                exist_ok=False, **overrides)


if __name__ == "__main__":
    main()
