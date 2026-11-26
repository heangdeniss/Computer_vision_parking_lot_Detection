#!/usr/bin/env python3
"""Fine-tune Faster R-CNN on the same YOLO parking-space bounding box annotations.

Example:
    .venv/bin/python src/faster_rcnn/train.py --dry-run
    .venv/bin/python src/faster_rcnn/train.py
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.utils import YAML

from src.data import YoloParkingDataset, detection_collate, load_data_yaml
from src.faster_rcnn.model import BACKBONES, build_faster_rcnn
from src.metrics import evaluate_detector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "parking-space.v1i.yolov11/data.yaml")
    p.add_argument("--config", type=Path, default=ROOT / "configs/faster_rcnn.yaml")
    p.add_argument("--backbone", choices=sorted(BACKBONES))
    p.add_argument("--project", type=Path, default=ROOT / "runs/faster_rcnn")
    p.add_argument("--name", default="parking_frcnn")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="validate dataset, model & plan without training")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(project: Path, name: str, exist_ok: bool) -> Path:
    candidate = project / name
    if exist_ok or not candidate.exists():
        return candidate
    idx = 2
    while (project / f"{name}{idx}").exists():
        idx += 1
    return project / f"{name}{idx}"


def lr_factor(epoch: int, total_epochs: int, warmup_epochs: int, min_ratio: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_ratio + (1.0 - min_ratio) * cosine


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum: int,
    grad_clip_norm: float,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0,
              "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    steps = 0

    for step, (images, targets) in enumerate(loader, start=1):
        gpu_images = [img.to(device, non_blocking=True) for img in images]
        gpu_targets = [
            {"boxes": t["boxes"].to(device, non_blocking=True),
             "labels": t["labels"].to(device, non_blocking=True)}
            for t in targets
        ]
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(gpu_images, gpu_targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(loss.item())
        for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            if key in loss_dict:
                totals[key] += float(loss_dict[key].item())
        steps += 1

    return {k: v / max(steps, 1) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = YAML.load(args.config)
    data = load_data_yaml(args.data)

    backbone = args.backbone or cfg["backbone"]
    epochs = args.epochs if args.epochs is not None else int(cfg["epochs"])
    batch = args.batch if args.batch is not None else int(cfg["batch"])
    grad_accum = args.grad_accum if args.grad_accum is not None else int(cfg.get("grad_accum", 1))
    lr = args.lr if args.lr is not None else float(cfg["lr"])
    min_lr = float(cfg.get("min_lr", lr * 0.05))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    warmup_epochs = int(cfg.get("warmup_epochs", 2))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 10.0))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))
    patience = args.patience if args.patience is not None else int(cfg.get("patience", 10))
    min_size = int(cfg.get("min_size", 640))
    max_size = int(cfg.get("max_size", 800))
    box_nms_thresh = float(cfg.get("box_nms_thresh", 0.6))
    box_detections_per_img = int(cfg.get("box_detections_per_img", 300))
    use_amp = bool(cfg.get("amp", True)) and not args.no_amp and torch.device(args.device).type == "cuda"
    seed_everything(int(cfg.get("seed", 42)))

    train_ds = YoloParkingDataset(
        data["train"], names=data["names"], augment=True,
        fliplr=float(cfg.get("fliplr", 0.5)), color_jitter=cfg.get("color_jitter"),
    )
    val_ds = YoloParkingDataset(data["val"], names=data["names"], augment=False)

    num_classes = data["nc"] + 1
    model = build_faster_rcnn(
        backbone=backbone, num_classes=num_classes, pretrained=True,
        min_size=min_size, max_size=max_size,
        box_nms_thresh=box_nms_thresh, box_detections_per_img=box_detections_per_img,
    )
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"Faster R-CNN parking detector | backbone={backbone}")
    print(f"Dataset: {args.data} | nc={data['nc']} names={data['names']} | train={len(train_ds)} valid={len(val_ds)}")
    print(f"Params: {params:,} total ({trainable:,} trainable)")
    print(f"Schedule: {epochs} epochs | batch={batch} x grad_accum={grad_accum} | lr={lr} -> {min_lr} | amp={use_amp}")
    print("=" * 80)

    if args.dry_run:
        img, tgt = train_ds[0]
        print(f"Sample[0]: image={tuple(img.shape)}, boxes={tuple(tgt['boxes'].shape)}, labels={tgt['labels'][:6].tolist()}")
        print("Dry run: dataset, YOLO->xyxy box conversion, and Faster R-CNN model verified.")
        return

    device = torch.device(args.device)
    model.to(device)

    run_dir = make_run_dir(args.project, args.name, exist_ok=args.exist_ok)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                              pin_memory=pin, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=pin, collate_fn=detection_collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    fieldnames = ["epoch", "time", "lr", "train_loss", "loss_cls", "loss_box",
                  "loss_obj", "loss_rpn", "val_precision", "val_recall", "val_map50", "val_map50_95"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_map, best_epoch, stale_epochs = -1.0, 0, 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        cur_lr = lr * lr_factor(epoch - 1, epochs, warmup_epochs, min_lr / lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        losses = train_one_epoch(model, train_loader, optimizer, scaler, device, grad_accum, grad_clip_norm, use_amp)
        val_metrics = evaluate_detector(model, val_loader, device, data["names"], amp=use_amp)
        print(
            f"Epoch {epoch:>2}/{epochs} ({time.time() - ep_start:5.1f}s) | lr={cur_lr:.6f} | "
            f"loss={losses['loss']:.4f} | val P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"mAP50={val_metrics['map50']:.4f} mAP50-95={val_metrics['map50_95']:.4f}",
            flush=True,
        )

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "epoch": epoch, "time": round(time.time() - t_start, 2), "lr": round(cur_lr, 8),
                "train_loss": round(losses["loss"], 5), "loss_cls": round(losses["loss_classifier"], 5),
                "loss_box": round(losses["loss_box_reg"], 5), "loss_obj": round(losses["loss_objectness"], 5),
                "loss_rpn": round(losses["loss_rpn_box_reg"], 5),
                "val_precision": round(val_metrics["precision"], 5), "val_recall": round(val_metrics["recall"], 5),
                "val_map50": round(val_metrics["map50"], 5), "val_map50_95": round(val_metrics["map50_95"], 5),
            })

        checkpoint = {
            "model_state_dict": model.state_dict(), "backbone": backbone, "names": data["names"],
            "nc": data["nc"], "min_size": min_size, "max_size": max_size,
            "box_nms_thresh": box_nms_thresh, "box_detections_per_img": box_detections_per_img,
            "epoch": epoch,
            "val_metrics": {k: val_metrics[k] for k in ("precision", "recall", "map50", "map50_95")},
        }
        torch.save(checkpoint, weights_dir / "last.pt")

        if val_metrics["map50_95"] > best_map:
            best_map, best_epoch, stale_epochs = val_metrics["map50_95"], epoch, 0
            torch.save(checkpoint, weights_dir / "best.pt")
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no mAP50-95 gain for {patience} epochs).")
                break

    print(f"Finished in {(time.time() - t_start) / 60:.1f} min | best val mAP50-95={best_map:.4f} (epoch {best_epoch})")
    print(f"Best checkpoint: {weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
