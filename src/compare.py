#!/usr/bin/env python3
"""Side-by-side benchmark comparing YOLOv11 and Faster R-CNN on the same split.

Example:
    .venv/bin/python src/compare.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from ultralytics import YOLO

from src.data import YoloParkingDataset, detection_collate, load_data_yaml
from src.faster_rcnn.model import load_faster_rcnn_checkpoint
from src.metrics import evaluate_detector


def eval_yolo(
    weights: Path, data_yaml: Path, split: str, device: str, imgsz: int, iou: float
) -> dict[str, Any]:
    model = YOLO(str(weights), verbose=False)
    yolo_device = 0 if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=8,
        device=yolo_device, conf=0.001, iou=iou, plots=False, verbose=False,
    )
    names = metrics.names
    per_class: dict[str, dict[str, Any]] = {}
    cm = getattr(metrics.confusion_matrix, "matrix", None)
    for i, cid in enumerate(metrics.box.ap_class_index):
        cid = int(cid)
        p, r, ap50, ap50_95 = metrics.box.class_result(i)
        col = cm[:-1, cid].sum() if (cm is not None and cid < cm.shape[0] - 1) else 0
        per_class[names[cid]] = {
            "class_id": cid, "precision": float(p), "recall": float(r),
            "map50": float(ap50), "map50_95": float(ap50_95),
            "cm_recall": float(cm[cid, cid] / col) if col else None,
            "instances": int(metrics.nt_per_class[cid]),
        }

    nc = len(names)
    swaps = int(cm[:nc, :nc].sum() - np.trace(cm[:nc, :nc])) if cm is not None else 0

    data = load_data_yaml(data_yaml)
    ds = YoloParkingDataset(data[split], names=data["names"], augment=False)
    pil_images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in ds.image_paths]

    infer_model = YOLO(str(weights), verbose=False).to(yolo_device).fuse()
    if yolo_device != "cpu":
        infer_model.model.half()
    infer_model.predict(pil_images[0], imgsz=imgsz, conf=0.25, iou=iou, verbose=False)
    if torch.cuda.is_available() and yolo_device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for img in pil_images:
        infer_model.predict(img, imgsz=imgsz, conf=0.25, iou=iou, verbose=False)
    if torch.cuda.is_available() and yolo_device != "cpu":
        torch.cuda.synchronize()
    ms_per_img = round(((time.perf_counter() - t0) / len(pil_images)) * 1000.0, 2)

    return {
        "name": "YOLOv11s", "weights": str(weights),
        "params": sum(p.numel() for p in model.model.parameters()),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "ms_per_image": ms_per_img, "cross_class_swaps": swaps, "per_class": per_class,
    }


def eval_frcnn(
    weights: Path, data_yaml: Path, split: str, device: str, iou: float
) -> dict[str, Any]:
    data = load_data_yaml(data_yaml)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, meta = load_faster_rcnn_checkpoint(weights, device=dev, box_score_thresh=0.001, box_nms_thresh=iou)
    ds = YoloParkingDataset(data[split], names=data["names"], augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=detection_collate)
    metrics = evaluate_detector(model, loader, device=dev, names=data["names"], conf_eval=0.001)

    model.roi_heads.score_thresh = 0.25
    use_amp = dev.type == "cuda"
    tensors = [ds[i][0].to(dev) for i in range(len(ds))]
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        model([tensors[0]])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in tensors:
            model([t])
        if dev.type == "cuda":
            torch.cuda.synchronize()
    ms_per_img = round(((time.perf_counter() - t0) / len(tensors)) * 1000.0, 2)

    return {
        "name": f"Faster R-CNN ({meta['backbone']})", "weights": str(weights), "backbone": meta["backbone"],
        "params": sum(p.numel() for p in model.parameters()),
        "precision": metrics["precision"], "recall": metrics["recall"],
        "map50": metrics["map50"], "map50_95": metrics["map50_95"],
        "ms_per_image": ms_per_img, "cross_class_swaps": metrics["cross_class_swaps"],
        "per_class": metrics["per_class"],
    }


def save_comparison_chart(yolo: dict[str, Any], frcnn: dict[str, Any], path: Path, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_keys = [("mAP50", "map50"), ("mAP50-95", "map50_95"), ("Precision", "precision"), ("Recall", "recall")]
    labels = [m[0] for m in metrics_keys]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    x, w = np.arange(len(labels)), 0.35
    b1 = ax1.bar(x - w / 2, [yolo[m[1]] for m in metrics_keys], w, label=yolo["name"], color="#2f81f7")
    b2 = ax1.bar(x + w / 2, [frcnn[m[1]] for m in metrics_keys], w, label=frcnn["name"], color="#f59e0b")
    ax1.set_ylim(0, 1.08)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Score")
    ax1.set_title(f"Detection Accuracy ({split} split)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    speed_bars = ax2.bar([yolo["name"], "Faster R-CNN"], [yolo["ms_per_image"], frcnn["ms_per_image"]],
                         color=["#2f81f7", "#f59e0b"], width=0.5)
    ax2.set_ylabel("Latency (ms / image, lower is faster)")
    ax2.set_title("Inference Latency (batch=1)")
    ax2.grid(axis="y", alpha=0.25)
    for bar in speed_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f} ms", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-weights", type=Path, default=ROOT / "runs/detect/parking_yolo11s/weights/best.pt")
    p.add_argument("--frcnn-weights", type=Path, default=ROOT / "runs/faster_rcnn/parking_frcnn/weights/best.pt")
    p.add_argument("--data", type=Path, default=ROOT / "parking-space.v1i.yolov11/data.yaml")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=ROOT / "runs/compare")
    args = p.parse_args()

    yolo = eval_yolo(args.yolo_weights, args.data, args.split, args.device, args.imgsz, args.iou)
    frcnn = eval_frcnn(args.frcnn_weights, args.data, args.split, args.device, args.iou)

    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps({"split": args.split, "yolo11": yolo, "faster_rcnn": frcnn}, indent=2), encoding="utf-8"
    )
    save_comparison_chart(yolo, frcnn, out_dir / "comparison.png", args.split)

    print("\n" + "=" * 84)
    print(f"Side-by-Side Model Comparison | split={args.split} | same YOLO bounding boxes")
    print("=" * 84)
    print(f"{'Metric':<24} {'YOLOv11s':>18} {'Faster R-CNN':>18} {'Delta (F - Y)':>18}")
    print("-" * 84)
    for label, yv, fv, dv in [
        ("mAP50", f"{yolo['map50']:.4f}", f"{frcnn['map50']:.4f}", f"{frcnn['map50'] - yolo['map50']:+.4f}"),
        ("mAP50-95", f"{yolo['map50_95']:.4f}", f"{frcnn['map50_95']:.4f}", f"{frcnn['map50_95'] - yolo['map50_95']:+.4f}"),
        ("Precision", f"{yolo['precision']:.4f}", f"{frcnn['precision']:.4f}", f"{frcnn['precision'] - yolo['precision']:+.4f}"),
        ("Recall", f"{yolo['recall']:.4f}", f"{frcnn['recall']:.4f}", f"{frcnn['recall'] - yolo['recall']:+.4f}"),
        ("empty mAP50", f"{yolo['per_class']['empty']['map50']:.4f}", f"{frcnn['per_class']['empty']['map50']:.4f}",
         f"{frcnn['per_class']['empty']['map50'] - yolo['per_class']['empty']['map50']:+.4f}"),
        ("occupied mAP50", f"{yolo['per_class']['occupied']['map50']:.4f}", f"{frcnn['per_class']['occupied']['map50']:.4f}",
         f"{frcnn['per_class']['occupied']['map50'] - yolo['per_class']['occupied']['map50']:+.4f}"),
        ("Cross-class swaps", f"{yolo['cross_class_swaps']}", f"{frcnn['cross_class_swaps']}",
         f"{frcnn['cross_class_swaps'] - yolo['cross_class_swaps']:+d}"),
        ("Latency (ms/img)", f"{yolo['ms_per_image']:.2f}", f"{frcnn['ms_per_image']:.2f}",
         f"{frcnn['ms_per_image'] - yolo['ms_per_image']:+.2f}"),
        ("Parameters", f"{yolo['params']:,}", f"{frcnn['params']:,}", f"{frcnn['params'] - yolo['params']:+,}"),
    ]:
        print(f"{label:<24} {yv:>18} {fv:>18} {dv:>18}")
    print("=" * 84)
    print(f"Saved report & chart to {out_dir}")


if __name__ == "__main__":
    main()
