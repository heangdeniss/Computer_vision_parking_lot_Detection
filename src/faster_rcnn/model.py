"""Torchvision Faster R-CNN model builder and checkpoint loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BACKBONES = {
    "resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    "resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
    "mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    ),
    "mobilenet_v3_large_320_fpn": (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
    ),
}


def build_faster_rcnn(
    backbone: str = "resnet50_fpn_v2",
    num_classes: int = 3,
    pretrained: bool = True,
    min_size: int = 640,
    max_size: int = 800,
    box_nms_thresh: float = 0.6,
    box_detections_per_img: int = 300,
    box_score_thresh: float = 0.001,
) -> torch.nn.Module:
    """Build a torchvision Faster R-CNN with a replaced classification/box head."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    builder, default_weights = BACKBONES[backbone]
    kwargs: dict[str, Any] = {
        "min_size": min_size,
        "max_size": max_size,
        "box_nms_thresh": box_nms_thresh,
        "box_detections_per_img": box_detections_per_img,
        "box_score_thresh": box_score_thresh,
    }
    model = (
        builder(weights=default_weights, **kwargs)
        if pretrained
        else builder(weights=None, weights_backbone=None, **kwargs)
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_faster_rcnn_checkpoint(
    weights_path: Path | str,
    device: torch.device | str = "cpu",
    box_score_thresh: float = 0.001,
    box_nms_thresh: float | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained Faster R-CNN checkpoint saved by src/faster_rcnn/train.py."""
    ckpt = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
    names = list(ckpt.get("names", ["empty", "occupied"]))
    backbone = ckpt.get("backbone", "resnet50_fpn_v2")
    min_size = int(ckpt.get("min_size", 640))
    max_size = int(ckpt.get("max_size", 800))
    nms = float(box_nms_thresh if box_nms_thresh is not None else ckpt.get("box_nms_thresh", 0.6))
    max_det = int(ckpt.get("box_detections_per_img", 300))

    model = build_faster_rcnn(
        backbone=backbone,
        num_classes=len(names) + 1,
        pretrained=False,
        min_size=min_size,
        max_size=max_size,
        box_nms_thresh=nms,
        box_detections_per_img=max_det,
        box_score_thresh=box_score_thresh,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    meta = {
        "backbone": backbone,
        "names": names,
        "min_size": min_size,
        "max_size": max_size,
        "box_nms_thresh": nms,
        "box_detections_per_img": max_det,
        "epoch": ckpt.get("epoch"),
        "val_metrics": ckpt.get("val_metrics"),
    }
    return model, meta
