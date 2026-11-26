"""Model wrappers (YOLOv11 and Faster R-CNN) sharing one detection interface."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from src.faster_rcnn.model import load_faster_rcnn_checkpoint

from .config import (
    BATCH,
    DEFAULT_CONF,
    FASTER_RCNN_BATCH,
    FASTER_RCNN_WEIGHTS,
    IMGSZ,
    IOU,
    MODEL_LABELS,
    WEIGHTS,
)


def detections_of(result: Any) -> list[dict[str, Any]]:
    """Format one Ultralytics YOLO result into the shared API detection list."""
    return [
        {
            "class": result.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]


def frcnn_detections_of(
    output: dict[str, torch.Tensor], names: list[str], conf: float = 0.0
) -> list[dict[str, Any]]:
    """Format one torchvision Faster R-CNN output into the shared API detection list."""
    boxes = output["boxes"].detach().float().cpu().tolist()
    scores = output["scores"].detach().float().cpu().tolist()
    labels = output["labels"].detach().long().cpu().tolist()
    results: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        cls_id = int(label) - 1  # shift back from Faster R-CNN's 1-based class ids
        if score < conf or cls_id < 0 or cls_id >= len(names):
            continue
        results.append({
            "class": names[cls_id],
            "confidence": round(float(score), 4),
            "bbox": [round(float(v), 1) for v in box],
        })
    return results


def counts_of(detections: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    """Count detections per class label."""
    return {label: sum(d["class"] == label for d in detections) for label in labels}


class YoloDetector:
    """Fused, GPU-ready YOLOv11 detector with startup warm-up."""

    model_key = "yolo11"
    label = MODEL_LABELS["yolo11"]

    def __init__(self, weights: Any = WEIGHTS) -> None:
        self.weights_path = Path(weights)
        self._lock = threading.Lock()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights), verbose=False).to(self.device).fuse()
        if self.device != "cpu":
            self.model.model.half()
        self.model.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)
        self.names = list(self.model.names.values())

    def predict(self, images: Iterable[Image.Image], conf: float = DEFAULT_CONF,
                batch: int = BATCH) -> list[list[dict[str, Any]]]:
        return self.predict_timed(images, conf=conf, batch=batch)[0]

    def predict_timed(self, images: Iterable[Image.Image], conf: float = DEFAULT_CONF,
                      batch: int = BATCH) -> tuple[list[list[dict[str, Any]]], float]:
        image_list = list(images)
        if not image_list:
            return [], 0.0
        with self._lock:
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = self.model.predict(image_list, imgsz=IMGSZ, conf=conf, iou=IOU, batch=batch, verbose=False)
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = round(((time.perf_counter() - t0) / len(image_list)) * 1000.0, 2)
        return [detections_of(r) for r in preds], ms

    def predict_frame_bgr(self, bgr: np.ndarray,
                          conf: float = DEFAULT_CONF) -> tuple[list[dict[str, Any]], float]:
        """Low-overhead single-frame path for realtime streaming."""
        with self._lock:
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = self.model.predict(bgr, imgsz=IMGSZ, conf=conf, iou=IOU, verbose=False)[0]
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return detections_of(result), ms

    def counts(self, detections: list[dict[str, Any]]) -> dict[str, int]:
        return counts_of(detections, self.names)


class FasterRCNNDetector:
    """GPU-ready torchvision Faster R-CNN detector sharing the YoloDetector interface."""

    model_key = "faster_rcnn"
    label = MODEL_LABELS["faster_rcnn"]

    def __init__(self, weights: Any = FASTER_RCNN_WEIGHTS) -> None:
        self.weights_path = Path(weights)
        self._lock = threading.Lock()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.model, self.meta = load_faster_rcnn_checkpoint(
            self.weights_path, device=self.device, box_score_thresh=DEFAULT_CONF, box_nms_thresh=IOU
        )
        self.names = list(self.meta["names"])
        self.backbone = self.meta.get("backbone", "resnet50_fpn_v2")
        self.label = f"Faster R-CNN ({self.backbone})"
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            self.model([torch.zeros((3, IMGSZ, IMGSZ), device=self.device)])

    def predict(self, images: Iterable[Image.Image], conf: float = DEFAULT_CONF,
                batch: int = FASTER_RCNN_BATCH) -> list[list[dict[str, Any]]]:
        return self.predict_timed(images, conf=conf, batch=batch)[0]

    def predict_timed(self, images: Iterable[Image.Image], conf: float = DEFAULT_CONF,
                      batch: int = FASTER_RCNN_BATCH) -> tuple[list[list[dict[str, Any]]], float]:
        image_list = list(images)
        if not image_list:
            return [], 0.0
        chunk_size = max(1, min(batch, FASTER_RCNN_BATCH))
        all_detections: list[list[dict[str, Any]]] = []
        with self._lock:
            self.model.roi_heads.score_thresh = conf
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                for start in range(0, len(image_list), chunk_size):
                    tensors = [TF.to_tensor(img).to(self.device) for img in image_list[start:start + chunk_size]]
                    for out in self.model(tensors):
                        all_detections.append(frcnn_detections_of(out, self.names, conf))
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = round(((time.perf_counter() - t0) / len(image_list)) * 1000.0, 2)
        return all_detections, ms

    def predict_frame_bgr(self, bgr: np.ndarray,
                          conf: float = DEFAULT_CONF) -> tuple[list[dict[str, Any]], float]:
        """Low-overhead single-frame path for realtime streaming."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self.model.roi_heads.score_thresh = conf
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            tensor = torch.from_numpy(rgb).to(self.device, non_blocking=True)
            tensor = tensor.permute(2, 0, 1).float().div_(255.0)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model([tensor])[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return frcnn_detections_of(out, self.names, conf), ms

    def counts(self, detections: list[dict[str, Any]]) -> dict[str, int]:
        return counts_of(detections, self.names)


def load_detectors() -> dict[str, Any]:
    """Load YOLOv11 and (if trained locally) Faster R-CNN at startup."""
    loaded: dict[str, Any] = {"yolo11": YoloDetector(WEIGHTS)}
    if FASTER_RCNN_WEIGHTS.exists():
        loaded["faster_rcnn"] = FasterRCNNDetector(FASTER_RCNN_WEIGHTS)
    return loaded
