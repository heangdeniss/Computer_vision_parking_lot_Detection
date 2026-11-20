"""Model loading, image decoding, annotation drawing and result formatting."""
from __future__ import annotations

import base64
import io
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO

from .config import BATCH, COLORS, DEFAULT_CONF, IMGSZ, IOU, IMAGE_PREVIEW_SIDE, WEIGHTS

register_heif_opener()  # lets PIL read .heic/.heif


class Detector:
    """Fused, GPU-ready YOLO model with a single warm-up pass at startup."""

    def __init__(self, weights: Any = WEIGHTS) -> None:
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights), verbose=False).to(self.device).fuse()
        if self.device != "cpu":
            self.model.model.half()  # FP16 inference on CUDA; CPU stays FP32
        self.model.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8),
                           imgsz=IMGSZ, verbose=False)
        self.names = list(self.model.names.values())

    def predict(self, images: Iterable[Image.Image], conf: float = DEFAULT_CONF,
                batch: int = BATCH) -> list[list[dict[str, Any]]]:
        predictions = self.model.predict(list(images), imgsz=IMGSZ, conf=conf,
                                         iou=IOU, batch=batch, verbose=False)
        return [detections_of(result) for result in predictions]

    def counts(self, detections: list[dict[str, Any]]) -> dict[str, int]:
        return counts_of(detections, self.names)


def load_image(raw: bytes) -> Image.Image:
    """Decode any supported format and apply EXIF orientation (common in HEIC photos)."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def jpeg_url(image: Image.Image, max_side: int = IMAGE_PREVIEW_SIDE) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def draw_boxes(image: Image.Image, detections: list[dict[str, Any]],
               max_side: int = IMAGE_PREVIEW_SIDE) -> str:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    font, scale = cv2.FONT_HERSHEY_DUPLEX, 0.6
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection["bbox"])
        color = COLORS.get(detection["class"], (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{detection['class']} {detection['confidence']:.2f}"
        position = (x1, max(22, y1 - 8))
        # Dark stroke keeps class-colored text readable on bright pavement.
        cv2.putText(canvas, text, position, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, position, font, scale, color, 2, cv2.LINE_AA)
    return jpeg_url(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), max_side=max_side)


def detections_of(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "class": result.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]


def counts_of(detections: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    return {label: sum(d["class"] == label for d in detections) for label in labels}
