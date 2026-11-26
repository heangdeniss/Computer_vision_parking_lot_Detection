"""Paths and constants shared by the inference API."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", ROOT / "runs/detect/parking_yolo11s/weights/best.pt"))
FASTER_RCNN_WEIGHTS = Path(
    os.environ.get("FASTER_RCNN_WEIGHTS", ROOT / "runs/faster_rcnn/parking_frcnn/weights/best.pt")
)
DIST = ROOT / "frontend/dist"

IMGSZ = 640
IOU = 0.6
DEFAULT_CONF = 0.25
BATCH = 4              # fixed internal batch for YOLOv11: safe on 4 GB GPU
FASTER_RCNN_BATCH = 2  # fixed internal batch for Faster R-CNN: safe on 4 GB GPU alongside YOLO

IMAGE_PREVIEW_SIDE = 1600
FRAME_PREVIEW_SIDE = 640

MODEL_LABELS = {
    "yolo11": "YOLOv11s",
    "faster_rcnn": "Faster R-CNN",
}

# BGR colors for OpenCV: green = empty, red = occupied.
COLORS = {"empty": (0, 255, 0), "occupied": (0, 0, 255)}
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
