"""Paths and constants shared by the inference API."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/detect/parking_yolo11s/weights/best.pt"
DIST = ROOT / "frontend/dist"

IMGSZ = 640
IOU = 0.6
DEFAULT_CONF = 0.25
BATCH = 4  # fixed internal batch: safe on the 4 GB GPU, no UI knob needed

IMAGE_PREVIEW_SIDE = 1600
FRAME_PREVIEW_SIDE = 640

# BGR colors for OpenCV: green = empty, red = occupied.
COLORS = {"empty": (0, 255, 0), "occupied": (0, 0, 255)}
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
