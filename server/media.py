"""Image decoding, JPEG encoding, box annotation drawing, and video frame sampling."""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .config import COLORS, IMAGE_PREVIEW_SIDE

register_heif_opener()  # lets PIL read .heic/.heif


def load_image(raw: bytes) -> Image.Image:
    """Decode any supported image format and apply EXIF orientation."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def decode_frame_bgr(raw: bytes) -> np.ndarray:
    """Fast JPEG/PNG decode straight to BGR numpy array for realtime frames."""
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("unreadable frame")
    return bgr


def jpeg_url(image: Image.Image, max_side: int = IMAGE_PREVIEW_SIDE) -> str:
    """Resize preserving aspect ratio and return a base64 JPEG data URL."""
    copy = image.copy()
    copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def draw_boxes(
    image: Image.Image,
    detections: list[dict[str, Any]],
    max_side: int = IMAGE_PREVIEW_SIDE,
) -> str:
    """Draw green (empty) / red (occupied) bounding boxes and return a JPEG data URL."""
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    font, scale = cv2.FONT_HERSHEY_DUPLEX, 0.6
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
        color = COLORS.get(det["class"], (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{det['class']} {det['confidence']:.2f}"
        pos = (x1, max(22, y1 - 8))
        cv2.putText(canvas, text, pos, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, pos, font, scale, color, 2, cv2.LINE_AA)
    return jpeg_url(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), max_side=max_side)


def sample_frames(
    raw: bytes, filename: str | None, stride: int, limit: int
) -> tuple[float, int, list[tuple[int, Image.Image]]]:
    """Return (fps, total_frames, [(frame_index, RGB Image), ...]) for sampled frames."""
    suffix = Path(filename or "video.mp4").suffix or ".mp4"
    frames: list[tuple[int, Image.Image]] = []
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        capture = cv2.VideoCapture(tmp.name)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = 0
        while len(frames) < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append((index, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
            index += 1
        capture.release()
    return fps, total_frames, frames
