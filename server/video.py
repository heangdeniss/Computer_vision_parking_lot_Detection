"""Video frame sampling for the analysis endpoint."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
from PIL import Image


def sample_frames(raw: bytes, filename: str | None, stride: int,
                  limit: int) -> tuple[float, int, list[tuple[int, Image.Image]]]:
    """Return (fps, total frames, [(frame index, RGB image), ...]) for sampled frames."""
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
