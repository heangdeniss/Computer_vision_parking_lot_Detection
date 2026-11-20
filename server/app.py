#!/usr/bin/env python3
"""Inference API for the parking-occupancy YOLOv11 model.

Run from the project root: .venv/bin/python -m uvicorn server.app:app --port 8000
POST /api/predict accepts one or many images (JPG/PNG/HEIC) and returns counts + boxes.
POST /api/preview converts an image (including HEIC) to a browser-friendly JPEG.
"""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/detect/parking_yolo11s/weights/best.pt"
DIST = ROOT / "frontend/dist"
BATCH = 4  # fixed internal batch: safe on the 4 GB GPU, no UI knob needed
register_heif_opener()  # lets PIL read .heic/.heif

app = FastAPI(title="Parking occupancy detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(WEIGHTS), verbose=False).to(DEVICE).fuse()
if DEVICE != "cpu":
    model.model.half()  # FP16 inference: ~10% faster, eager and safe on this checkpoint
model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)  # warm up the first request
COLORS = {"empty": (0, 255, 0), "occupied": (0, 0, 255)}  # BGR: bold green / bold red


def load_image(raw: bytes) -> Image.Image:
    """Decode any supported format and apply EXIF orientation (common in HEIC photos)."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def jpeg_url(image: Image.Image, max_side: int = 1600) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def draw_boxes(image: Image.Image, detections: list[dict[str, Any]], max_side: int = 1600) -> str:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    font, scale = cv2.FONT_HERSHEY_DUPLEX, 0.6
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection["bbox"])
        color = COLORS.get(detection["class"], (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{detection['class']} {detection['confidence']:.2f}"
        position = (x1, max(22, y1 - 8))
        # Dark stroke keeps the original green/orange text readable on bright pavement.
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


def counts_of(detections: list[dict[str, Any]]) -> dict[str, int]:
    return {label: sum(d["class"] == label for d in detections) for label in model.names.values()}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": str(WEIGHTS.relative_to(ROOT)),
            "classes": list(model.names.values()), "device": str(DEVICE), "batch": BATCH}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)) -> JSONResponse:
    """HEIC/HEIF files cannot be shown by browsers - return a JPEG copy."""
    try:
        image = load_image(await file.read())
    except Exception:
        return JSONResponse({"error": "unreadable image"}, status_code=400)
    return JSONResponse({"image": jpeg_url(image), "width": image.width, "height": image.height})


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...), conf: float = Form(0.25)) -> JSONResponse:
    if not files or not 0.0 <= conf <= 1.0:
        return JSONResponse({"error": "no images or invalid conf"}, status_code=400)

    images: list[Image.Image] = []
    names: list[str] = []
    for upload in files:
        try:
            images.append(load_image(await upload.read()))
            names.append(upload.filename or f"image_{len(names) + 1}")
        except Exception:
            return JSONResponse({"error": f"{upload.filename}: unreadable image"}, status_code=400)

    predictions = model.predict(images, imgsz=640, conf=conf, iou=0.6, batch=BATCH, verbose=False)

    results = []
    for name, image, result in zip(names, images, predictions):
        detections = detections_of(result)
        results.append({
            "name": name,
            "counts": counts_of(detections),
            "detections": detections,
            "image": draw_boxes(image, detections),
            "width": image.width,
            "height": image.height,
        })

    return JSONResponse({"results": results})


@app.post("/api/frame")
async def frame(file: UploadFile = File(...), conf: float = Form(0.25)) -> JSONResponse:
    """Single-frame inference for the browser's realtime loop (boxes only, no image)."""
    if not 0.0 <= conf <= 1.0:
        return JSONResponse({"error": "invalid conf"}, status_code=400)
    try:
        image = load_image(await file.read())
    except Exception:
        return JSONResponse({"error": "unreadable frame"}, status_code=400)
    prediction = model.predict(image, imgsz=640, conf=conf, iou=0.6, verbose=False)[0]
    detections = detections_of(prediction)
    return JSONResponse({"counts": counts_of(detections), "detections": detections})


@app.post("/api/video")
async def video(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    stride: int = Form(15),
    limit: int = Form(240),
) -> JSONResponse:
    """Sample every Nth frame, detect in batches, return a browsable frame timeline."""
    if not 0.0 <= conf <= 1.0 or not 1 <= stride <= 300 or not 1 <= limit <= 600:
        return JSONResponse({"error": "invalid conf / stride / limit"}, status_code=400)

    raw = await file.read()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
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

    if not frames:
        return JSONResponse({"error": "no readable frames in video"}, status_code=400)

    predictions = model.predict([image for _index, image in frames], imgsz=640, conf=conf,
                                iou=0.6, batch=BATCH, verbose=False)
    results = []
    for (index, image), prediction in zip(frames, predictions):
        detections = detections_of(prediction)
        results.append({
            "index": index,
            "time": round(index / fps, 2),
            "counts": counts_of(detections),
            "detections": detections,
            "image": draw_boxes(image, detections, max_side=640),
        })

    totals = {label: sum(frame["counts"][label] for frame in results) for label in model.names.values()}
    return JSONResponse({
        "name": file.filename, "fps": round(fps, 2), "total_frames": total_frames,
        "sampled": len(results), "stride": stride, "totals": totals, "results": results,
    })


# Serves the built React app at http://localhost:8000 when frontend/dist exists.
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
