"""FastAPI application for parking-occupancy detection (YOLOv11 & Faster R-CNN).

Run from the project root:
    .venv/bin/python -m uvicorn server.app:app --port 8000
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import (
    BATCH,
    CORS_ORIGINS,
    DEFAULT_CONF,
    DIST,
    FASTER_RCNN_WEIGHTS,
    FRAME_PREVIEW_SIDE,
    MODEL_LABELS,
    ROOT,
    WEIGHTS,
)
from .detectors import FasterRCNNDetector, load_detectors
from .media import decode_frame_bgr, draw_boxes, jpeg_url, load_image, sample_frames

app = FastAPI(title="Parking occupancy detector (YOLOv11 + Faster R-CNN)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

detectors: dict[str, Any] = load_detectors()
detector = detectors["yolo11"]
VALID_MODEL_MODES = {"yolo11", "faster_rcnn", "both"}


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _ensure_faster_rcnn() -> Any | None:
    if "faster_rcnn" not in detectors and FASTER_RCNN_WEIGHTS.exists():
        detectors["faster_rcnn"] = FasterRCNNDetector(FASTER_RCNN_WEIGHTS)
    return detectors.get("faster_rcnn")


def _resolve_models(mode: str) -> tuple[list[str], JSONResponse | None]:
    if mode not in VALID_MODEL_MODES:
        return [], JSONResponse(
            {"error": f"invalid model '{mode}'; expected one of {sorted(VALID_MODEL_MODES)}"},
            status_code=400,
        )
    if mode in {"faster_rcnn", "both"} and _ensure_faster_rcnn() is None:
        return [], JSONResponse(
            {"error": f"Faster R-CNN checkpoint not found at {_rel_path(FASTER_RCNN_WEIGHTS)}. "
                      "Train it with: .venv/bin/python src/faster_rcnn/train.py"},
            status_code=503,
        )
    return (["yolo11", "faster_rcnn"] if mode == "both" else [mode]), None


@app.get("/api/health")
def health() -> dict:
    frcnn = _ensure_faster_rcnn()
    return {
        "ok": True,
        "model": _rel_path(WEIGHTS),
        "classes": detector.names,
        "device": str(detector.device),
        "batch": BATCH,
        "models": {
            "yolo11": {
                "available": True,
                "label": detector.label,
                "path": _rel_path(WEIGHTS),
                "device": str(detector.device),
            },
            "faster_rcnn": {
                "available": frcnn is not None,
                "label": frcnn.label if frcnn is not None else MODEL_LABELS["faster_rcnn"],
                "path": _rel_path(FASTER_RCNN_WEIGHTS),
                "device": str(frcnn.device) if frcnn is not None else None,
            },
        },
    }


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)) -> JSONResponse:
    """Convert HEIC/HEIF uploads into a browser-readable JPEG preview."""
    try:
        image = load_image(await file.read())
    except Exception:
        return JSONResponse({"error": "unreadable image"}, status_code=400)
    return JSONResponse({"image": jpeg_url(image), "width": image.width, "height": image.height})


async def _predict_impl(files: list[UploadFile], conf: float, model: str) -> JSONResponse:
    if not files or not 0.0 <= conf <= 1.0:
        return JSONResponse({"error": "no images or invalid conf"}, status_code=400)
    keys, err = _resolve_models(model)
    if err is not None:
        return err

    images: list[Image.Image] = []
    names: list[str] = []
    for upload in files:
        try:
            images.append(load_image(await upload.read()))
            names.append(upload.filename or f"image_{len(names) + 1}")
        except Exception:
            return JSONResponse({"error": f"{upload.filename}: unreadable image"}, status_code=400)

    if model != "both":
        det = detectors[keys[0]]
        preds, latency_ms = det.predict_timed(images, conf)
        results = [
            {
                "name": name,
                "counts": det.counts(detections),
                "detections": detections,
                "image": draw_boxes(image, detections),
                "width": image.width,
                "height": image.height,
                "latency_ms": latency_ms,
            }
            for name, image, detections in zip(names, images, preds)
        ]
        return JSONResponse({"model": keys[0], "label": det.label, "results": results})

    per_model = {key: detectors[key].predict_timed(images, conf) for key in keys}
    results = []
    for idx, (name, image) in enumerate(zip(names, images)):
        results.append({
            "name": name,
            "width": image.width,
            "height": image.height,
            "models": {
                key: {
                    "label": detectors[key].label,
                    "counts": detectors[key].counts(per_model[key][0][idx]),
                    "detections": per_model[key][0][idx],
                    "image": draw_boxes(image, per_model[key][0][idx]),
                    "latency_ms": per_model[key][1],
                }
                for key in keys
            },
        })
    return JSONResponse({"mode": "both", "models": keys, "results": results})


@app.post("/api/predict")
async def predict(
    files: list[UploadFile] = File(...),
    conf: float = Form(DEFAULT_CONF),
    model: str = Form("yolo11"),
) -> JSONResponse:
    return await _predict_impl(files, conf, model)


@app.post("/api/compare")
async def compare(
    files: list[UploadFile] = File(...),
    conf: float = Form(DEFAULT_CONF),
) -> JSONResponse:
    """Run YOLOv11 and Faster R-CNN side by side on uploaded images."""
    return await _predict_impl(files, conf, "both")


@app.post("/api/frame")
async def frame(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    model: str = Form("yolo11"),
) -> JSONResponse:
    """Single-frame inference for the realtime loop (boxes + counts only)."""
    if not 0.0 <= conf <= 1.0:
        return JSONResponse({"error": "invalid conf"}, status_code=400)
    keys, err = _resolve_models(model)
    if err is not None:
        return err
    try:
        bgr = decode_frame_bgr(await file.read())
    except Exception:
        return JSONResponse({"error": "unreadable frame"}, status_code=400)

    if model != "both":
        det = detectors[keys[0]]
        detections, latency_ms = await asyncio.to_thread(det.predict_frame_bgr, bgr, conf)
        return JSONResponse({
            "model": keys[0],
            "label": det.label,
            "counts": det.counts(detections),
            "detections": detections,
            "latency_ms": latency_ms,
        })

    models_out: dict[str, Any] = {}
    for key in keys:
        det = detectors[key]
        detections, latency_ms = await asyncio.to_thread(det.predict_frame_bgr, bgr, conf)
        models_out[key] = {
            "label": det.label,
            "counts": det.counts(detections),
            "detections": detections,
            "latency_ms": latency_ms,
        }
    return JSONResponse({"mode": "both", "models": models_out})


@app.post("/api/video")
async def video(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    stride: int = Form(15),
    limit: int = Form(240),
    model: str = Form("yolo11"),
) -> JSONResponse:
    """Sample every Nth frame and return a browsable annotated timeline."""
    if not 0.0 <= conf <= 1.0 or not 1 <= stride <= 300 or not 1 <= limit <= 600:
        return JSONResponse({"error": "invalid conf / stride / limit"}, status_code=400)
    keys, err = _resolve_models(model)
    if err is not None:
        return err

    fps, total_frames, frames = sample_frames(await file.read(), file.filename, stride, limit)
    if not frames:
        return JSONResponse({"error": "no readable frames in video"}, status_code=400)

    pil_frames = [image for _, image in frames]

    if model != "both":
        det = detectors[keys[0]]
        preds, avg_ms = det.predict_timed(pil_frames, conf)
        results = [
            {
                "index": index,
                "time": round(index / fps, 2),
                "counts": det.counts(detections),
                "detections": detections,
                "image": draw_boxes(image, detections, max_side=FRAME_PREVIEW_SIDE),
            }
            for (index, image), detections in zip(frames, preds)
        ]
        totals = {label: sum(item["counts"][label] for item in results) for label in det.names}
        return JSONResponse({
            "model": keys[0], "label": det.label, "avg_latency_ms": avg_ms,
            "name": file.filename, "fps": round(fps, 2), "total_frames": total_frames,
            "sampled": len(results), "stride": stride, "totals": totals, "results": results,
        })

    per_model = {key: detectors[key].predict_timed(pil_frames, conf) for key in keys}
    results = []
    for idx, (frame_index, image) in enumerate(frames):
        results.append({
            "index": frame_index,
            "time": round(frame_index / fps, 2),
            "models": {
                key: {
                    "label": detectors[key].label,
                    "counts": detectors[key].counts(per_model[key][0][idx]),
                    "detections": per_model[key][0][idx],
                    "image": draw_boxes(image, per_model[key][0][idx], max_side=FRAME_PREVIEW_SIDE),
                }
                for key in keys
            },
        })

    return JSONResponse({
        "mode": "both",
        "name": file.filename,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "sampled": len(results),
        "stride": stride,
        "models": {
            key: {
                "label": detectors[key].label,
                "avg_latency_ms": per_model[key][1],
                "totals": {
                    label: sum(item["models"][key]["counts"][label] for item in results)
                    for label in detectors[key].names
                },
            }
            for key in keys
        },
        "results": results,
    })


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
