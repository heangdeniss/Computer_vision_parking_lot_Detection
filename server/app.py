"""FastAPI application for parking-occupancy detection.

Run from the project root:
    .venv/bin/python -m uvicorn server.app:app --port 8000

The built React UI is served at / when frontend/dist exists.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import BATCH, CORS_ORIGINS, DEFAULT_CONF, DIST, FRAME_PREVIEW_SIDE, ROOT, WEIGHTS
from .inference import Detector, draw_boxes, jpeg_url, load_image
from .video import sample_frames

app = FastAPI(title="Parking occupancy detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": str(WEIGHTS.relative_to(ROOT)),
            "classes": detector.names, "device": str(detector.device), "batch": BATCH}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)) -> JSONResponse:
    """HEIC/HEIF files cannot be shown by browsers — return a JPEG copy."""
    try:
        image = load_image(await file.read())
    except Exception:
        return JSONResponse({"error": "unreadable image"}, status_code=400)
    return JSONResponse({"image": jpeg_url(image), "width": image.width, "height": image.height})


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...),
                  conf: float = Form(DEFAULT_CONF)) -> JSONResponse:
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

    results = []
    for name, image, detections in zip(names, images, detector.predict(images, conf)):
        results.append({
            "name": name,
            "counts": detector.counts(detections),
            "detections": detections,
            "image": draw_boxes(image, detections),
            "width": image.width,
            "height": image.height,
        })
    return JSONResponse({"results": results})


@app.post("/api/frame")
async def frame(file: UploadFile = File(...), conf: float = Form(DEFAULT_CONF)) -> JSONResponse:
    """Single-frame inference for the browser's realtime loop (boxes only, no image)."""
    if not 0.0 <= conf <= 1.0:
        return JSONResponse({"error": "invalid conf"}, status_code=400)
    try:
        image = load_image(await file.read())
    except Exception:
        return JSONResponse({"error": "unreadable frame"}, status_code=400)
    detections = detector.predict([image], conf, batch=1)[0]
    return JSONResponse({"counts": detector.counts(detections), "detections": detections})


@app.post("/api/video")
async def video(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    stride: int = Form(15),
    limit: int = Form(240),
) -> JSONResponse:
    """Sample every Nth frame, detect in batches, return a browsable frame timeline."""
    if not 0.0 <= conf <= 1.0 or not 1 <= stride <= 300 or not 1 <= limit <= 600:
        return JSONResponse({"error": "invalid conf / stride / limit"}, status_code=400)

    fps, total_frames, frames = sample_frames(await file.read(), file.filename, stride, limit)
    if not frames:
        return JSONResponse({"error": "no readable frames in video"}, status_code=400)

    results = []
    for (index, image), detections in zip(frames, detector.predict([image for _, image in frames], conf)):
        results.append({
            "index": index,
            "time": round(index / fps, 2),
            "counts": detector.counts(detections),
            "detections": detections,
            "image": draw_boxes(image, detections, max_side=FRAME_PREVIEW_SIDE),
        })

    totals = {label: sum(frame["counts"][label] for frame in results) for label in detector.names}
    return JSONResponse({
        "name": file.filename, "fps": round(fps, 2), "total_frames": total_frames,
        "sampled": len(results), "stride": stride, "totals": totals, "results": results,
    })


# Serves the built React app at http://localhost:8000 when frontend/dist exists.
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
