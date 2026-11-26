# Parking Lot Occupancy Detection — YOLOv11 vs. Faster R-CNN

**A full-stack computer vision system that detects and classifies parking spaces as `empty` or `occupied` using both a one-stage detector (YOLOv11s) and a two-stage detector (Faster R-CNN ResNet50-FPN-v2) trained on identical bounding-box annotations — complete with side-by-side benchmarking, a real-time React web app, Docker deployment, and a Graphify knowledge graph.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-small-FF6F00)
![Faster R-CNN](https://img.shields.io/badge/Faster%20R--CNN-ResNet50--FPN--v2-F59E0B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.165-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18%20%2B%20Vite%205-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CC%20BY%204.0-blue)

---

## Highlights

- **Apples-to-apples model comparison:** Both **YOLOv11s** and **Faster R-CNN (`resnet50_fpn_v2`)** read the exact same YOLO `.txt` annotations (`parking-space.v1i.yolov11/`) and are evaluated with identical COCO IoU (`0.50:0.95`) matching.
- **Side-by-side web application:** Compare YOLOv11 and Faster R-CNN simultaneously on uploaded images (with `Δ Empty`, `Δ Occupied`, speed ratio, and detection tables), sampled video timelines, and **real-time dual-canvas live video/webcam streams** (independent per-model async workers — YOLOv11 runs at ~44 FPS without blocking on Faster R-CNN).
- **Consumer-GPU friendly:** Both training pipelines and simultaneous dual-model inference fit comfortably inside a **4 GB NVIDIA RTX 3050 Laptop GPU** using AMP FP16 and gradient accumulation.
- **One-command Docker deployment:** Multi-stage `Dockerfile` (Node 20 + Python 3.13) with `docker-compose.yml` for Docker Desktop and `docker-compose.gpu.yml` for NVIDIA GPU hosts.
- **Queryable architecture graph:** Pre-indexed with [Graphify](https://github.com/Graphify-Labs/graphify) (`graphify-out/`) for instant code navigation.

> For deep technical breakdowns of both neural architectures, training curves, error analysis, and streaming internals, see **[docs/architecture.md](docs/architecture.md)**.

---

## Benchmark Results (Held-Out Test Split)

Evaluated on 62 held-out test images (514 ground-truth spaces: 372 `empty`, 142 `occupied`) at `conf=0.001` (standard mAP) and `iou=0.6` on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB):

| Metric | YOLOv11s | Faster R-CNN (`resnet50_fpn_v2`) | Delta (F − Y) |
|---|---:|---:|---:|
| **mAP50** | **0.9691** | 0.9619 | −0.0072 |
| **mAP50-95** | **0.7048** | 0.6770 | −0.0277 |
| **Precision** | 0.9109 | **0.9465** | **+0.0355** |
| **Recall** | **0.9500** | 0.9268 | −0.0231 |
| `empty` AP50 / AP50-95 | **0.9843** / **0.6968** | 0.9776 / 0.6649 | −0.0067 / −0.0319 |
| `occupied` AP50 / AP50-95 | **0.9538** / **0.7127** | 0.9463 / 0.6891 | −0.0075 / −0.0236 |
| **Cross-class swaps (@ 0.25)** | **0** | **0** | 0 |
| **Parameters** | **9.43 M** | 43.26 M | +33.83 M (4.6×) |
| **Checkpoint size** | **18.3 MB** | 165.4 MB | +147.1 MB (9.0×) |
| **Batch=1 GPU latency** | **17.2 ms/img** | 71.7 ms/img | +54.5 ms (4.2×) |
| **Realtime `/api/frame` FPS** | **44.4 FPS** | 12.6 FPS | — |

**Takeaway:** Both architectures separate `empty` vs. `occupied` with **zero cross-class confusions**. **YOLOv11s** is the clear production choice (4.2× faster, 9× smaller, higher `mAP50-95`), while **Faster R-CNN** offers higher peak precision (`0.9465`) and fewer missed stalls at `conf=0.25` as a strong two-stage comparison baseline.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Dataset Setup](#dataset-setup)
- [Training, Evaluation & Benchmarking](#training-evaluation--benchmarking)
- [Web App & API Reference](#web-app--api-reference)
- [Graphify Knowledge Graph](#graphify-knowledge-graph)
- [Testing](#testing)
- [Limitations](#limitations)
- [Attribution](#attribution)

---

## Repository Structure

```text
Computer_vision_parking_lot_Detection/
├── README.md                       # project overview, benchmarks & usage
├── Dockerfile                      # multi-stage React + FastAPI container
├── docker-compose.yml              # local Docker Desktop deployment
├── docker-compose.gpu.yml          # optional NVIDIA GPU Compose override
├── requirements.txt                # pinned Python dependencies
├── .gitignore                      # excludes >100 MB weights, datasets, caches & envs
├── configs/
│   ├── yolo11_parking.yaml         # YOLOv11 architecture definition (nc=2, scale s)
│   ├── train_hyp.yaml              # YOLOv11 training hyperparameters
│   └── faster_rcnn.yaml            # Faster R-CNN hyperparameters (4 GB VRAM tuned)
├── docs/
│   └── architecture.md             # detailed architecture & comparison study for both models
├── notebooks/
│   └── plot_results.ipynb          # training-curve plotting notebook
├── scripts/
│   └── install-docker-popos.sh     # Pop!_OS / Ubuntu Docker CLI + Desktop UI installer
├── src/
│   ├── data.py                     # shared YOLO data.yaml & bounding-box dataset loader
│   ├── metrics.py                  # shared IoU matching, COCO mAP & confusion matrix
│   ├── compare.py                  # side-by-side YOLOv11 vs. Faster R-CNN benchmark
│   ├── yolo/
│   │   ├── inspect_model.py        # layer table, GFLOPs & pretrained weight check
│   │   ├── train.py                # fine-tune YOLOv11 (--dry-run)
│   │   └── evaluate.py             # evaluate YOLOv11 on val/test
│   └── faster_rcnn/
│       ├── model.py                # multi-backbone Faster R-CNN builder & checkpoint loader
│       ├── train.py                # fine-tune Faster R-CNN on YOLO labels (--dry-run)
│       └── evaluate.py             # evaluate Faster R-CNN on val/test
├── server/
│   ├── config.py                   # paths, thresholds, colors & model labels
│   ├── media.py                    # image/EXIF/HEIC decode, cv2 frame decode, box drawing, video sampling
│   ├── detectors.py                # thread-safe YoloDetector & FasterRCNNDetector wrappers
│   └── app.py                      # FastAPI routes + static React UI host
├── frontend/                       # React 18 + Vite 5 UI (single & side-by-side modes)
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/             # Badges, Controls, DetectionTable, ImageCard, LiveView, UploadZone, VideoTimeline
│   │   ├── hooks/useLiveDetection.js
│   │   └── lib/                    # annotations.js, api.js, constants.js
│   └── dist/                       # prebuilt production UI served by FastAPI
├── tests/
│   ├── test_server.py              # media, box drawing & count helper tests
│   └── test_faster_rcnn.py         # shared YOLO dataset, metrics & label-remap tests
├── graphify-out/                   # Graphify AST knowledge graph (graph.json, graph.html, GRAPH_REPORT.md)
└── runs/detect/parking_yolo11s/
    └── weights/best.pt             # included YOLOv11s checkpoint (18.3 MB)
```

> **Note on checkpoints & GitHub's 100 MB limit:** The 18.3 MB YOLOv11s checkpoint (`runs/detect/parking_yolo11s/weights/best.pt`) is committed so the app works immediately after cloning. Faster R-CNN's checkpoint (`165.4 MB`) is gitignored; train it locally with `src/faster_rcnn/train.py` (see below) and the server/UI will automatically detect and enable side-by-side mode.

---

## Quick Start

### Prerequisites
- **Python 3.13** (NVIDIA CUDA GPU recommended; CPU fallback is automatic)
- **Node.js 18+** and `npm` (only needed if editing the React UI)

### 1. Clone & install Python dependencies

```bash
git clone https://github.com/heangdeniss/Computer_vision_parking_lot_Detection.git
cd Computer_vision_parking_lot_Detection

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run the web app (API + prebuilt React UI)

```bash
.venv/bin/python -m uvicorn server.app:app --port 8000
```

Open **http://localhost:8000** (verify loaded models at `http://localhost:8000/api/health`).

### 3. Frontend development (hot reload)

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (auto-proxies /api to port 8000)
# npm run build  # rebuild frontend/dist/ for FastAPI static hosting
```

---

## Docker Deployment

The multi-stage [`Dockerfile`](Dockerfile) builds the React bundle in `node:20-slim` and serves FastAPI + PyTorch in `python:3.13-slim`.

### Pop!_OS / Ubuntu host setup (optional)
If you need Docker CLI + Docker Desktop UI on Pop!_OS or Ubuntu, run from a host terminal:

```bash
./scripts/install-docker-popos.sh
```

### Deploy with Docker Compose (recommended)

```bash
docker compose up -d --build
# Open http://localhost:8000
# Health check: curl http://localhost:8000/api/health
```

### Deploy with plain Docker CLI

```bash
docker build -t parking-lot-detector:latest .
docker run -d --name parking-lot-detector -p 8000:8000 --restart unless-stopped parking-lot-detector:latest
```

### Optional: NVIDIA GPU inside Docker
Docker Desktop on Linux runs inside a VM (CPU mode). For RTX GPU acceleration inside Docker, use native Docker Engine + NVIDIA Container Toolkit:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## Dataset Setup

Required only for training or running dataset benchmarks (not needed for inference):

- **Source:** [Roboflow parking-space v1](https://universe.roboflow.com/denischan/parking-space-bxjkf-jus8n/dataset/1) (**CC BY 4.0**)
- **Splits:** 496 train (3,780 boxes) · 62 valid (429 boxes) · 62 test (514 boxes)
- **Classes:** `0: empty`, `1: occupied`

Extract the YOLOv11 export into `parking-space.v1i.yolov11/`:

```bash
unzip -o parking-space.v1i.yolov11.zip -d parking-space.v1i.yolov11
```

Both YOLOv11 and Faster R-CNN read `parking-space.v1i.yolov11/data.yaml` directly — no COCO JSON conversion required.

---

## Training, Evaluation & Benchmarking

### YOLOv11s

```bash
# Inspect layers, shapes, GFLOPs, and pretrained weight transfer
.venv/bin/python src/yolo/inspect_model.py

# Validate config & weights without training
.venv/bin/python src/yolo/train.py --dry-run

# Fine-tune (100 epochs, AdamW, batch=8, 640 px)
.venv/bin/python src/yolo/train.py --weights yolo11s.pt

# Evaluate on test or val split
.venv/bin/python src/yolo/evaluate.py --weights runs/detect/parking_yolo11s/weights/best.pt --split test
```

### Faster R-CNN (same YOLO bounding-box annotations)

```bash
# Validate YOLO->xyxy loader and model without training
.venv/bin/python src/faster_rcnn/train.py --dry-run

# Fine-tune ResNet50-FPN-v2 (AMP FP16, batch=2 x grad_accum=4)
.venv/bin/python src/faster_rcnn/train.py --epochs 20 --exist-ok

# Optional lighter backbone (MobileNetV3-FPN)
.venv/bin/python src/faster_rcnn/train.py --backbone mobilenet_v3_large_fpn --batch 4 --grad-accum 2

# Evaluate on test or val split
.venv/bin/python src/faster_rcnn/evaluate.py --weights runs/faster_rcnn/parking_frcnn/weights/best.pt --split test
```

Outputs are written to `runs/faster_rcnn/parking_frcnn/weights/{best.pt,last.pt}` and `results.csv`. Once `best.pt` exists, the FastAPI server automatically exposes Faster R-CNN and enables side-by-side mode in the UI.

### Side-by-Side Benchmark

```bash
.venv/bin/python src/compare.py --split test
```

Prints a full metric & latency delta table and saves `runs/compare/test/comparison.json` + `comparison.png`.

---

## Web App & API Reference

### Web UI Modes
| Mode | Capabilities |
|---|---|
| **Model selector** | Switch between **Side by side (YOLOv11 vs Faster R-CNN)**, **YOLOv11s**, and **Faster R-CNN** at any time. |
| **Images** | Drag-and-drop multiple JPG/PNG/HEIC photos. Side-by-side cards display annotated outputs, per-model latency pills, `Δ Empty` / `Δ Occupied`, speed ratio, and expandable detection tables. |
| **Video · timeline** | Samples every 1/5/15/30/60 frames (cap 240). Shows per-model aggregate totals, average ms/frame, and side-by-side frame inspection with download links. |
| **Video · realtime** | Smooth 60 FPS video/webcam playback with decoupled per-model async detection workers (dual canvases in side-by-side mode, per-model live FPS & latency readouts). |

Box colors: **green** = `empty`, **red** = `occupied` (with dark text stroke for daylight legibility).

### REST API Endpoints

| Method | Endpoint | Form Fields | Response |
|---|---|---|---|
| `GET` | `/api/health` | — | Loaded model paths, per-model availability (`yolo11`, `faster_rcnn`), classes, device, batch size |
| `POST` | `/api/preview` | `file` | Browser-friendly JPEG preview data URL + dimensions (for HEIC/HEIF) |
| `POST` | `/api/predict` | `files[]`, `conf` (0.25), `model` (`yolo11` \| `faster_rcnn` \| `both`) | Per-image counts, detections, latency (`ms`), annotated image(s) |
| `POST` | `/api/compare` | `files[]`, `conf` (0.25) | Convenience alias for `/api/predict` with `model=both` |
| `POST` | `/api/frame` | `file`, `conf`, `model` (`yolo11` \| `faster_rcnn` \| `both`) | Boxes, counts & latency only (low-overhead OpenCV decode path for realtime streaming) |
| `POST` | `/api/video` | `file`, `conf`, `stride` (15), `limit` (240), `model` | Sampled frame timeline, totals & per-frame detections (single or side-by-side) |

Example side-by-side request:

```bash
curl -F "files=@lot.jpg" -F "conf=0.25" http://localhost:8000/api/compare
```

---

## Graphify Knowledge Graph

This project includes a local AST knowledge graph in [`graphify-out/`](graphify-out/) (`graph.json`, `graph.html`, [`GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md)) plus git hooks that keep it current on code changes.

```bash
graphify update .                       # refresh graph after code changes (local AST, zero API cost)
graphify query "FasterRCNNDetector"     # trace classes, callers, and imports
graphify explain "YoloParkingDataset"   # inspect a node and its neighbors
```

---

## Testing

Run the pure-Python unit test suite (no GPU or server required):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Covers YOLO `.txt` → `xyxy` box parsing, clipping, label shifting (`+1`/`-1`), horizontal-flip box geometry, IoU matching, confusion-matrix updates, image/frame decoding, and box rendering.

---

## Limitations

- **Frame-level dataset split:** The Roboflow v1 export splits by frame rather than by video ID, so reported mAP is optimistic for completely unseen camera perspectives. Re-split by video ID for strict out-of-domain evaluation.
- **Class imbalance:** Training has ~1.8× more `empty` than `occupied` boxes; monitor per-class recall when adding new lots.
- **Realtime box latency:** Video renders at native 30/60 FPS while boxes track at each model's GPU frame rate (~44 FPS for YOLOv11, ~12.6 FPS for Faster R-CNN on RTX 3050). Webcam capture requires `localhost` or HTTPS.

---

## Attribution

- **Dataset:** [Roboflow parking-space v1](https://universe.roboflow.com/denischan/parking-space-bxjkf-jus8n/dataset/1) — CC BY 4.0
- **YOLOv11:** [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) — AGPL-3.0
- **Faster R-CNN:** [Torchvision Detection (`fasterrcnn_resnet50_fpn_v2`)](https://pytorch.org/vision/stable/models/faster_rcnn.html) — BSD-3-Clause
- **Frontend:** React 18 & Vite 5 — MIT
- **Code Graph:** [Graphify](https://github.com/Graphify-Labs/graphify)
