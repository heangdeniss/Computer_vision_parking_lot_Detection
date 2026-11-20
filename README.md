# Parking Lot Occupancy Detection

**A production-ready YOLOv11 detector for classifying parking spaces as `empty` or `occupied`, with a React web app for image, video, and live-camera inference.**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLO11](https://img.shields.io/badge/YOLO11-small-orange)
![Dataset](https://img.shields.io/badge/Dataset-CC%20BY%204.0-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-AGPL--3.0-lightgrey)

| Metric (held-out test split) | Result |
|---|---|
| mAP50 | **0.9691** |
| mAP50-95 | **0.7048** |
| Precision | 0.9109 |
| Recall | 0.9500 |
| Cross-class confusions at operating threshold | **0** |
| Inference speed (RTX 3050 Laptop) | 6.0–6.7 ms/image |
| Live video ceiling | ~35 fps |

Trained checkpoint: `runs/detect/parking_yolo11s/weights/best.pt` (included in this repository).

---

## Table of contents

- [Overview](#overview)
- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [System architecture](#system-architecture)
- [Prerequisites and setup](#prerequisites-and-setup)
- [Dataset](#dataset)
- [Training pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Inference application](#inference-application)
  - [Quick start](#quick-start)
  - [Web interface](#web-interface)
  - [API reference](#api-reference)
  - [Performance engineering](#performance-engineering)
- [Measured performance](#measured-performance)
- [Limitations and honest interpretation](#limitations-and-honest-interpretation)
- [Troubleshooting](#troubleshooting)
- [Attribution](#attribution)

---

## Overview

Parking occupancy is a binary visual problem: given an image or video frame of a parking area, locate every space and decide whether it is free or taken. This project fine-tunes **YOLOv11-small** on a Roboflow parking dataset and wraps the resulting model in a FastAPI service and a React UI.

The system supports three ways to use the model:

1. **Images** — upload any number of JPG, PNG, or HEIC photos at once; receive counts, boxes, confidences, and annotated previews.
2. **Video analysis** — sample a video every *N* frames and browse an annotated timeline with per-frame statistics.
3. **Realtime detection** — stream frames from an uploaded video or a webcam and draw boxes live as the scene plays.

The design goal was a small, fast, self-contained detector that runs on a consumer 4 GB GPU without a complex serving stack.

---

## Key results

### Training summary

| Field | Value |
|---|---|
| Run | `parking_yolo11s` |
| Architecture | YOLOv11-small, `nc: 2`, pretrained from `yolo11s.pt` |
| Epochs completed | 100 (full schedule, no early stop) |
| Training time | 939 s (~15.7 min) |
| Hardware | NVIDIA GeForce RTX 3050 Laptop, 4 GB, CUDA 13.2 |
| Best validation mAP50-95 | 0.7514 at epoch 97 |
| Best validation mAP50 | 0.9755 at epoch 78 |
| Final epoch mAP50-95 | 0.7483 |
| Best checkpoint | `runs/detect/parking_yolo11s/weights/best.pt` |

Training converged smoothly. Box loss fell from 1.624 (epoch 1) to 0.753 (epoch 100); classification loss fell from 1.623 to 0.375; DFL loss from 1.436 to 0.969. Curves, confusion matrices, and per-epoch history are in `plot_results.ipynb` and `runs/detect/parking_yolo11s/`.

### Held-out metrics

Evaluated with `src/evaluate.py` at the standard mAP confidence threshold (0.001), IoU 0.6, image size 640.

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 62 | 429 | 0.9279 | 0.9331 | 0.9708 | **0.7489** |
| Test | 62 | 514 | 0.9109 | 0.9500 | 0.9691 | 0.7048 |

Per-class breakdown (test split):

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `empty` | 372 | 0.9163 | 0.9543 | 0.9843 | 0.6968 |
| `occupied` | 142 | 0.9056 | 0.9456 | 0.9538 | 0.7127 |

Per-class breakdown (validation split):

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `empty` | 284 | 0.9203 | 0.9351 | 0.9664 | 0.7261 |
| `occupied` | 145 | 0.9355 | 0.9310 | 0.9752 | 0.7717 |

The two classes are learned equally well — there is no class that the model simply fails to recognise.

### Error profile at the operating threshold

With the UI's default confidence of 0.25, the test-set confusion matrix (rows = ground truth, columns = prediction) is:

| True \ Predicted | `empty` | `occupied` | Background | Total |
|---|---:|---:|---:|---:|
| `empty` | **361** | 0 | 48 | 409 |
| `occupied` | 0 | **135** | 18 | 153 |
| Background (false alarms) | 11 | 7 | — | 18 |

**Zero empty↔occupied swaps.** The dominant error mode is *missed spaces* (66 total), not incorrect occupancy decisions. That is the safer failure mode for a counting assistant, and it points to small/distant spaces and occlusion as the remaining work rather than a fundamental class-separation problem.

---

## Repository layout

```
Computer_vision_parking_lot_Detection/
├── README.md                       # this document
├── architecture.md                 # architecture study + training process record
├── input test image/               # sample photos and videos for manual testing
├── plot_results.ipynb              # training curves from results.csv
├── requirements.txt                # pinned Python dependencies
├── configs/
│   ├── yolo11_parking.yaml         # full YOLO11 graph, nc=2, scale s
│   └── train_hyp.yaml              # optimizer, schedule, augmentation
├── src/
│   ├── inspect_model.py            # layer table, GFLOPs, weight-transfer check
│   ├── train.py                    # build → transfer → fine-tune (--dry-run)
│   └── evaluate.py                 # val/test mAP and per-class metrics
├── server/
│   └── app.py                      # FastAPI inference + static UI host
├── frontend/
│   ├── src/App.jsx                 # React UI (images, video, realtime)
│   ├── src/styles.css              # dark responsive styling
│   ├── vite.config.js              # /api proxy + automatic API startup
│   ├── package.json
│   └── dist/                       # prebuilt UI served by FastAPI
├── runs/detect/parking_yolo11s/
│   └── weights/best.pt             # deployed checkpoint
└── .gitignore
```

The project-local `.venv/`, `frontend/node_modules/`, the Roboflow dataset, pretrained base weights, raw videos, and training logs are intentionally excluded from version control. See [Dataset](#dataset) and [Setup](#prerequisites-and-setup) for restoration steps.

---

## System architecture

```
        ┌──────────────────────── Browser ────────────────────────┐
        │  Images tab   │   Video analysis tab   │  Realtime tab  │
        └───────┬───────────────┬──────────────────────┬──────────┘
                │               │                      │
        POST /api/predict  POST /api/video      POST /api/frame
                └───────────────┼──────────────────────┘
                                ▼
                   FastAPI (server/app.py)
              decode / EXIF / HEIC → batch → infer → format
                                ▼
              YOLOv11s best.pt · fused · FP16 · CUDA
                    ┌───────────┴───────────┐
                    ▼                       ▼
          annotated JPEGs           boxes + counts
```

### Model

YOLOv11-small is a one-stage, anchor-free detector. The mirrored, annotated definition lives in `configs/yolo11_parking.yaml`.

```
Input 640×640×3
  │
  ├─ BACKBONE   Conv stem → C3k2 blocks → SPPF(5) → C2PSA spatial attention
  │              emits P3/8, P4/16, P5/32 feature maps
  ├─ NECK       PAN-FPN: top-down upsample + concat, then bottom-up stride convs
  └─ HEAD       decoupled Detect on P3/P4/P5:
                  box branch  → CIoU + DFL (reg_max = 16) → xyxy
                  class branch → sigmoid scores for {empty, occupied}
                  NMS → final boxes
```

| Property | Value |
|---|---|
| Stages | 24 (11 backbone, 11 neck, 1 Detect + utilities) |
| Parameters | 9,428,566 (9,413,574 after fusion) |
| GFLOPs @ 640 px | 21.7 (21.4 fused) |
| Feature strides | P3/8, P4/16, P5/32 |
| Pretrained transfer | 493/499 keys from `yolo11s.pt`; only the class head is re-initialized |
| Scale | `s: [0.50, 0.50, 1024]`; switch to `m: [0.50, 1.00, 512]` for `yolo11m.pt` |

Augmentation is deliberately conservative — no rotation, shear, or perspective distortion — because parking-stall geometry carries the signal. Horizontal flips, mild translation/scale, HSV jitter, mosaic, and random erasing are used.

---

## Prerequisites and setup

- **Python 3.13** with CUDA-enabled PyTorch 2.15 (tested with CUDA 13.2)
- **Node.js 18+** and npm (only for frontend development)
- An NVIDIA GPU is recommended; inference also runs on CPU

All Python packages are pinned in `requirements.txt`. Use the project virtual environment — the system `python3` on the development machine has none of the required packages.

```bash
git clone https://github.com/heangdeniss/Computer_vision_parking_lot_Detection.git
cd Computer_vision_parking_lot_Detection

# Create the environment (adjust the interpreter path for your machine):
uv venv --python /home/heangdeniss/anaconda3/envs/eda/bin/python .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Frontend dependencies are only needed when changing the UI:

```bash
cd frontend
npm install
npm run build      # refresh frontend/dist/ served by FastAPI
cd ..
```

> On the original development machine, `node`/`npm`/`npx` are wrappers in `~/.local/bin` backed by `nodejs-bin` in `.venv`. On another machine, install Node normally.

---

## Dataset

Source: [Roboflow workspace — parking-space-bxjkf-jus8n, v1](https://app.roboflow.com/denischan/parking-space-bxjkf-jus8n/1) ([public dataset page](https://universe.roboflow.com/denischan/parking-space-bxjkf-jus8n/dataset/1)), licensed **CC BY 4.0**. The dataset folder is not committed; place the export at `parking-space.v1i.yolov11/` before training or evaluation.

```bash
unzip -o parking-space.v1i.yolov11.zip -d parking-space.v1i.yolov11
```

| Split | Images | Boxes |
|---|---:|---:|
| Train | 496 | 3,780 |
| Valid | 62 | 429 |
| Test | 62 | 514 |
| **Total** | **620** | **4,723** |

`data.yaml` uses paths relative to its own directory (`train/images`, `valid/images`, `test/images`), so it resolves from any working directory. Classes are `0: empty`, `1: occupied`; labels are normalized YOLO detection boxes (`class cx cy w h`).

Class distribution in training: 2,422 `empty` vs 1,358 `occupied` (~1.8:1).

---

## Training pipeline

Inspect the model and validate the plan before committing GPU time:

```bash
.venv/bin/python src/inspect_model.py     # layer table, shapes, GFLOPs, weight compatibility
.venv/bin/python src/train.py --dry-run   # checks config, nc, hyperparameters, weights
```

Fine-tune:

```bash
.venv/bin/python src/train.py --weights yolo11s.pt
```

The script builds the graph from `configs/yolo11_parking.yaml`, transfers pretrained weights, then trains with `configs/train_hyp.yaml` and `parking-space.v1i.yolov11/data.yaml`. Override any of `--epochs`, `--batch`, `--imgsz`, `--device`, `--name` on the CLI. Outputs go to `runs/detect/<run>/`.

Key hyperparameters:

| Group | Settings |
|---|---|
| Optimizer | AdamW, `lr0=0.001` → `lrf=0.01` cosine, momentum 0.937, weight decay 0.0005 |
| Schedule | 100 epochs, batch 8, 640 px, 3 warmup epochs, `close_mosaic=10`, patience 20 |
| Loss gains | box 7.5, cls 0.5, dfl 1.5 |
| Augmentation | HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0, erasing 0.4 |
| Deliberately disabled | rotation, shear, perspective, mixup, copy-paste, vertical flip |

Equivalent direct Ultralytics command:

```bash
yolo detect train model=yolo11s.pt data=parking-space.v1i.yolov11/data.yaml \
  cfg=configs/train_hyp.yaml project=runs/detect name=parking_yolo11s
```

---

## Evaluation

```bash
.venv/bin/python src/evaluate.py \
  --weights runs/detect/parking_yolo11s/weights/best.pt \
  --split test        # or: val
```

`src/evaluate.py` reports overall mAP50/mAP50-95, precision and recall, plus per-class precision, recall, AP, and confusion-matrix recall. Use `--split val` for model selection and reserve `test` for the final report. Plots and raw predictions are written under `runs/detect/eval/`.

Reproduce the analysis figures with the notebook `plot_results.ipynb`, which reads `runs/detect/parking_yolo11s/results.csv` and writes PNGs to `analysis_plots/`.

---

## Inference application

### Quick start

Host the prebuilt UI and API together:

```bash
.venv/bin/python -m uvicorn server.app:app --port 8000
```

Open **http://localhost:8000**.

For development with hot reload:

```bash
cd frontend && npm run dev     # http://localhost:5173/
```

Vite proxies `/api` to port 8000 and starts uvicorn automatically when it is not already running. No second terminal is needed. `GET /api/health` confirms the loaded checkpoint, device, classes, and internal batch size.

### Web interface

| Mode | What it does |
|---|---|
| **Images** | Drag-and-drop or browse for multiple JPG/PNG/HEIC files. Additional drops append to the selection; each card can be removed individually. One batched forward pass produces per-image counts, confidences, annotated previews, and expandable detection tables. |
| **Video · analysis** | `POST /api/video` samples every 1/5/15/30/60 frames (default 15), capped at 240 samples. Results appear as a clickable timeline with frame index, timestamp, and counts; selecting a frame shows the annotated image and a download link. |
| **Video · realtime** | Enable *Realtime (live boxes)* and press *Start live*. Uses the selected video or the webcam. Frames are drawn live with boxes and a 1 Hz FPS/count readout; *Stop live* releases the camera. |

Boxes and labels use **bold green for `empty`** and **bold red for `occupied`**, with a dark text outline for legibility in bright daylight. UI badges use the same colors.

### API reference

| Method | Endpoint | Fields | Returns |
|---|---|---|---|
| `GET` | `/api/health` | — | Model path, classes, device, batch size |
| `POST` | `/api/preview` | `file` | Browser-friendly JPEG data URL + dimensions (for HEIC) |
| `POST` | `/api/predict` | `files` (repeatable), `conf` (default 0.25) | Per-image counts, detections, annotated image |
| `POST` | `/api/frame` | `file`, `conf` | Boxes + counts only, for the realtime loop |
| `POST` | `/api/video` | `file`, `conf`, `stride`, `limit` | Frame timeline, totals, per-frame boxes |

Example:

```bash
curl -F "files=@photo.jpg" -F "files=@photo2.heic" -F "conf=0.25" \
  http://localhost:8000/api/predict
```

```json
{
  "results": [{
    "name": "photo.jpg",
    "counts": { "empty": 14, "occupied": 5 },
    "detections": [
      { "class": "empty", "confidence": 0.9011, "bbox": [792.6, 535.7, 1256.8, 955.3] }
    ],
    "image": "data:image/jpeg;base64,...",
    "width": 1920, "height": 1080
  }]
}
```

### Performance engineering

- **Fusion** — Conv+BN layers are merged at startup.
- **FP16** — the checkpoint runs in half precision on CUDA; CPU falls back to FP32.
- **Warm-up** — one dummy inference at startup removes cold-start latency from the first request.
- **Fixed internal batch of 4** — multiple uploads are inferred in a single pass; safe on 4 GB, no UI knob.
- **HEIC + EXIF** — `pillow-heif` decodes iPhone photos and `ImageOps.exif_transpose` fixes orientation before inference.
- **Realtime backpressure** — the browser sends at most ~12 requests/s and never overlaps them; slow frames are skipped rather than queued. Only boxes cross the wire, not annotated images.

---

## Measured performance

| Path | Measurement |
|---|---|
| Full validation inference | 6.0–6.7 ms/image (0.4 ms preprocess, ~1 ms postprocess) |
| Batched API, FP32 baseline | ~11.8 ms/image |
| Batched API, fused FP16 | ~10.3 ms/image |
| Realtime round trip (`/api/frame`) | ~28 ms/frame → ~35 fps server ceiling |
| Training throughput | 100 epochs in ~15.7 min |

Timings were measured on the RTX 3050 Laptop GPU with 640 px inputs. CPU inference is supported and considerably slower.

---

## Limitations and honest interpretation

- **Split leakage.** The Roboflow export was split by frame, not by video: 17 of 37 training videos also appear in validation and 18 in test, and some adjacent frames are near-duplicates. Reported mAP is therefore optimistic. Re-split by *video ID* for a strict benchmark.
- **Class imbalance.** Training has ~1.8× more `empty` than `occupied` boxes. Per-class recall is currently balanced, but this should be watched after any data change.
- **Missed spaces dominate errors.** At `conf=0.25`, there are 66 misses versus 18 false alarms and zero class swaps. Small, distant, or heavily occluded spaces are the likely source.
- **Far-view handling.** Tiled zoom was evaluated but removed because it over-counted at tile borders; manual perspective alignment was not implemented. Re-add tiling with stricter NMS if distant spaces matter more than false alarms.
- **Realtime boxes lag by design.** A displayed frame can be one inference behind the newest video frame. Webcam mode requires `localhost` or HTTPS for camera permission.
- **FP16 confidence shift.** Half precision can move borderline detections slightly across the threshold.
- **Dataset not in repo.** Training and evaluation require the Roboflow export described above.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | Use `.venv/bin/python`, not the system interpreter. |
| HEIC thumbnail says "preview unavailable" | The API must be running — `/api/preview` generates the JPEG preview. |
| Webcam does not start | Use `localhost` or HTTPS and allow camera access in the browser. |
| Port 8000 already in use | Stop the existing process; Vite reuses a healthy API automatically. |
| `npm: command not found` | Install Node.js 18+ and npm. |
| CUDA out of memory | Lower `batch` in `configs/train_hyp.yaml`; API batch is fixed at 4. |

---

## Attribution

- Dataset: [Roboflow parking-space (v1, workspace)](https://app.roboflow.com/denischan/parking-space-bxjkf-jus8n/1) / [public page](https://universe.roboflow.com/denischan/parking-space-bxjkf-jus8n/dataset/1) — CC BY 4.0.
- Model architecture and training code follow [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) — AGPL-3.0.
- Frontend: React 18 and Vite 5 (MIT).
- Python stack: Ultralytics 8.4.215, PyTorch 2.15.0, OpenCV 5.1.0.95, FastAPI 0.165.0, Uvicorn 0.54.0, pillow-heif 1.8.1.
