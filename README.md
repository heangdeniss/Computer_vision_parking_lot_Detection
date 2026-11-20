# Parking-lot occupancy detection — YOLOv11

Two classes: **0 = empty**, **1 = occupied**. Dataset: `parking-space.v1i.yolov11/` (Roboflow v1, 620 images, 4,723 boxes).

> **Status: architecture scaffolded, training NOT started.** Inspect the files and run `src/inspect_model.py` first.

## Directory structure

```
Parking-lot Detection/
├── .venv/                    # project-local Python env (use .venv/bin/python)
├── plot_results.ipynb        # plots runs/detect/parking_yolo11s/results.csv
├── configs/
│   ├── yolo11_parking.yaml   # Full YOLOv11 detection graph, nc=2, scale s (matches yolo11s.pt)
│   └── train_hyp.yaml        # Fine-tuning hyperparameters + augmentation
├── src/
│   ├── inspect_model.py      # Layer table, params/GFLOPs, dummy forward, weight compatibility
│   ├── train.py              # Build from config -> transfer pretrained weights -> fine-tune
│   └── evaluate.py           # Test-set mAP and per-class recall (empty vs occupied)
├── server/app.py             # FastAPI inference API (uses runs/.../weights/best.pt)
├── frontend/                 # React + Vite upload UI (dist/ is already built)
├── parking-space.v1i.yolov11/
│   ├── data.yaml             # train/val/test paths relative to this folder
│   ├── train/                # 496 images + 496 labels
│   ├── valid/                # 62 images + 62 labels
│   └── test/                 # 62 images + 62 labels
├── parking-space.v1i.yolov11.zip
├── yolo11s.pt / yolo11m.pt   # pretrained weights (downloaded during inspection)
└── runs/                     # created by train.py / evaluate.py
```

`dataset_problem.md`, `annotation_plan.md` and `unlabeled_images/` belong to an older **v2i** export and are unrelated to this dataset.

## Architecture

YOLOv11 (small scale) is a one-stage anchor-free detector. Colours in the diagram map to the stages commented in `configs/yolo11_parking.yaml`.

```
Input 640x640x3
      │
      ▼
┌────────────────────────── BACKBONE ─────────────────────────────────┐
│ Conv 3x3/s2 (P1/2) -> Conv 3x3/s2 (P2/4)                            │
│ C3k2 x2 -> Conv (P3/8) -> C3k2 x2   ─────────────► P3 low-level      │
│ Conv (P4/16) -> C3k2 x2 (k=3)       ─────────────► P4 mid-level      │
│ Conv (P5/32) -> C3k2 x2 -> SPPF(k=5) -> C2PSA (spatial attention)   │
│                                                  ─► P5 deep features │
└──────────────────────────────────────────────────────────────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
┌──────────────────────── NECK (PAN-FPN) ─────────────────────────────┐
│ top-down:  upsample(P5) + Concat(P4) -> C3k2 x2                      │
│            upsample     + Concat(P3) -> C3k2 x2  ──► P3/8 feats      │
│ bottom-up: Conv/s2 + Concat(head P4) -> C3k2 x2  ──► P4/16 feats     │
│            Conv/s2 + Concat(backbone P5) -> C3k2 x2 ─► P5/32 feats   │
└──────────────────────────────────────────────────────────────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
┌──────────────────────── DETECT HEAD ────────────────────────────────┐
│ anchor-free, decoupled per scale:                                    │
│   box branch  -> CIoU + DFL (reg_max=16)  -> xyxy                    │
│   cls branch  -> sigmoid scores for {empty, occupied}                │
│ predictions from P3/8 (small), P4/16 (medium), P5/32 (large)         │
└──────────────────────────────────────────────────────────────────────┘
      ▼
NMS -> final boxes, class + confidence
```

Scale `s` = depth 0.50 / width 0.50 / max channels 1024. To use `yolo11m.pt`, change the single `s:` line in `configs/yolo11_parking.yaml` to `m: [0.50, 1.00, 512]` and pass `--weights yolo11m.pt`.

## Setup

A project-local virtual environment `.venv/` holds CUDA PyTorch and all dependencies (Python 3.13). Always use its Python — the system `/usr/bin/python3` has none of the packages.

```bash
cd "/media/heangdeniss/New Volume/Parking-lot Detection"
PY=.venv/bin/python

# Recreate it if needed:
uv venv --python /home/heangdeniss/anaconda3/envs/eda/bin/python .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Step-by-step

**1. Verify the dataset** (already done in this workspace):

```bash
unzip -o parking-space.v1i.yolov11.zip -d parking-space.v1i.yolov11
$PY -c "from ultralytics.data.utils import check_det_dataset; d=check_det_dataset('parking-space.v1i.yolov11/data.yaml'); print(d['nc'], d['names'], d['train'], d['val'], d['test'])"
```

`data.yaml` uses paths relative to its own directory (no `../`), so it resolves regardless of the current working directory.

**2. Inspect the architecture — safe, does not train:**

```bash
$PY src/inspect_model.py
```

Prints all 24 stages with parameters and tensor shapes, P3/P4/P5 feature maps, GFLOPs at 640px, a dummy forward pass, and whether `yolo11s.pt` / `yolo11m.pt` weights transfer. Add `--no-weight-check` to skip checkpoint downloads.

**3. Check the training plan without training:**

```bash
$PY src/train.py --dry-run
```

**4. Train (when you are ready):**

```bash
$PY src/train.py --weights yolo11s.pt
```

CLI flags `--epochs`, `--batch`, `--imgsz`, `--device` override `configs/train_hyp.yaml`. Output goes to `runs/detect/parking_yolo11s/`.

Equivalent one-liner: `yolo detect train model=yolo11s.pt data=parking-space.v1i.yolov11/data.yaml cfg=configs/train_hyp.yaml`

**5. Evaluate on held-out test images:**

```bash
$PY src/evaluate.py --weights runs/detect/parking_yolo11s/weights/best.pt --split test
```

Reports overall mAP50/mAP50-95 plus per-class precision, recall, mAP and confusion-matrix recall for `empty` and `occupied`. Use `--split val` for model-selection checks.

**6. Try your own images or video in the browser (React frontend):**

```bash
cd "/media/heangdeniss/New Volume/Parking-lot Detection"
.venv/bin/python -m uvicorn server.app:app --port 8000
```

Open **http://localhost:8000** — the built React UI is served there. Use the **Images** tab for one or many JPG/PNG/HEIC files, or the **Video** tab for MP4/MOV/AVI. Drop files onto the upload area, adjust confidence, then click *Detect*.

- **Images** — drag-and-drop or browse for multiple images; dropping more appends to the selection and each card has a remove button. One batched inference pass returns per-image counts, boxes with confidences, annotated previews and expandable detection tables.
- **Video — analysis** — `POST /api/video` samples every Nth frame (1/5/15/30/60, default 15), capped at 240 frames. Results appear as a clickable timeline with frame number, timestamp and counts; selecting a frame shows the full annotated image and boxes, with a download link.
- **Video — realtime** — tick *Realtime (live boxes)* and click *Start live*. Boxes are drawn on each frame as it plays, using the uploaded video if selected, otherwise the webcam (browser prompts for camera access; `localhost` counts as a secure context). Frames go to the lean `POST /api/frame` endpoint, which returns boxes only — the browser draws them, so no annotated JPEG is transferred. Requests are throttled to ~12/s and never overlap; a live FPS readout and per-class counts update every second. *Stop live* releases the camera. Measured server ceiling: ~35 fps on the RTX 3050 Laptop.
- **Optimized inference** — the checkpoint is fused (Conv+BN merged), converted to FP16 on CUDA, warmed up at startup, and uploads run in one fixed internal batch of 4. Measured on the RTX 3050 Laptop: ~11.8 → ~10.3 ms/image (and no slow first request).
- **Labels** — bold green for `empty`, bold red for `occupied`, with dark-outlined text so they stay readable in daylight. UI badges use the same colours.
- **HEIC / orientation** — `pillow-heif` decodes iPhone HEIC; `POST /api/preview` supplies a JPEG copy for browsers, and EXIF rotation is applied before inference.
- `GET /api/health` reports the loaded model, device and internal batch size.

For UI editing, just run Vite — it prints `http://localhost:5173/` and starts the inference API automatically (no second terminal needed):

```bash
cd "/media/heangdeniss/New Volume/Parking-lot Detection/frontend"
npm run dev
```

`node`, `npm` and `npx` are on `PATH` via wrappers in `~/.local/bin`, backed by `nodejs-bin` installed in `.venv` (this machine has no system Node). After edits, run `npm run build` to refresh the version FastAPI serves from `frontend/dist/`.

## Known data caveats

- Class imbalance in train: 2,422 `empty` vs 1,358 `occupied` (~1.8:1). Watch `occupied` recall.
- Roboflow split frames randomly: 17/37 train videos also appear in `valid`, 18 in `test`, and a few near-identical adjacent frames cross splits. Validation numbers will be slightly optimistic; re-split by video for a strict benchmark.
- All labels are bounding boxes (not masks) — use the `detect` models, not `*-seg`.
# Computer_vision_parking_lot_Detection
