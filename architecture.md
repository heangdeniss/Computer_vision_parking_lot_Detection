# YOLOv11 parking-occupancy detector — architecture & training process

**Status:** dataset restored and architecture scaffolded; **training has not been started yet**.
Classes: `0 = empty`, `1 = occupied`. Working directory below is `PROJECT`.

```bash
PROJECT="/media/heangdeniss/New Volume/Parking-lot Detection"
PY=.venv/bin/python   # project-local env (system python3 has no ultralytics)
cd "$PROJECT"
```

## 1. Architecture (what is being trained)

Model: Ultralytics YOLOv11 **small** (`configs/yolo11_parking.yaml`, `nc: 2`), pretrained from `yolo11s.pt`.

```
Input 640x640x3
  │
  ├─ BACKBONE   Conv/Conv/C3k2 -> Conv/C3k2 -> Conv/C3k2
  │             -> SPPF -> C2PSA (spatial attention)
  │             emits P3/8, P4/16, P5/32 feature maps
  │
  ├─ NECK       PAN-FPN: top-down Upsample + Concat + C3k2,
  │             then bottom-down Conv + Concat + C3k2
  │
  └─ HEAD       decoupled anchor-free Detect on P3/P4/P5:
                box branch (CIoU + DFL), class branch (2 scores), NMS
```

Inspected facts (`$PY src/inspect_model.py`):

| Item | Value |
|---|---|
| Stages | 24 (11 backbone, 11 neck, 1 Detect) |
| Parameters / gradients | 9,428,566 / 9,428,550 |
| GFLOPs @ 640px | 21.7 |
| Strides | P3/8 = `(1,128,80,80)`, P4/16 = `(1,256,40,40)`, P5/32 = `(1,512,20,20)` |
| Pretrained transfer | `yolo11s.pt`: 493/499 keys, 6 head-only changes (class branch rebuilt); `yolo11m.pt` requires switching the scale line to `m: [0.50, 1.00, 512]` |

## 2. Dataset state (verified)

| Split | Images | Labels |
|---|---|---|
| train | 496 | 496 |
| valid | 62 | 62 |
| test | 62 | 62 |

`parking-space.v1i.yolov11/data.yaml` uses split paths relative to its own folder, `nc: 2`, `names: [empty, occupied]`. Boxes are already in YOLO detection format (`class cx cy w h`, normalized).

## 3. Training procedure

### Step 0 — one-time setup

```bash
unzip -o parking-space.v1i.yolov11.zip -d parking-space.v1i.yolov11   # restores labels + data.yaml
uv venv --python /home/heangdeniss/anaconda3/envs/eda/bin/python .venv # only if .venv/ is missing
uv pip install --python .venv/bin/python -r requirements.txt          # only on first setup
```

If the pretrained checkpoints are missing they are auto-downloaded by `inspect_model.py` / `train.py` into `PROJECT` (`yolo11s.pt`, `yolo11m.pt`).

### Step 1 — inspect (no training)

```bash
$PY src/inspect_model.py              # layer table, GFLOPs, dummy forward, weight compatibility
$PY src/train.py --dry-run            # validates nc, config, hyperparameters and weight transfer
```

### Step 2 — fine-tune

```bash
$PY src/train.py --weights yolo11s.pt
```

This builds the model from `configs/yolo11_parking.yaml`, transfers pretrained weights (class-score branch re-initialized for 2 classes), then calls `model.train(cfg=configs/train_hyp.yaml, data=parking-space.v1i.yolov11/data.yaml, ...)`.

Optional CLI overrides: `--epochs`, `--batch`, `--imgsz`, `--device`, `--name`, `--weights yolo11m.pt` (after editing the scale line).

Hyperparameters (`configs/train_hyp.yaml`): AdamW, `lr0=0.001` cosine to `lrf=0.01`, `epochs=100`, `batch=8`, `imgsz=640`, 3 warmup epochs, `close_mosaic=10`, `patience=20`, losses `box=7.5 cls=0.5 dfl=1.5`, moderate HSV/translate/scale/flip/mosaic augmentation, no rotation/shear/perspective (keeps stall geometry intact).

Artifacts: `runs/detect/parking_yolo11s/weights/{best.pt,last.pt}`, `results.csv`, confusion matrix and PR plots.

Equivalent CLI (no `src/train.py`):

```bash
yolo detect train model=yolo11s.pt \
  data=parking-space.v1i.yolov11/data.yaml \
  cfg=configs/train_hyp.yaml \
  project=runs/detect name=parking_yolo11s
```

### Step 3 — evaluate on held-out test images

```bash
$PY src/evaluate.py \
  --weights runs/detect/parking_yolo11s/weights/best.pt \
  --split test
```

Reports overall mAP50 / mAP50-95 plus per-class precision, recall, mAP and confusion-matrix recall for `empty` and `occupied`. Use `--split val` for model selection before touching the test set.

### Step 4 — record the outcome

Fill in after training:

| Field | Value |
|---|---|
| Date / run name | — |
| Weights used | — |
| Epochs actually run / early stop | — |
| Best val mAP50 / mAP50-95 | — |
| Test mAP50 / mAP50-95 | — |
| Per-class recall (empty / occupied) | — |
| Best checkpoint path | — |

## 4. Known risks before trusting the numbers

- Train class mix: 2,422 `empty` vs 1,358 `occupied` (~1.8:1). Watch `occupied` recall.
- Roboflow split by frame, not by video: 17/37 train videos also appear in `valid`, 18 in `test`; a few adjacent frames are near duplicates. Reported mAP is optimistic — re-split by video for a strict benchmark.
- `dataset_problem.md`, `annotation_plan.md` and `unlabeled_images/` refer to an older v2i export; they are unrelated to this dataset.
