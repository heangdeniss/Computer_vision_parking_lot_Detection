"""Shared YOLO dataset loader used by both YOLOv11 and Faster R-CNN.

Reads ``parking-space.v1i.yolov11/data.yaml`` and normalized YOLO bounding-box
labels (``class cx cy w h``) directly without duplicating or converting files.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F
from ultralytics.utils import YAML

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(data_yaml: Path) -> dict[str, Any]:
    """Load a YOLO data.yaml and resolve train/val/test image directories."""
    data_yaml = Path(data_yaml).resolve()
    raw = YAML.load(data_yaml)
    base = data_yaml.parent
    names_raw = raw["names"]
    names = [names_raw[i] for i in sorted(names_raw)] if isinstance(names_raw, dict) else list(names_raw)

    def resolve_split(key: str) -> Path | None:
        val = raw.get(key)
        if not val:
            return None
        path = Path(val)
        return path if path.is_absolute() else (base / path).resolve()

    return {
        "nc": int(raw["nc"]),
        "names": names,
        "train": resolve_split("train"),
        "val": resolve_split("val"),
        "test": resolve_split("test"),
    }


def labels_dir_for(images_dir: Path) -> Path:
    """Resolve the sibling YOLO labels/ folder for an images/ directory."""
    return images_dir.parent / "labels" if images_dir.name == "images" else images_dir.with_name("labels")


def parse_yolo_label_file(
    label_path: Path, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read normalized YOLO boxes (class cx cy w h) into pixel xyxy + 0-based class ids."""
    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = max(0.0, min(float(width), (cx - bw / 2.0) * width))
            y1 = max(0.0, min(float(height), (cy - bh / 2.0) * height))
            x2 = max(0.0, min(float(width), (cx + bw / 2.0) * width))
            y2 = max(0.0, min(float(height), (cy + bh / 2.0) * height))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


class YoloParkingDataset(Dataset):
    """PyTorch Dataset reading YOLO-format images & .txt boxes for torchvision detectors.

    Torchvision reserves label 0 for ``__background__``, so YOLO class ids
    ``{0: empty, 1: occupied}`` are shifted by +1 to ``{1, 2}``.
    """

    def __init__(
        self,
        images_dir: Path,
        names: list[str],
        augment: bool = False,
        fliplr: float = 0.5,
        color_jitter: dict[str, float] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = labels_dir_for(self.images_dir)
        self.names = list(names)
        self.augment = augment
        self.fliplr = fliplr
        self.jitter = ColorJitter(**color_jitter) if (augment and color_jitter) else None
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img_path = self.image_paths[idx]
        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        width, height = image.size

        boxes_np, cls_np = parse_yolo_label_file(self.labels_dir / f"{img_path.stem}.txt", width, height)

        if self.augment:
            if self.jitter is not None:
                image = self.jitter(image)
            if self.fliplr > 0.0 and random.random() < self.fliplr:
                image = F.hflip(image)
                if len(boxes_np):
                    boxes_np = boxes_np.copy()
                    boxes_np[:, [0, 2]] = width - boxes_np[:, [2, 0]]

        boxes = torch.from_numpy(boxes_np)
        labels = torch.from_numpy(cls_np + 1)  # +1 because 0 is __background__
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else torch.zeros((0,), dtype=torch.float32)
        )
        return F.to_tensor(image), {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "path": str(img_path),
            "orig_size": (width, height),
        }


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
