#!/usr/bin/env python3
"""Architecture-only inspection for the parking-space YOLOv11 detector.

Example:
    .venv/bin/python src/yolo/inspect_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO


def tensor_shape(x) -> str:
    if isinstance(x, (list, tuple)):
        return " + ".join(tensor_shape(v) for v in x[:3])
    if isinstance(x, dict):
        return ", ".join(f"{k}:{tuple(v.shape) if hasattr(v, 'shape') else '?'}" for k, v in x.items())
    return str(tuple(x.shape)) if hasattr(x, "shape") else type(x).__name__


def stage_of(index: int, n_backbone: int, last_index: int) -> str:
    if index < n_backbone:
        return "backbone"
    return "head/Detect" if index == last_index else "neck/PAN-FPN"


def check_weights(model, name: str) -> None:
    ref = YOLO(name, verbose=False)
    src, dst = ref.model.state_dict(), model.model.state_dict()
    common = set(src) & set(dst)
    mismatched = {k for k in common if src[k].shape != dst[k].shape}
    head_prefix = f"model.{len(model.model.model) - 1}."
    structural = sorted(k for k in mismatched if not k.startswith(head_prefix))
    src_yaml = getattr(ref.model, "yaml", {})

    print(f"\n  {name}: source scale={src_yaml.get('scale') or 'from filename'}, "
          f"source nc={src_yaml.get('nc')}, source params={sum(v.numel() for v in src.values()):,}")
    print(f"    keys matched: {len(common) - len(mismatched)}/{len(dst)} "
          f"(head-only: {len(mismatched) - len(structural)}, structural: {len(structural)})")
    if not common or structural:
        print("    verdict: NOT compatible with this config scale.")
    else:
        print("    verdict: COMPATIBLE (class head re-initialized for nc=2).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/yolo11_parking.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-weight-check", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = YOLO(str(args.config), verbose=False)
    inner, seq = model.model, model.model.model
    n_backbone = len(model.yaml["backbone"])

    print("=" * 96)
    print(f"YOLOv11 parking detector | config: {args.config} | nc={model.yaml['nc']}")
    print("=" * 96)

    shapes: dict[int, object] = {}
    handles = [m.register_forward_hook(lambda _mod, _inp, out, i=i: shapes.__setitem__(i, out))
               for i, m in enumerate(seq)]
    inner.eval()
    with torch.no_grad():
        output = inner(torch.zeros(1, 3, args.imgsz, args.imgsz))
    for h in handles:
        h.remove()

    print(f"\n{'#':>3}  {'stage':<13} {'module':<28} {'params':>10}  output shape")
    print("-" * 96)
    for i, module in enumerate(seq):
        print(f"{i:>3}  {stage_of(i, n_backbone, len(seq) - 1):<13} "
              f"{type(module).__name__:<28} {sum(p.numel() for p in module.parameters()):>10,}  {tensor_shape(shapes[i])}")

    layers, params, grads, gflops = model.info(verbose=True, imgsz=args.imgsz)
    print(f"\nSummary: layers={layers} params={params:,} grads={grads:,} GFLOPs={gflops:.1f} @ {args.imgsz}px")
    print(f"Final head output: {tensor_shape(output)}")

    if not args.no_weight_check:
        for name in ("yolo11s.pt", "yolo11m.pt"):
            check_weights(model, name)


if __name__ == "__main__":
    main()
