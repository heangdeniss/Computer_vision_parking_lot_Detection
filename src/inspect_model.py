#!/usr/bin/env python3
"""Architecture-only inspection for the parking-space YOLOv11 detector.

Builds the model from configs/yolo11_parking.yaml, prints a layer-by-layer table
with tensor shapes / parameters / GFLOPs, runs one dummy forward pass and checks
whether the pretrained yolo11s.pt / yolo11m.pt checkpoints can be transferred.

No training is performed by this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    if index == last_index:
        return "head/Detect"
    return "neck/PAN-FPN"


def check_weights(model, name: str) -> None:
    """Report how much of a pretrained checkpoint transfers into this config."""
    ref = YOLO(name, verbose=False)
    src, dst = ref.model.state_dict(), model.model.state_dict()
    common = set(src) & set(dst)
    mismatched = {k for k in common if src[k].shape != dst[k].shape}
    head_prefix = f"model.{len(model.model.model) - 1}."
    structural = sorted(k for k in mismatched if not k.startswith(head_prefix))
    src_yaml = getattr(ref.model, "yaml", {})

    print(f"\n  {name}: source scale={src_yaml.get('scale') or 'n/s/m/l/x (from filename)'}, "
          f"source nc={src_yaml.get('nc')}, source params={sum(v.numel() for v in src.values()):,}")
    print(f"    keys matched: {len(common) - len(mismatched)}/{len(dst)}  "
          f"(head-only shape changes: {len(mismatched) - len(structural)}, "
          f"structural mismatches: {len(structural)})")
    if not common or structural:
        print("    verdict: NOT compatible with this config scale.")
        print("             Switch the `s:` line in configs/yolo11_parking.yaml to "
              "`m: [0.50, 1.00, 512]` for yolo11m.pt, then re-run.")
        for k in structural[:4]:
            print(f"       {k}: {tuple(src[k].shape)} vs {tuple(dst[k].shape)}")
    else:
        print("    verdict: COMPATIBLE. Backbone/neck/box head transfer as-is; "
              "the class-score branch is re-initialized for nc=2.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/yolo11_parking.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-weight-check", action="store_true",
                        help="skip downloading/comparing yolo11s.pt and yolo11m.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = YOLO(str(args.config), verbose=False)
    inner = model.model
    seq = inner.model  # nn.Sequential of the 24 stages
    n_backbone = len(model.yaml["backbone"])

    print("=" * 100)
    print(f"YOLOv11 parking detector | config: {args.config}")
    print(f"nc={model.yaml['nc']} names={list(model.names.values())} scale={model.yaml.get('scale') or 's (default)'}")
    print("=" * 100)

    # Dummy forward pass with per-layer output capture.
    shapes: dict[int, object] = {}
    handles = [m.register_forward_hook(lambda mod, inp, out, i=i: shapes.__setitem__(i, out))
               for i, m in enumerate(seq)]
    inner.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        output = inner(dummy)
    for h in handles:
        h.remove()

    print("\nLAYER-BY-LAYER (input 1x3x%d x%d, inference mode)" % (args.imgsz, args.imgsz))
    print(f"{'#':>3}  {'stage':<13} {'module':<28} {'params':>10}  output shape")
    print("-" * 100)
    for i, module in enumerate(seq):
        print(f"{i:>3}  {stage_of(i, n_backbone, len(seq) - 1):<13} "
              f"{type(module).__name__:<28} "
              f"{sum(p.numel() for p in module.parameters()):>10,}  {tensor_shape(shapes[i])}")

    print("\nMULTI-SCALE DETECTION OUTPUTS")
    strides = [int(s) for s in seq[-1].stride.tolist()]
    for level, (idx, stride) in enumerate(zip(seq[-1].f, strides), start=3):
        print(f"  P{level}/{stride:<2} (stride {stride:>2})  layer {idx:>2} "
              f"feature map: {tensor_shape(shapes[idx])}")

    print("\nFINAL HEAD OUTPUT")
    print(f"  {tensor_shape(output)}   (list per scale: [boxes xyxy, class scores])")

    print("\nMODEL SUMMARY")
    layers, params, grads, gflops = model.info(verbose=True, imgsz=args.imgsz)
    print(f"  layers={layers}  parameters={params:,}  gradients={grads:,}  GFLOPs={gflops:.1f} @ {args.imgsz}px")
    print(f"  strides={strides}  classes={model.yaml['nc']}  dummy forward pass: OK")

    if not args.no_weight_check:
        print("\nPRETRAINED WEIGHT COMPATIBILITY")
        for name in ("yolo11s.pt", "yolo11m.pt"):
            check_weights(model, name)

    print("\nDone. Architecture only - no training started.")


if __name__ == "__main__":
    main()
