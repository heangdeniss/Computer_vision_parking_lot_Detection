"""Detection evaluation metrics (IoU matching, COCO mAP, confusion matrix, plots)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """Greedy 1-to-1 prediction matching across IoU thresholds (Ultralytics convention)."""
    correct = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct

    ious = box_iou(gt_boxes, pred_boxes).cpu().numpy()
    same_cls = (gt_cls[:, None] == pred_cls[None, :]).cpu().numpy()
    for t_idx, thr in enumerate(iou_thresholds):
        gt_i, pred_i = np.where((ious >= thr) & same_cls)
        if len(gt_i) == 0:
            continue
        order = np.argsort(-ious[gt_i, pred_i])
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for g, p in zip(gt_i[order], pred_i[order]):
            gi, pi = int(g), int(p)
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
                correct[pi, t_idx] = True
    return correct


def update_confusion_matrix(
    cm: np.ndarray,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> None:
    """Update confusion matrix (rows = predicted + bg, cols = ground truth + bg)."""
    nc = cm.shape[0] - 1
    keep = pred_scores >= conf_thres
    p_boxes, p_cls = pred_boxes[keep], pred_cls[keep]

    if len(gt_boxes) == 0:
        for pc in p_cls.tolist():
            cm[int(pc), nc] += 1
        return
    if len(p_boxes) == 0:
        for gc in gt_cls.tolist():
            cm[nc, int(gc)] += 1
        return

    ious = box_iou(gt_boxes, p_boxes).cpu().numpy()
    gt_i, pred_i = np.where(ious >= iou_thres)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if len(gt_i):
        for g, p in zip(gt_i[np.argsort(-ious[gt_i, pred_i])], pred_i[np.argsort(-ious[gt_i, pred_i])]):
            gi, pi = int(g), int(p)
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
                cm[int(p_cls[pi]), int(gt_cls[gi])] += 1

    for gi in range(len(gt_boxes)):
        if gi not in matched_gt:
            cm[nc, int(gt_cls[gi])] += 1
    for pi in range(len(p_boxes)):
        if pi not in matched_pred:
            cm[int(p_cls[pi]), nc] += 1


def save_confusion_matrix_plot(
    cm: np.ndarray, names: list[str], save_path: Path, normalize: bool = False
) -> None:
    """Render a confusion matrix PNG with matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(names) + ["background"]
    mat = cm.astype(float)
    if normalize:
        col_sums = mat.sum(axis=0, keepdims=True)
        mat = np.divide(mat, col_sums, out=np.zeros_like(mat), where=col_sums > 0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
    thresh = (mat.max() if mat.size else 0.0) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


@torch.no_grad()
def evaluate_detector(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    names: list[str],
    conf_eval: float = 0.001,
    cm_conf: float = 0.25,
    amp: bool = True,
    plots_dir: Path | None = None,
) -> dict[str, Any]:
    """Run COCO/Ultralytics-compatible evaluation on a torchvision detector."""
    model.eval()
    prev_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = conf_eval

    nc = len(names)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    stats_tp: list[np.ndarray] = []
    stats_conf: list[np.ndarray] = []
    stats_pcls: list[np.ndarray] = []
    stats_tcls: list[np.ndarray] = []

    use_amp = amp and torch.device(device).type == "cuda"
    n_images = 0
    infer_seconds = 0.0

    try:
        for images, targets in loader:
            gpu_images = [img.to(device, non_blocking=True) for img in images]
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(gpu_images)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - t0
            n_images += len(images)

            for out, target in zip(outputs, targets):
                pred_boxes = out["boxes"].detach().float().cpu()
                pred_scores = out["scores"].detach().float().cpu()
                pred_cls = out["labels"].detach().long().cpu() - 1  # shift back to 0-based
                valid = pred_cls >= 0
                pred_boxes, pred_scores, pred_cls = pred_boxes[valid], pred_scores[valid], pred_cls[valid]

                gt_boxes = target["boxes"].float().cpu()
                gt_cls = target["labels"].long().cpu() - 1
                stats_tcls.append(gt_cls.numpy())

                if len(pred_boxes):
                    order = torch.argsort(pred_scores, descending=True)
                    pred_boxes, pred_scores, pred_cls = pred_boxes[order], pred_scores[order], pred_cls[order]
                    stats_tp.append(match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls))
                    stats_conf.append(pred_scores.numpy())
                    stats_pcls.append(pred_cls.numpy())

                update_confusion_matrix(
                    cm, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, conf_thres=cm_conf
                )
    finally:
        model.roi_heads.score_thresh = prev_thresh

    target_cls = np.concatenate(stats_tcls) if stats_tcls else np.zeros((0,), dtype=np.int64)
    if stats_tp:
        tp_arr = np.concatenate(stats_tp, axis=0)
        conf_arr = np.concatenate(stats_conf, axis=0)
        pcls_arr = np.concatenate(stats_pcls, axis=0)
    else:
        tp_arr = np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
        conf_arr = np.zeros((0,), dtype=np.float32)
        pcls_arr = np.zeros((0,), dtype=np.int64)

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    names_dict = {i: name for i, name in enumerate(names)}
    if len(tp_arr) and len(target_cls):
        _, _, p, r, _, ap, ap_classes, *_ = ap_per_class(
            tp_arr, conf_arr, pcls_arr, target_cls,
            plot=(plots_dir is not None), save_dir=plots_dir or Path("."),
            names=names_dict, prefix="Box",
        )
    else:
        p = np.zeros((0,), dtype=np.float64)
        r = np.zeros((0,), dtype=np.float64)
        ap = np.zeros((0, len(IOU_THRESHOLDS)), dtype=np.float64)
        ap_classes = np.zeros((0,), dtype=np.int64)

    if plots_dir is not None:
        save_confusion_matrix_plot(cm, names, plots_dir / "confusion_matrix.png", normalize=False)
        save_confusion_matrix_plot(cm, names, plots_dir / "confusion_matrix_normalized.png", normalize=True)

    per_class: dict[str, dict[str, Any]] = {}
    for idx, cls_id in enumerate(ap_classes.tolist()):
        col = int(cm[:-1, cls_id].sum())
        per_class[names[cls_id]] = {
            "class_id": cls_id,
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "map50": float(ap[idx, 0]),
            "map50_95": float(ap[idx].mean()),
            "cm_recall": float(cm[cls_id, cls_id] / col) if col else None,
            "instances": int((target_cls == cls_id).sum()),
        }

    real_cm = cm[:nc, :nc]
    return {
        "images": n_images,
        "instances": int(len(target_cls)),
        "precision": float(p.mean()) if len(p) else 0.0,
        "recall": float(r.mean()) if len(r) else 0.0,
        "map50": float(ap[:, 0].mean()) if len(ap) else 0.0,
        "map50_95": float(ap.mean()) if len(ap) else 0.0,
        "ms_per_image": round((infer_seconds / max(n_images, 1)) * 1000.0, 2),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "cross_class_swaps": int(real_cm.sum() - np.trace(real_cm)),
    }
