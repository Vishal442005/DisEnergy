"""Metrics Reloaded-compliant segmentation and classification evaluation."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _surface_pixels(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return mask ^ eroded


def compute_hd95(pred: np.ndarray, target: np.ndarray, max_penalty: float = 128.0) -> float:
    """Compute symmetric 95th-percentile Hausdorff Distance (pixels)."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(target, dtype=bool)
    if not np.any(p) and not np.any(t):
        return 0.0
    if not np.any(p) or not np.any(t):
        return float(max_penalty)

    sp = _surface_pixels(p)
    st = _surface_pixels(t)
    dt_t = ndi.distance_transform_edt(~st)
    dt_p = ndi.distance_transform_edt(~sp)
    d_p_to_t = dt_t[sp]
    d_t_to_p = dt_p[st]
    all_d = np.concatenate([d_p_to_t, d_t_to_p])
    return float(np.percentile(all_d, 95))


def compute_boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance_px: float = 2.0) -> float:
    """Compute boundary F1-score within a pixel distance tolerance."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(target, dtype=bool)
    if not np.any(p) and not np.any(t):
        return 1.0
    if not np.any(p) or not np.any(t):
        return 0.0

    sp = _surface_pixels(p)
    st = _surface_pixels(t)
    dt_t = ndi.distance_transform_edt(~st)
    dt_p = ndi.distance_transform_edt(~sp)

    prec = float((dt_t[sp] <= tolerance_px).mean()) if np.any(sp) else 0.0
    rec = float((dt_p[st] <= tolerance_px).mean()) if np.any(st) else 0.0
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def compute_segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute DSC, IoU, HD95, Boundary F1, and Background Rejection Rate (Specificity)."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(target, dtype=bool)

    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())

    dsc = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    hd95 = compute_hd95(p, t, max_penalty=float(max(p.shape)))
    bf1 = compute_boundary_f1(p, t)

    return {
        "dsc": float(dsc),
        "iou": float(iou),
        "hd95": float(hd95),
        "boundary_f1": float(bf1),
        "specificity": float(specificity),
    }


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over predicted confidence bins."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(bins):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def compute_classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute Macro-F1, AUROC, PR-AUC, MCC, and ECE."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float32)
    preds = p.argmax(axis=1)

    _, _, macro_f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y, preds))

    if p.shape[1] == 2:
        auroc = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) > 1 else 0.5
        pr_auc = float(average_precision_score(y, p[:, 1])) if len(np.unique(y)) > 1 else 0.5
    else:
        auroc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        pr_auc = float(
            np.mean([average_precision_score((y == c).astype(int), p[:, c]) for c in range(p.shape[1])])
        )

    return {
        "macro_f1": float(macro_f1),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "ece": compute_ece(p, y),
    }


def worst_decile_score(values: np.ndarray, lower_is_better: bool = False) -> float:
    """Return 10th-percentile (or 90th-percentile for HD95) worst-case robustness score."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 90 if lower_is_better else 10))
