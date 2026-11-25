"""Metric computation. Positive class = 1 = malignant throughout."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_KEYS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]


def confusion(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion(y_true, y_pred)
    spec_den = cm["tn"] + cm["fp"]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": cm["tn"] / spec_den if spec_den else 0.0,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }
    out = {k: float(v) for k, v in out.items()}
    out["confusion_matrix"] = cm
    out["threshold"] = float(threshold)
    return out


def bootstrap_ci(y_true, y_prob, threshold=0.5, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """Percentile bootstrap CIs over test samples (resampling rows with replacement)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = threshold_metrics(y_true[idx], y_prob[idx], threshold)
        for k in METRIC_KEYS:
            samples[k].append(m[k])
    return {
        k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
        for k, v in samples.items()
    }


def report(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["benign", "malignant"],
        output_dict=True, zero_division=0,
    )


def _thin(arrays: list[np.ndarray], max_points: int = 200) -> list[np.ndarray]:
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return [a[idx] for a in arrays]


def curves(y_true, y_prob) -> dict:
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    fpr, tpr = _thin([fpr, tpr])
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    prec, rec = _thin([prec, rec])
    n_bins = 8
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "roc": [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)],
        "pr": [{"recall": float(a), "precision": float(b)} for a, b in zip(rec, prec)],
        "calibration": [
            {"mean_predicted": float(a), "fraction_positive": float(b)}
            for a, b in zip(mean_pred, frac_pos)
        ],
    }


def threshold_sweep(y_true, y_prob, steps: int = 99) -> list[dict]:
    out = []
    for t in np.linspace(0.01, 0.99, steps):
        m = threshold_metrics(y_true, y_prob, float(t))
        out.append({
            "threshold": round(float(t), 2),
            "recall": m["recall"], "specificity": m["specificity"],
            "precision": m["precision"], "f1": m["f1"], "accuracy": m["accuracy"],
            **m["confusion_matrix"],
        })
    return out
