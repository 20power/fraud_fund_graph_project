from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return .5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def top_fraction_metrics(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float | int]:
    n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-scores)[:n]
    hits = int(np.asarray(y_true)[order].sum())
    positives = int(np.asarray(y_true).sum())
    return {
        "n": n,
        "hits": hits,
        "recall": float(hits / positives) if positives else 0.0,
        "precision": float(hits / n),
    }


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "top_1_percent": top_fraction_metrics(y_true, scores, .01),
        "top_3_percent": top_fraction_metrics(y_true, scores, .03),
        "top_5_percent": top_fraction_metrics(y_true, scores, .05),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    iterations: int = 300,
    seed: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    values: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "top_5_recall": []}
    for _ in range(iterations):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[idx]
        if y_sample.min() == y_sample.max():
            continue
        s_sample = scores[idx]
        values["roc_auc"].append(roc_auc_score(y_sample, s_sample))
        values["pr_auc"].append(average_precision_score(y_sample, s_sample))
        values["top_5_recall"].append(top_fraction_metrics(y_sample, s_sample, .05)["recall"])
    return {
        key: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if v else [None, None]
        for key, v in values.items()
    }


def improvement(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in ("roc_auc", "pr_auc"):
        absolute = candidate[key] - baseline[key]
        output[key] = {
            "absolute": float(absolute),
            "percentage_points": float(absolute * 100),
            "relative": float(absolute / baseline[key]) if baseline[key] else None,
        }
    c_top = candidate["top_5_percent"]["recall"]
    b_top = baseline["top_5_percent"]["recall"]
    output["top_5_recall"] = {
        "absolute": float(c_top - b_top),
        "percentage_points": float((c_top - b_top) * 100),
        "relative": float((c_top - b_top) / b_top) if b_top else None,
    }
    return output
