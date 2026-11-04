"""Validation-only threshold selection and shared binary anomaly metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
)


def _validate(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores)
            or not len(y) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(scores).all()):
        raise ValueError("Expected aligned binary labels and finite one-dimensional scores.")
    if len(np.unique(y)) != 2:
        raise ValueError("Both normal and fraud labels are required for evaluation.")
    return y, scores


def select_f1_threshold(y_val, scores):
    """Maximize validation F1 using score >= threshold; prefer higher threshold on ties."""
    y, scores = _validate(y_val, scores)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def evaluate_scores(y, scores, threshold):
    y, scores = _validate(y, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn))
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": float(fp / (fp + tn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
