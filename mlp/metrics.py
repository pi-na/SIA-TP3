"""Métricas multi-clase: accuracy, per-class precision/recall/f1, confusion matrix."""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, np.ndarray]:
    """Devuelve dict con arrays de precision/recall/f1 por clase (shape (n_classes,))."""
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_average(arr: np.ndarray) -> float:
    return float(arr.mean())


def weighted_average(arr: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    weights = np.array([(y_true == c).sum() for c in range(n_classes)], dtype=float)
    weights /= weights.sum()
    return float((arr * weights).sum())
