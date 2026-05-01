from __future__ import annotations

import numpy as np
import pytest

from mlp.metrics import (
    accuracy, per_class_metrics, macro_average, weighted_average,
    confusion_matrix, multiclass_metrics,
)


def test_accuracy_perfect():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = y_true.copy()
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_half():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    assert accuracy(y_true, y_pred) == 0.75


def test_per_class_metrics_balanced():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    pcm = per_class_metrics(y_true, y_pred, n_classes=3)
    # Class 0: TP=1, FP=0, FN=1 → precision=1.0, recall=0.5, f1=0.667
    assert np.isclose(pcm["precision"][0], 1.0)
    assert np.isclose(pcm["recall"][0], 0.5)
    # Class 1: TP=2, FP=1, FN=0 → precision=2/3, recall=1.0
    assert np.isclose(pcm["precision"][1], 2.0/3.0)
    assert np.isclose(pcm["recall"][1], 1.0)


def test_macro_vs_weighted():
    y_true = np.array([0]*80 + [1]*20)
    y_pred = np.array([0]*80 + [1]*10 + [0]*10)
    pcm = per_class_metrics(y_true, y_pred, n_classes=2)
    macro = macro_average(pcm["f1"])
    weighted = weighted_average(pcm["f1"], y_true, n_classes=2)
    # Class 0 dominates → weighted closer to f1[0], macro is mean
    assert macro != weighted


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 1, 2])
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
    assert cm[0, 0] == 2
    assert cm[1, 2] == 1
    assert cm[2, 1] == 1


def test_multiclass_metrics_returns_dict():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    m = multiclass_metrics(y_true, y_pred, n_classes=3)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert "precision" in m and len(m["precision"]) == 3
