"""Loss functions y gradientes para MSE, BCE, cross-entropy."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """∂MSE/∂y_pred. Mean over batch absorbido en factor 2/N."""
    n = y_true.size
    return 2.0 * (y_pred - y_true) / n


def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def bce_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    n = y_true.size
    return ((p - y_true) / (p * (1.0 - p))) / n


def cross_entropy(y_true_onehot: np.ndarray, y_pred_softmax: np.ndarray) -> float:
    p = np.clip(y_pred_softmax, EPS, 1.0)
    return float(-np.sum(y_true_onehot * np.log(p)) / y_true_onehot.shape[0])


def cross_entropy_grad_with_softmax(
    y_true_onehot: np.ndarray, y_pred_softmax: np.ndarray
) -> np.ndarray:
    """Truco: ∂CE/∂z = softmax(z) - y_true. Se cancelan softmax_grad y log_grad."""
    return (y_pred_softmax - y_true_onehot) / y_true_onehot.shape[0]


LOSSES = {
    "mse": (mse, mse_grad),
    "bce": (bce, bce_grad),
    "cross_entropy": (cross_entropy, cross_entropy_grad_with_softmax),
}
