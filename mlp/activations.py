"""Funciones de activación + gradientes. Funciones puras, no clases."""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide numéricamente estable (mask-based)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out


def sigmoid_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Gradiente. a = sigmoid(z) ya computado."""
    return a * (1.0 - a)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return 1.0 - a * a


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


def identity(z: np.ndarray) -> np.ndarray:
    return np.array(z, dtype=np.float64, copy=True)


def identity_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax estable: resta max por fila antes de exp.

    z must be 2-D of shape (batch, classes).
    """
    z = np.asarray(z, dtype=np.float64)
    z_shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)


# Softmax grad va junto con cross_entropy — no se usa aislado.
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "identity": (identity, identity_grad),
    "softmax": (softmax, None),
}
