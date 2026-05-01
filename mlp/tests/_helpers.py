"""Helpers for testing analytical gradients vs numerical."""
from __future__ import annotations

import numpy as np


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute ∂fn(x)/∂x via central differences. fn must return a scalar."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = float(fn(x))
        x[idx] = orig - eps
        minus = float(fn(x))
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def assert_grad_match(analytic: np.ndarray, numerical: np.ndarray, atol: float = 1e-5):
    """Assert analytical gradient matches numerical."""
    assert analytic.shape == numerical.shape, (
        f"shape mismatch: analytic={analytic.shape}, numerical={numerical.shape}"
    )
    diff = np.max(np.abs(analytic - numerical))
    assert diff < atol, f"max abs diff {diff:.2e} > {atol:.2e}"
