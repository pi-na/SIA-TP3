"""Helpers for testing analytical gradients vs numerical."""
from __future__ import annotations

import numpy as np


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute ∂fn(x)/∂x via central differences. fn must return a scalar."""
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(
            f"numerical_grad requires a floating-point array, got {x.dtype}"
        )
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
    diff_arr = np.abs(analytic - numerical)
    worst_idx = np.unravel_index(np.argmax(diff_arr), diff_arr.shape)
    diff = float(diff_arr[worst_idx])
    assert diff < atol, (
        f"max abs diff {diff:.2e} > {atol:.2e} at index {worst_idx} "
        f"(analytic={analytic[worst_idx]:.6e}, numerical={numerical[worst_idx]:.6e})"
    )
