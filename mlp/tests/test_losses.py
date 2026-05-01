from __future__ import annotations

import numpy as np
import pytest

from mlp.losses import (
    mse, mse_grad,
    bce, bce_grad,
    cross_entropy, cross_entropy_grad_with_softmax,
    LOSSES,
)
from mlp.tests._helpers import numerical_grad, assert_grad_match


def test_mse_grad(rng):
    y_true = rng.uniform(0, 1, size=(5, 3))
    y_pred = rng.uniform(0, 1, size=(5, 3))
    analytic = mse_grad(y_true, y_pred)
    def scalar(p):
        return mse(y_true, p)
    numerical = numerical_grad(scalar, y_pred.copy())
    assert_grad_match(analytic, numerical, atol=1e-4)


def test_bce_grad(rng):
    y_true = (rng.uniform(0, 1, size=(5, 1)) > 0.5).astype(float)
    y_pred = rng.uniform(0.05, 0.95, size=(5, 1))  # avoid log(0)
    analytic = bce_grad(y_true, y_pred)
    def scalar(p):
        return bce(y_true, p)
    numerical = numerical_grad(scalar, y_pred.copy())
    assert_grad_match(analytic, numerical, atol=1e-4)


def test_cross_entropy_with_softmax_grad(rng):
    """Gradient ∂CE/∂z_pre_softmax = softmax(z) - y_true."""
    from mlp.activations import softmax
    z = rng.uniform(-2, 2, size=(4, 5))
    y_true = np.eye(5)[rng.integers(0, 5, size=4)]  # one-hot
    p = softmax(z)
    analytic = cross_entropy_grad_with_softmax(y_true, p)
    def scalar(zz):
        return cross_entropy(y_true, softmax(zz))
    numerical = numerical_grad(scalar, z.copy())
    assert_grad_match(analytic, numerical, atol=1e-4)


def test_losses_registered():
    assert set(LOSSES.keys()) == {"mse", "bce", "cross_entropy"}
