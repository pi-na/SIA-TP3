from __future__ import annotations

import numpy as np
import pytest

from mlp.activations import (
    sigmoid, sigmoid_grad,
    tanh, tanh_grad,
    relu, relu_grad,
    identity, identity_grad,
    softmax,
    ACTIVATIONS,
)
from mlp.tests._helpers import numerical_grad, assert_grad_match


@pytest.mark.parametrize("name,fn,grad_fn", [
    ("sigmoid", sigmoid, sigmoid_grad),
    ("tanh", tanh, tanh_grad),
    ("relu", relu, relu_grad),
    ("identity", identity, identity_grad),
])
def test_activation_grad_matches_numerical(name, fn, grad_fn, rng):
    z = rng.uniform(-2, 2, size=(3, 4))
    a = fn(z)
    analytic = grad_fn(z, a)
    # build scalar fn for numerical_grad
    def scalar(x):
        return fn(x).sum()
    numerical = numerical_grad(scalar, z.copy())
    assert_grad_match(analytic, numerical, atol=1e-4)


def test_softmax_sums_to_one(rng):
    z = rng.uniform(-5, 5, size=(8, 10))
    out = softmax(z)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all() and (out <= 1).all()


def test_softmax_numerical_stability():
    # Without stability trick, exp(1000) overflows
    z = np.array([[1000.0, 1001.0, 999.0]])
    out = softmax(z)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.isclose(out.sum(), 1.0)


def test_activations_dict_complete():
    assert set(ACTIVATIONS.keys()) == {"sigmoid", "tanh", "relu", "identity", "softmax"}
    for name, (fn, grad_fn) in ACTIVATIONS.items():
        if name != "softmax":
            assert callable(fn) and callable(grad_fn)
    assert ACTIVATIONS["softmax"][1] is None, "softmax grad_fn must be None"
