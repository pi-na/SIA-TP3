from __future__ import annotations

import numpy as np
import pytest

from mlp.network import MLP
from mlp.optimizers import SGD, Adam


def test_init_validates_activations_length():
    """activations debe tener len(layer_sizes) - 1."""
    with pytest.raises(ValueError, match="activations"):
        MLP(layer_sizes=[2, 3, 1], activations=["relu"], loss="mse",
            optimizer=SGD(0.01))


def test_init_validates_cross_entropy_requires_softmax():
    with pytest.raises(ValueError, match="softmax"):
        MLP(layer_sizes=[2, 3, 4], activations=["relu", "sigmoid"],
            loss="cross_entropy", optimizer=Adam(0.001))


def test_init_weight_shapes():
    """weights[l] shape (n_l, n_{l-1}+1) por bias trick."""
    mlp = MLP(layer_sizes=[3, 5, 2], activations=["relu", "softmax"],
              loss="cross_entropy", optimizer=Adam(0.001), seed=42)
    assert len(mlp.weights) == 2
    assert mlp.weights[0].shape == (5, 4)  # 3 inputs + 1 bias
    assert mlp.weights[1].shape == (2, 6)  # 5 hidden + 1 bias


def test_init_auto_picks_initializer():
    """initializer='auto' usa He para ReLU, Xavier para tanh/softmax/sigmoid."""
    mlp = MLP(layer_sizes=[100, 50, 10], activations=["relu", "softmax"],
              loss="cross_entropy", optimizer=Adam(0.001),
              initializer="auto", seed=42)
    # He init: std ≈ sqrt(2/100) = 0.141 para capa 0
    layer0_std = mlp.weights[0][:, 1:].std()  # excluye bias column
    assert 0.1 < layer0_std < 0.2


def test_init_seed_reproducibility():
    a = MLP([2, 3, 1], ["tanh", "tanh"], "mse", SGD(0.01), seed=42)
    b = MLP([2, 3, 1], ["tanh", "tanh"], "mse", SGD(0.01), seed=42)
    for wa, wb in zip(a.weights, b.weights):
        assert np.allclose(wa, wb)


def test_forward_output_shape():
    mlp = MLP([3, 5, 2], ["relu", "softmax"], "cross_entropy",
              Adam(0.001), seed=42)
    X = np.random.default_rng(0).uniform(0, 1, size=(8, 3))
    out, cache = mlp.forward(X)
    assert out.shape == (8, 2)
    # softmax sums to 1
    assert np.allclose(out.sum(axis=1), 1.0)
    # cache: una tupla (z, a) por capa
    assert len(cache) == 2


def test_forward_cache_pre_and_post_activation():
    mlp = MLP([2, 3, 1], ["tanh", "sigmoid"], "bce", SGD(0.01), seed=42)
    X = np.array([[0.5, -0.5]])
    out, cache = mlp.forward(X)
    z0, a0, _ = cache[0]
    z1, a1, _ = cache[1]
    # a0 = tanh(z0), a1 = sigmoid(z1) = out
    assert np.allclose(a0, np.tanh(z0))
    assert np.allclose(a1, out)


def test_backward_matches_numerical_grad():
    """Gold standard test: analítico vs diferenciación numérica."""
    from mlp.tests._helpers import numerical_grad
    rng = np.random.default_rng(42)
    mlp = MLP([3, 4, 2], ["tanh", "softmax"], "cross_entropy",
              SGD(0.01), seed=42)
    X = rng.uniform(-1, 1, size=(5, 3))
    y_true = np.eye(2)[rng.integers(0, 2, size=5)]

    # Analítico
    pred, cache = mlp.forward(X)
    grads_analytic = mlp.backward(X, y_true, cache)

    # Numérico (sobre cada capa)
    for layer_idx, W_orig in enumerate(mlp.weights):
        def loss_fn(W_test):
            saved = mlp.weights[layer_idx].copy()
            mlp.weights[layer_idx] = W_test
            p, _ = mlp.forward(X)
            from mlp.losses import cross_entropy
            l = cross_entropy(y_true, p)
            mlp.weights[layer_idx] = saved
            return l
        numerical = numerical_grad(loss_fn, W_orig.copy(), eps=1e-5)
        assert np.max(np.abs(grads_analytic[layer_idx] - numerical)) < 1e-4, (
            f"layer {layer_idx}: grad check failed"
        )


def test_backward_grad_check_bce_sigmoid():
    """Gradient check on BCE+sigmoid path."""
    from mlp.tests._helpers import numerical_grad
    rng = np.random.default_rng(7)
    mlp = MLP([3, 4, 1], ["tanh", "sigmoid"], "bce",
              SGD(0.01), seed=7)
    X = rng.uniform(-1, 1, size=(5, 3))
    y_true = (rng.uniform(0, 1, size=(5, 1)) > 0.5).astype(float)

    pred, cache = mlp.forward(X)
    grads_analytic = mlp.backward(X, y_true, cache)

    for layer_idx, W_orig in enumerate(mlp.weights):
        def loss_fn(W_test):
            saved = mlp.weights[layer_idx].copy()
            mlp.weights[layer_idx] = W_test
            p, _ = mlp.forward(X)
            from mlp.losses import bce
            l = bce(y_true, p)
            mlp.weights[layer_idx] = saved
            return l
        numerical = numerical_grad(loss_fn, W_orig.copy(), eps=1e-5)
        assert np.max(np.abs(grads_analytic[layer_idx] - numerical)) < 1e-4, (
            f"BCE+sigmoid layer {layer_idx}: grad check failed"
        )


def test_backward_grad_check_mse_identity():
    """Gradient check on MSE+identity path (regression)."""
    from mlp.tests._helpers import numerical_grad
    rng = np.random.default_rng(13)
    mlp = MLP([3, 4, 2], ["tanh", "identity"], "mse",
              SGD(0.01), seed=13)
    X = rng.uniform(-1, 1, size=(5, 3))
    y_true = rng.uniform(-1, 1, size=(5, 2))

    pred, cache = mlp.forward(X)
    grads_analytic = mlp.backward(X, y_true, cache)

    for layer_idx, W_orig in enumerate(mlp.weights):
        def loss_fn(W_test):
            saved = mlp.weights[layer_idx].copy()
            mlp.weights[layer_idx] = W_test
            p, _ = mlp.forward(X)
            from mlp.losses import mse
            l = mse(y_true, p)
            mlp.weights[layer_idx] = saved
            return l
        numerical = numerical_grad(loss_fn, W_orig.copy(), eps=1e-5)
        assert np.max(np.abs(grads_analytic[layer_idx] - numerical)) < 1e-4, (
            f"MSE+identity layer {layer_idx}: grad check failed"
        )
