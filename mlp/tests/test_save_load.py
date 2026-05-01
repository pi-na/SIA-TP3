from __future__ import annotations

from pathlib import Path

import numpy as np

from mlp.network import MLP
from mlp.optimizers import SGD


def test_save_load_roundtrip(tmp_path: Path):
    mlp = MLP([3, 4, 2], ["tanh", "softmax"], "cross_entropy",
              SGD(0.01), seed=42)
    X = np.random.default_rng(0).uniform(-1, 1, size=(5, 3))
    pred_before = mlp.predict_proba(X)

    p = tmp_path / "model.npz"
    mlp.save(p)
    loaded = MLP.load(p)
    pred_after = loaded.predict_proba(X)

    np.testing.assert_allclose(pred_before, pred_after)
    assert loaded.layer_sizes == mlp.layer_sizes
    assert loaded.activations == mlp.activations


def test_predict_returns_class_indices():
    mlp = MLP([3, 4, 5], ["tanh", "softmax"], "cross_entropy",
              SGD(0.01), seed=42)
    X = np.random.default_rng(0).uniform(-1, 1, size=(8, 3))
    pred = mlp.predict(X)
    assert pred.shape == (8,)
    assert pred.dtype.kind == "i"
    assert (pred >= 0).all() and (pred < 5).all()
