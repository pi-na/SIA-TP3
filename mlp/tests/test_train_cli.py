from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlp.train import load_and_validate_config


def write_config(path: Path, **overrides) -> Path:
    base = {
        "model_name": "test",
        "dataset": {
            "csv_path": "x.csv", "feature_col": "image",
            "target_col": "label", "num_classes": 10, "extra_csv_paths": [],
        },
        "split": {"k_folds": 5, "stratify": True,
                  "val_fraction_if_k1": 0.2, "random_seed": 42},
        "preprocessing": {"normalization": "none", "one_hot_targets": True},
        "architecture": {"layer_sizes": [784, 100, 10],
                         "activations": ["relu", "softmax"], "initializer": "auto"},
        "training": {
            "loss": "cross_entropy",
            "optimizer": {"name": "adam", "lr": 0.001},
            "epochs": 50, "batch_size": 64, "early_stopping_patience": 10,
        },
        "regularization": {"l2": 0.0, "dropout": 0.0,
                           "lr_schedule": None, "augmentation": None},
    }
    for k, v in overrides.items():
        keys = k.split(".")
        d = base
        for kk in keys[:-1]:
            d = d[kk]
        d[keys[-1]] = v
    path.write_text(json.dumps(base, indent=2))
    return path


def test_valid_config_loads(tmp_path):
    p = write_config(tmp_path / "c.json")
    cfg = load_and_validate_config(p)
    assert cfg["model_name"] == "test"


def test_activations_length_mismatch_rejected(tmp_path):
    p = write_config(tmp_path / "c.json", **{"architecture.activations": ["relu"]})
    with pytest.raises(ValueError, match="activations"):
        load_and_validate_config(p)


def test_cross_entropy_requires_softmax_output(tmp_path):
    p = write_config(tmp_path / "c.json", **{"architecture.activations": ["relu", "sigmoid"]})
    with pytest.raises(ValueError, match="softmax"):
        load_and_validate_config(p)


def test_unknown_optimizer_rejected(tmp_path):
    p = write_config(tmp_path / "c.json", **{"training.optimizer": {"name": "xyz", "lr": 0.1}})
    with pytest.raises(ValueError, match="optimizer"):
        load_and_validate_config(p)


def test_run_fold_returns_summary_and_history(tmp_path):
    """Smoke test: corre 1 fold sobre datos sintéticos chicos."""
    from mlp.train import run_fold
    rng = np.random.default_rng(0)
    n = 200
    X = rng.uniform(-1, 1, size=(n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    cfg = json.loads(write_config(tmp_path / "c.json", **{
        "architecture.layer_sizes": [5, 8, 2],
        "architecture.activations": ["relu", "softmax"],
        "training.epochs": 10,
        "training.batch_size": 32,
        "training.early_stopping_patience": None,
        "dataset.num_classes": 2,
    }).read_text())
    train_idx = np.arange(160)
    val_idx = np.arange(160, 200)
    summary, history, weights = run_fold(
        cfg, X, y, train_idx, val_idx, fold_idx=0, fold_seed=42,
    )
    assert "val_acc_final" in summary
    assert "macro_f1" in summary
    assert len(history) <= 10
    assert isinstance(weights, list)
