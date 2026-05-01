from __future__ import annotations

import json
from pathlib import Path

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
