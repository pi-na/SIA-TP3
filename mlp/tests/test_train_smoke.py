"""End-to-end smoke test del CLI con datos sintéticos en CSV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlp.train import load_and_validate_config, run_experiment


def test_e2e_synthetic_csv(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(-1, 1, size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["label"] = y

    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "model_name": "smoke",
        "dataset": {
            "csv_path": "data.csv", "feature_col": ["f0", "f1", "f2", "f3"],
            "target_col": "label", "num_classes": 2, "extra_csv_paths": [],
        },
        "split": {"k_folds": 2, "stratify": True,
                  "val_fraction_if_k1": 0.2, "random_seed": 42},
        "preprocessing": {"normalization": "zscore", "one_hot_targets": True},
        "architecture": {
            "layer_sizes": [4, 8, 2],
            "activations": ["relu", "softmax"], "initializer": "auto",
        },
        "training": {
            "loss": "cross_entropy",
            "optimizer": {"name": "adam", "lr": 0.01},
            "epochs": 30, "batch_size": 32, "early_stopping_patience": None,
        },
        "regularization": {"l2": 0.0, "dropout": 0.0,
                           "lr_schedule": None, "augmentation": None},
    }
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    out_dir = tmp_path / "out"
    cfg_loaded = load_and_validate_config(cfg_path)
    run_dir = run_experiment(cfg_loaded, csv_root=tmp_path, output_dir=out_dir, workers=2)

    # CSVs deben existir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run_summary.csv").exists()
    assert (run_dir / "epoch_history.csv").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "confusion_matrix.csv").exists()
    assert (run_dir / "weights.npz").exists()

    # Sanity check on metrics
    summary = pd.read_csv(run_dir / "run_summary.csv")
    val_acc = float(summary[summary["fold"] == "mean"]["val_acc_final"].iloc[0]) \
              if "mean" in summary["fold"].astype(str).values \
              else float(summary.iloc[0]["val_acc_final"])
    assert val_acc > 0.7, f"toy problem should hit >70% accuracy, got {val_acc}"

    # Bug fix verification: mean/std rows have correct string labels
    fold_values = summary["fold"].astype(str).tolist()
    assert "mean" in fold_values, f"mean row label missing: {fold_values}"
    assert "std" in fold_values, f"std row label missing: {fold_values}"
