"""CLI: config.json → output dir con CSVs + weights.npz."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlp.activations import ACTIVATIONS
from mlp.data import (
    parse_features, stratified_kfold, train_val_split, BatchIterator,
)
from mlp.losses import LOSSES
from mlp.metrics import multiclass_metrics
from mlp.network import MLP
from mlp.optimizers import build_optimizer


REQUIRED_TOP = {"model_name", "dataset", "split", "preprocessing",
                "architecture", "training", "regularization"}


def load_and_validate_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    missing = REQUIRED_TOP - set(cfg)
    if missing:
        raise ValueError(f"Config inválido: faltan campos {sorted(missing)}")

    arch = cfg["architecture"]
    if len(arch["activations"]) != len(arch["layer_sizes"]) - 1:
        raise ValueError(
            f"activations debe tener {len(arch['layer_sizes'])-1} elementos "
            f"(uno por transición), got {len(arch['activations'])}"
        )

    loss = cfg["training"]["loss"]
    if loss not in LOSSES:
        raise ValueError(f"loss desconocido: {loss!r}")
    if loss == "cross_entropy" and arch["activations"][-1] != "softmax":
        raise ValueError("loss='cross_entropy' requiere última activación='softmax'")
    if loss == "bce" and arch["activations"][-1] != "sigmoid":
        raise ValueError("loss='bce' requiere última activación='sigmoid'")

    opt_name = cfg["training"]["optimizer"]["name"]
    if opt_name not in {"sgd", "momentum", "adam"}:
        raise ValueError(f"optimizer desconocido: {opt_name!r}")

    if cfg["split"]["k_folds"] < 1:
        raise ValueError("k_folds debe ser >= 1")
    return cfg


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y.astype(int)]


def run_fold(
    cfg: dict, X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
    fold_idx: int, fold_seed: int,
) -> tuple[dict, list[dict], list[np.ndarray]]:
    """Entrena un fold. Devuelve (summary, history, weights)."""
    arch = cfg["architecture"]
    train_cfg = cfg["training"]
    num_classes = cfg["dataset"]["num_classes"]
    one_hot = cfg["preprocessing"]["one_hot_targets"]

    X_train, y_train_raw = X[train_idx], y[train_idx]
    X_val, y_val_raw = X[val_idx], y[val_idx]

    # Normalización opcional (z-score fit-on-train)
    norm = cfg["preprocessing"]["normalization"]
    if norm == "zscore":
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        stds[stds == 0] = 1.0
        X_train = (X_train - means) / stds
        X_val = (X_val - means) / stds
    elif norm == "minmax":
        mins = X_train.min(axis=0)
        rng_ = X_train.max(axis=0) - mins
        rng_[rng_ == 0] = 1.0
        X_train = (X_train - mins) / rng_
        X_val = (X_val - mins) / rng_
    elif norm != "none":
        raise ValueError(f"normalization desconocida: {norm!r}")

    # Targets one-hot si aplica
    y_train = _one_hot(y_train_raw, num_classes) if one_hot else y_train_raw.reshape(-1, 1).astype(float)
    y_val = _one_hot(y_val_raw, num_classes) if one_hot else y_val_raw.reshape(-1, 1).astype(float)

    # Build optimizer + MLP
    opt_cfg = dict(train_cfg["optimizer"])
    opt_name = opt_cfg.pop("name")
    optimizer = build_optimizer(opt_name, **opt_cfg)
    mlp = MLP(
        layer_sizes=arch["layer_sizes"],
        activations=arch["activations"],
        loss=train_cfg["loss"],
        optimizer=optimizer,
        initializer=arch["initializer"],
        seed=fold_seed,
        regularization=cfg["regularization"],
    )

    # Entrenar
    t0 = time.time()
    history_compact = []

    def on_epoch(epoch, m):
        # Evaluar también accuracy en cada época (para epoch_history.csv)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_acc = float((train_pred == y_train_raw).mean())
        val_acc = float((val_pred == y_val_raw).mean())
        history_compact.append({
            "epoch": epoch,
            "time_elapsed_s": time.time() - t0,
            "train_loss": m["train_loss"],
            "val_loss": m["val_loss"],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr_actual": opt_cfg.get("lr", 0.0),
        })

    mlp.fit(
        X_train, y_train, X_val, y_val,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        callback=on_epoch,
    )
    elapsed = time.time() - t0

    # Métricas finales sobre val
    val_pred = mlp.predict(X_val)
    final = multiclass_metrics(y_val_raw, val_pred, num_classes)
    train_pred = mlp.predict(X_train)
    train_acc_final = float((train_pred == y_train_raw).mean())

    summary = {
        "fold": fold_idx,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "total_epochs": len(history_compact),
        "best_epoch": int(np.argmin([h["val_loss"] for h in history_compact])),
        "train_loss_final": history_compact[-1]["train_loss"],
        "val_loss_final": history_compact[-1]["val_loss"],
        "train_acc_final": train_acc_final,
        "val_acc_final": final["accuracy"],
        "macro_precision": final["macro_precision"],
        "macro_recall": final["macro_recall"],
        "macro_f1": final["macro_f1"],
        "weighted_f1": final["weighted_f1"],
        "time_seconds": elapsed,
    }
    for c in range(num_classes):
        summary[f"precision_{c}"] = float(final["precision"][c])
        summary[f"recall_{c}"] = float(final["recall"][c])
        summary[f"f1_{c}"] = float(final["f1"][c])

    return summary, history_compact, mlp.weights
