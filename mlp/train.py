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
