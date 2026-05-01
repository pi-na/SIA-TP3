from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from step_perceptron import train_step_perceptron, predict


def test_step_converges_on_AND():
    df = pd.read_csv(Path(__file__).parent.parent / "step_dataset.csv")
    weights, errors_per_epoch = train_step_perceptron(
        df, learning_rate=0.1, epochs=100, seed=42,
    )
    # Convergencia: 0 errores al final
    assert errors_per_epoch[-1] == 0
    # Verificación funcional
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    preds = predict(weights, X)
    assert (preds == y).all()


def test_step_separator_geometry():
    """w1·x1 + w2·x2 + bias = 0 debe separar (1,1) del resto."""
    df = pd.read_csv(Path(__file__).parent.parent / "step_dataset.csv")
    weights, _ = train_step_perceptron(df, learning_rate=0.1, epochs=100, seed=42)
    # Punto (1,1) debe quedar en el lado positivo
    h_positive = weights[0] + weights[1] * 1 + weights[2] * 1
    assert h_positive > 0
    # Resto en lado negativo
    for x1, x2 in [(-1, 1), (1, -1), (-1, -1)]:
        h = weights[0] + weights[1] * x1 + weights[2] * x2
        assert h < 0
