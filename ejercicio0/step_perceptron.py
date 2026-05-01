"""Perceptrón simple escalón (signo) para clasificación binaria con bipolar.

Usado para validar AND (TP3, validación). Implementa el algoritmo clásico
de Rosenblatt: actualización online con regla delta.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def step_activation(h: np.ndarray) -> np.ndarray:
    """Función signo. h>=0 → +1, h<0 → -1."""
    return np.where(h >= 0, 1, -1)


def predict(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predice usando weights = [bias, w1, w2, ...]."""
    P = len(X)
    X_bias = np.column_stack([np.ones(P), X])
    h = X_bias @ weights
    return step_activation(h)


def train_step_perceptron(
    df: pd.DataFrame, learning_rate: float = 0.1,
    epochs: int = 100, seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Entrena perceptrón escalón online.

    Devuelve (weights, errors_per_epoch).
    weights[0] = bias, weights[1:] = pesos de features.
    """
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    P, n = X.shape
    X_bias = np.column_stack([np.ones(P), X])

    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.1, 0.1, size=n + 1)

    errors_per_epoch = []
    for _ in range(epochs):
        n_errors = 0
        for mu in range(P):
            h = float(weights @ X_bias[mu])
            o = 1 if h >= 0 else -1
            if o != y[mu]:
                n_errors += 1
                weights = weights + learning_rate * (y[mu] - o) * X_bias[mu]
        errors_per_epoch.append(n_errors)
        if n_errors == 0:
            break
    return weights, errors_per_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ejercicio0/step_dataset.csv", type=Path)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    weights, errors = train_step_perceptron(
        df, learning_rate=args.learning_rate, epochs=args.epochs, seed=args.seed,
    )
    print(f"Pesos finales: bias={weights[0]:.4f}, "
          f"w={weights[1:]}")
    print(f"Errores por época: {errors}")
    print(f"Convergió en {len(errors)} épocas (0 errores)" if errors[-1] == 0
          else f"NO convergió en {args.epochs} épocas")


if __name__ == "__main__":
    main()
