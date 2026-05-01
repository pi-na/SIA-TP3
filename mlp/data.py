"""Data utilities: parsing, K-fold, train/val split, mini-batch iter."""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import pandas as pd


def parse_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Convierte feature_cols a np.ndarray (P, n_features).

    Soporta dos formatos:
    - Columnas numéricas: cada columna es una feature.
    - UNA sola columna con strings tipo "[0.1, 0.2, ...]" (formato digits.csv):
      se parsea cada string a array y se stackea.

    Mezclar ambos en feature_cols es ambiguo → ValueError.
    """
    if len(feature_cols) == 1:
        col = feature_cols[0]
        first = df[col].iloc[0]
        if isinstance(first, str) and first.startswith("["):
            arr = np.stack([np.array(json.loads(s), dtype=np.float64) for s in df[col]])
            return arr
    # Validate todas son numéricas
    for c in feature_cols:
        first = df[c].iloc[0]
        if isinstance(first, str) and first.startswith("["):
            raise ValueError(
                f"Column {c!r} contiene stringified arrays pero no es la única feature_col. "
                "No mezcles arrays serializados con features escalares."
            )
    return df[feature_cols].to_numpy(dtype=np.float64)


def stratified_kfold(
    y: np.ndarray, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold estratificado por clase. Devuelve [(train_idx, test_idx), ...]."""
    if k < 2:
        raise ValueError(f"stratified_kfold requiere k>=2, got {k}")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    chunks_per_class = []
    for cls in classes:
        idx = np.where(y == cls)[0].copy()
        if len(idx) < k:
            raise ValueError(f"Clase {cls} tiene {len(idx)} muestras < k={k}.")
        rng.shuffle(idx)
        chunks_per_class.append(np.array_split(idx, k))
    folds = []
    for i in range(k):
        test_idx = np.concatenate([cc[i] for cc in chunks_per_class])
        train_idx = np.concatenate([cc[j] for cc in chunks_per_class for j in range(k) if j != i])
        folds.append((train_idx, test_idx))
    return folds


def train_val_split(
    y: np.ndarray, val_fraction: float, stratify: bool, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single train/val split. Stratified si stratify=True.

    Caso especial: val_fraction == 0.0 ⇒ val_idx = train_idx (eval sobre train).
    Útil para datasets minúsculos como XOR donde no hay holdout posible.
    """
    rng = np.random.default_rng(seed)
    if val_fraction == 0.0:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return idx, idx.copy()
    if not stratify:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_val = int(len(y) * val_fraction)
        return idx[n_val:], idx[:n_val]
    classes = np.unique(y)
    train_chunks, val_chunks = [], []
    for cls in classes:
        idx = np.where(y == cls)[0].copy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_chunks.append(idx[:n_val])
        train_chunks.append(idx[n_val:])
    return np.concatenate(train_chunks), np.concatenate(val_chunks)


class BatchIterator:
    """Itera sobre (X, y) en mini-batches. Reshufflea cada época."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool = True, seed: int | None = None):
        if len(X) != len(y):
            raise ValueError(f"X y y de tamaños distintos: {len(X)} vs {len(y)}")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(len(self.X))
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            chunk = idx[start:start + self.batch_size]
            yield self.X[chunk], self.y[chunk]

    def __len__(self) -> int:
        return (len(self.X) + self.batch_size - 1) // self.batch_size
