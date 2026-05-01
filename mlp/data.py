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
