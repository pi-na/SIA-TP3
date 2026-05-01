"""Inicializadores de pesos: uniform, He, Xavier."""
from __future__ import annotations

import numpy as np


def init_uniform(shape: tuple[int, int], rng: np.random.Generator,
                  scale: float = 0.1) -> np.ndarray:
    """Uniforme [-scale, scale]. shape: (n_out, n_in + 1) — +1 por bias."""
    return rng.uniform(-scale, scale, size=shape)


def init_he(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """He: N(0, 2/fan_in). fan_in = shape[1] - 1 (excluye bias)."""
    fan_in = shape[1] - 1
    return rng.normal(0, np.sqrt(2.0 / fan_in), size=shape)


def init_xavier(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Xavier: N(0, 1/fan_in)."""
    fan_in = shape[1] - 1
    return rng.normal(0, np.sqrt(1.0 / fan_in), size=shape)


def auto_pick(activation_name: str) -> str:
    """Init recomendado según activación de la capa."""
    return {
        "relu": "he",
        "tanh": "xavier",
        "sigmoid": "xavier",
        "softmax": "xavier",
        "identity": "uniform",
    }[activation_name]


INITIALIZERS = {
    "uniform": init_uniform,
    "he": init_he,
    "xavier": init_xavier,
}
