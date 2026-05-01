"""Optimizadores: SGD, Momentum, Adam. Cada uno mantiene su state interno."""
from __future__ import annotations

import numpy as np


class Optimizer:
    """Interface base. Subclases mantienen state interno."""

    def step(self, weights: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Actualiza weights in-place usando grads."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, weights, grads):
        for i, (w, g) in enumerate(zip(weights, grads)):
            weights[i] = w - self.lr * g


# Momentum y Adam se agregan en tasks 6 y 7
OPTIMIZERS = {"sgd": SGD}


def build_optimizer(name: str, **kwargs) -> Optimizer:
    """Factory: convierte config dict en instancia."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer desconocido: {name!r}. Disponibles: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](**kwargs)
