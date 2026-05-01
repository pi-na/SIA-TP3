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


class Momentum(Optimizer):
    """SGD with momentum: v = β·v - lr·g; w += v."""

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.velocity: list[np.ndarray] | None = None

    def step(self, weights, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(w) for w in weights]
        for i, (w, g) in enumerate(zip(weights, grads)):
            self.velocity[i] = self.beta * self.velocity[i] - self.lr * g
            weights[i] = w + self.velocity[i]


# Adam se agrega en task 7
OPTIMIZERS = {"sgd": SGD, "momentum": Momentum}


def build_optimizer(name: str, **kwargs) -> Optimizer:
    """Factory: convierte config dict en instancia."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer desconocido: {name!r}. Disponibles: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](**kwargs)
