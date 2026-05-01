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


class Adam(Optimizer):
    """Adam: adaptive moments. Standard implementation with bias correction."""

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: list[np.ndarray] | None = None
        self.v: list[np.ndarray] | None = None
        self.t: int = 0

    def step(self, weights, grads):
        if self.m is None:
            self.m = [np.zeros_like(w) for w in weights]
            self.v = [np.zeros_like(w) for w in weights]
        self.t += 1
        for i, (w, g) in enumerate(zip(weights, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            weights[i] = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}


def build_optimizer(name: str, **kwargs) -> Optimizer:
    """Factory: convierte config dict en instancia."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer desconocido: {name!r}. Disponibles: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](**kwargs)
