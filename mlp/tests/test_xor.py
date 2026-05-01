"""Validación: el MLP genérico debe resolver XOR.

Validamos las dos arquitecturas que el TP recomienda calcular a mano:
[2, 2, 1] (mínima que resuelve XOR) y [2, 3, 2, 1].

XOR con [2,2,1] es notoriamente sensible a inicialización; aceptamos si al menos
2 de 5 seeds convergen.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlp.network import MLP
from mlp.optimizers import Adam


XOR_X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=np.float64)
XOR_Y = np.array([1, 1, -1, -1], dtype=np.float64).reshape(-1, 1)


@pytest.mark.parametrize("layer_sizes,activations", [
    ([2, 2, 1], ["tanh", "tanh"]),
    ([2, 3, 2, 1], ["tanh", "tanh", "tanh"]),
])
def test_xor_converges(layer_sizes, activations):
    convergences = 0
    for seed in [0, 1, 2, 3, 4]:
        mlp = MLP(layer_sizes, activations, loss="mse",
                  optimizer=Adam(lr=0.05), seed=seed)
        history = mlp.fit(XOR_X, XOR_Y, XOR_X, XOR_Y,
                          epochs=2000, batch_size=4)
        if history[-1]["train_loss"] < 0.05:
            convergences += 1
    # [2,2,1] es sensible — pedimos al menos 2/5 seeds
    assert convergences >= 2, f"Sólo {convergences}/5 seeds convergieron"
