from __future__ import annotations

import numpy as np
import pytest

from mlp.optimizers import SGD, OPTIMIZERS


def quadratic_loss_and_grad(w_list):
    """L = 0.5 * sum(w^2). Optimum at w=0. Grad = w."""
    loss = 0.5 * sum((w ** 2).sum() for w in w_list)
    grads = [w.copy() for w in w_list]
    return loss, grads


def run_optim_steps(opt, w_list, n_steps=200):
    initial_loss, _ = quadratic_loss_and_grad(w_list)
    for _ in range(n_steps):
        _, grads = quadratic_loss_and_grad(w_list)
        opt.step(w_list, grads)
    final_loss, _ = quadratic_loss_and_grad(w_list)
    return initial_loss, final_loss


def test_sgd_reduces_loss(rng):
    w = [rng.uniform(-1, 1, size=(5, 10))]
    opt = SGD(lr=0.1)
    initial, final = run_optim_steps(opt, w, n_steps=200)
    assert final < initial * 0.01, f"SGD didn't converge: {initial} → {final}"
