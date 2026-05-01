from __future__ import annotations

import numpy as np
import pytest

from mlp.optimizers import SGD, Momentum, Adam, OPTIMIZERS, build_optimizer


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


def test_momentum_converges_faster_than_sgd(rng):
    """Momentum should reach lower loss than vanilla SGD in same steps."""
    seed_state = rng.bit_generator.state
    w_sgd = [rng.uniform(-1, 1, size=(5, 10))]
    rng.bit_generator.state = seed_state
    w_mom = [rng.uniform(-1, 1, size=(5, 10))]

    _, sgd_final = run_optim_steps(SGD(lr=0.05), w_sgd, n_steps=50)
    _, mom_final = run_optim_steps(Momentum(lr=0.05, beta=0.9), w_mom, n_steps=50)
    assert mom_final < sgd_final


def test_momentum_lazy_init():
    """Velocity is None until first step()."""
    opt = Momentum(lr=0.1, beta=0.9)
    assert opt.velocity is None
    w = [np.ones((3, 4))]
    g = [np.ones((3, 4)) * 0.1]
    opt.step(w, g)
    assert opt.velocity is not None
    assert len(opt.velocity) == 1


def test_adam_converges(rng):
    w = [rng.uniform(-1, 1, size=(5, 10))]
    opt = Adam(lr=0.01)
    initial, final = run_optim_steps(opt, w, n_steps=300)
    assert final < initial * 0.05


def test_adam_state_initialized():
    opt = Adam(lr=0.001)
    assert opt.m is None and opt.v is None and opt.t == 0
    w = [np.ones((3, 4))]
    g = [np.ones((3, 4)) * 0.1]
    opt.step(w, g)
    assert opt.m is not None and opt.v is not None
    assert opt.t == 1


def test_build_optimizer():
    sgd = build_optimizer("sgd", lr=0.01)
    assert isinstance(sgd, SGD)
    adam = build_optimizer("adam", lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    assert isinstance(adam, Adam)
    with pytest.raises(ValueError):
        build_optimizer("xyz", lr=0.1)
