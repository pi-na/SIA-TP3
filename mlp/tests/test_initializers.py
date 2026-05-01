from __future__ import annotations

import numpy as np
import pytest

from mlp.initializers import init_uniform, init_he, init_xavier, auto_pick


def test_uniform_in_range(rng):
    w = init_uniform((10, 20), rng, scale=0.1)
    assert w.shape == (10, 20)
    assert (w >= -0.1).all() and (w <= 0.1).all()


def test_he_std_correct(rng):
    """He: std ≈ sqrt(2/fan_in)."""
    w = init_he((50, 100 + 1), rng)  # fan_in = 100 (sin contar bias)
    assert w.shape == (50, 101)
    expected_std = np.sqrt(2.0 / 100)
    assert abs(w.std() - expected_std) < expected_std * 0.2  # ±20% tolerancia


def test_xavier_std_correct(rng):
    w = init_xavier((50, 100 + 1), rng)
    expected_std = np.sqrt(1.0 / 100)
    assert abs(w.std() - expected_std) < expected_std * 0.2


@pytest.mark.parametrize("act,expected_init", [
    ("relu", "he"),
    ("tanh", "xavier"),
    ("sigmoid", "xavier"),
    ("identity", "uniform"),
    ("softmax", "xavier"),
])
def test_auto_pick(act, expected_init):
    assert auto_pick(act) == expected_init


def test_reproducibility(rng):
    """Same seed → same weights."""
    rng2 = np.random.default_rng(42)
    rng3 = np.random.default_rng(42)
    a = init_he((10, 20), rng2)
    b = init_he((10, 20), rng3)
    assert np.allclose(a, b)
