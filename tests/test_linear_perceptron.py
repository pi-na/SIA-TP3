# tests/test_linear_perceptron.py
import numpy as np
import pandas as pd
from linear_perceptron import train_perceptron


def test_perceptron_learns_exact_line():
    """Given y=3x+2 with no noise, perceptron should learn w0~2, w1~3."""
    x = np.linspace(-5, 5, 30)
    y = 3 * x + 2
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history = train_perceptron(df, learning_rate=0.001, epochs=500, epsilon=1e-6)

    # weights[0] = bias (w0 ~ 2), weights[1] = slope (w1 ~ 3)
    assert abs(weights[0] - 2.0) < 0.5, f"bias w0={weights[0]}, expected ~2"
    assert abs(weights[1] - 3.0) < 0.5, f"slope w1={weights[1]}, expected ~3"


def test_perceptron_mse_decreases():
    """MSE should generally decrease over training."""
    x = np.linspace(-5, 5, 30)
    y = 3 * x + 2
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history = train_perceptron(df, learning_rate=0.001, epochs=100, epsilon=1e-10)

    assert mse_history[-1] < mse_history[0], "MSE should decrease"


def test_perceptron_early_stop():
    """With exact data and enough epochs, should converge before max epochs."""
    x = np.linspace(-1, 1, 20)  # small range for fast convergence
    y = 2 * x + 1
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history = train_perceptron(df, learning_rate=0.01, epochs=5000, epsilon=1e-4)

    assert len(mse_history) < 5000, f"Should converge early, got {len(mse_history)} epochs"
