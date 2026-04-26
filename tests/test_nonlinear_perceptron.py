import numpy as np
import pandas as pd
from nonlinear_perceptron import train_perceptron


def test_perceptron_learns_tanh_normalized_target():
    """
    For a linearly separable target (z=ax), the non-linear perceptron with
    tanh activation should drive normalized MSE to a small value.
    Use y = x to make sure the model can fit it after target normalization.
    """
    x = np.linspace(-2, 2, 40)
    y = x  # linear target — easy for the model
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate=0.05, epochs=5000, epsilon=1e-6, beta=1.0
    )

    assert mse_history[-1] < 1e-2, f"final MSE too high: {mse_history[-1]}"


def test_perceptron_mse_decreases():
    """MSE should drop substantially over training."""
    x = np.linspace(-5, 5, 50)
    y = np.tanh(x)
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate=0.01, epochs=200, epsilon=1e-10, beta=1.0
    )

    assert mse_history[-1] < mse_history[0], "MSE should decrease"


def test_normalization_bounds_returned():
    """train_perceptron must return z_min, z_max so callers can denormalize."""
    x = np.linspace(-5, 5, 50)
    y = np.tanh(x)
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate=0.01, epochs=10, epsilon=1e-10, beta=1.0
    )

    assert z_min == y.min()
    assert z_max == y.max()
