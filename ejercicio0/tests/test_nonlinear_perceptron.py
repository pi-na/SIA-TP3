import os

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


def test_run_and_save_creates_output(tmp_path):
    """Full pipeline: train + save should create weights.csv and plot.png."""
    from nonlinear_perceptron import run_and_save

    x = np.linspace(-5, 5, 50)
    y = np.tanh(x)
    csv_path = tmp_path / "test_tanh.csv"
    pd.DataFrame({"x": x, "y": y}).to_csv(csv_path, index=False)

    output_dir = tmp_path / "output_test"
    run_and_save(
        csv_path=str(csv_path),
        learning_rate=0.01,
        epochs=200,
        epsilon=1e-6,
        beta=1.0,
        output_dir=str(output_dir),
    )

    assert os.path.isfile(output_dir / "weights.csv")
    assert os.path.isfile(output_dir / "plot.png")

    wdf = pd.read_csv(output_dir / "weights.csv")
    for col in ["w0", "w1", "beta", "mse", "z_min", "z_max"]:
        assert col in wdf.columns, f"missing column {col}"
