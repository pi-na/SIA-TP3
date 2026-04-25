import os
import tempfile
import pandas as pd
import numpy as np
from generate_linear_dataset import generate_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_points=50, seed=42)
    assert df.shape == (50, 2)
    assert list(df.columns) == ["x", "y"]


def test_generate_dataset_x_range():
    df = generate_dataset(n_points=50, seed=42)
    assert df["x"].min() >= -10.0
    assert df["x"].max() <= 10.0


def test_generate_dataset_y_follows_linear_trend():
    """With known seed, y should be close to 3x+2."""
    df = generate_dataset(n_points=1000, seed=42)
    coeffs = np.polyfit(df["x"], df["y"], 1)
    assert abs(coeffs[0] - 3.0) < 0.2
    assert abs(coeffs[1] - 2.0) < 0.2
