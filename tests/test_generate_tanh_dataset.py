import numpy as np
from generate_tanh_dataset import generate_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_points=50, seed=42)
    assert df.shape == (50, 2)
    assert list(df.columns) == ["x", "y"]


def test_generate_dataset_x_range():
    df = generate_dataset(n_points=50, seed=42)
    assert df["x"].min() >= -5.0
    assert df["x"].max() <= 5.0


def test_generate_dataset_y_equals_tanh_x():
    """Dataset is noiseless: y must equal tanh(x) exactly."""
    df = generate_dataset(n_points=50, seed=42)
    np.testing.assert_allclose(df["y"].values, np.tanh(df["x"].values), atol=1e-12)


def test_generate_dataset_seed_reproducibility():
    df1 = generate_dataset(n_points=50, seed=123)
    df2 = generate_dataset(n_points=50, seed=123)
    np.testing.assert_array_equal(df1["x"].values, df2["x"].values)
