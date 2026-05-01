import numpy as np
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def xor_data():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=np.float64)
    y = np.array([1, 1, -1, -1], dtype=np.float64).reshape(-1, 1)
    return X, y
