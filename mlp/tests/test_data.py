from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from mlp.data import parse_features


def test_parse_features_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6], "label": [0, 1, 0]})
    X = parse_features(df, feature_cols=["a", "b"])
    assert X.shape == (3, 2)
    assert X.dtype == np.float64


def test_parse_features_stringified_array_column():
    """Single column with stringified array (digits.csv format)."""
    df = pd.DataFrame({
        "image": ["[0.1, 0.2, 0.3]", "[0.4, 0.5, 0.6]"],
        "label": [0, 1],
    })
    X = parse_features(df, feature_cols=["image"])
    assert X.shape == (2, 3)
    np.testing.assert_allclose(X[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(X[1], [0.4, 0.5, 0.6])


def test_parse_features_mixed_raises():
    """Mixing stringified arrays with other features is ambiguous → error."""
    df = pd.DataFrame({"image": ["[0.1, 0.2]"], "amount": [42.0]})
    with pytest.raises(ValueError, match="stringified"):
        parse_features(df, feature_cols=["image", "amount"])
