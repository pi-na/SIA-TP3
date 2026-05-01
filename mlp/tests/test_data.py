from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from mlp.data import parse_features, stratified_kfold, train_val_split, BatchIterator


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


def test_stratified_kfold_preserves_class_proportion():
    y = np.array([0]*80 + [1]*15 + [2]*5)
    folds = stratified_kfold(y, k=5, seed=42)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        # Proporciones aprox preservadas
        test_y = y[test_idx]
        for cls, total in [(0, 80), (1, 15), (2, 5)]:
            cls_count = (test_y == cls).sum()
            expected = total / 5
            assert abs(cls_count - expected) <= 1, f"clase {cls}: {cls_count} vs {expected}"


def test_stratified_kfold_disjoint_test_folds():
    y = np.array([0]*50 + [1]*50)
    folds = stratified_kfold(y, k=5, seed=42)
    all_test = np.concatenate([t for _, t in folds])
    assert len(all_test) == len(y)
    assert len(np.unique(all_test)) == len(y)


def test_train_val_split_ratio():
    y = np.array([0]*80 + [1]*20)
    train_idx, val_idx = train_val_split(y, val_fraction=0.2, stratify=True, seed=42)
    assert abs(len(val_idx) - 20) <= 1
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    val_y = y[val_idx]
    assert (val_y == 1).sum() in {3, 4, 5}  # ~20% of 20 positives


def test_batch_iterator_yields_full_dataset():
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.arange(100).reshape(-1, 1).astype(float)
    it = BatchIterator(X, y, batch_size=32, shuffle=False, seed=42)
    seen_rows = []
    for xb, yb in it:
        assert xb.shape[0] == yb.shape[0]
        seen_rows.extend(xb[:, 0].astype(int).tolist())
    assert sorted(seen_rows) == list(range(100))


def test_batch_iterator_shuffles_per_epoch():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20).reshape(-1, 1).astype(float)
    it = BatchIterator(X, y, batch_size=20, shuffle=True, seed=42)
    first = next(iter(it))[0][:, 0].astype(int).tolist()
    second = next(iter(it))[0][:, 0].astype(int).tolist()
    # Same epoch, different shuffles
    assert first != list(range(20))


def test_train_val_split_zero_val_fraction():
    """val_fraction=0.0 → val=train (caso XOR sin holdout)."""
    y = np.array([0, 1, 0, 1])
    train_idx, val_idx = train_val_split(y, val_fraction=0.0, stratify=False, seed=42)
    assert len(train_idx) == 4
    assert len(val_idx) == 4
    assert set(train_idx.tolist()) == set(val_idx.tolist())
