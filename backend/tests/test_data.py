import numpy as np
from sklearn.datasets import load_breast_cancer

from app.ml import data as D


def test_dataset_shape_and_features():
    X, y = D.load_dataset()
    assert X.shape == (569, 30)
    assert len(D.FEATURE_NAMES) == 30 and list(X.columns) == D.FEATURE_NAMES
    assert X.isna().sum().sum() == 0 and np.isfinite(X.to_numpy()).all()


def test_target_orientation_malignant_is_positive():
    X, y = D.load_dataset()
    raw = load_breast_cancer()
    assert int(y.sum()) == 212 and int((1 - y).sum()) == 357
    # sklearn's 0 means malignant; ours must be 1 for exactly those rows
    assert np.array_equal(y.to_numpy(), (raw.target == 0).astype(int))
    assert D.CLASS_MAPPING[1] == "malignant"


def test_no_target_or_id_leakage_in_features():
    X, y = D.load_dataset()
    banned = {"id", "target", "malignant", "diagnosis", "label", "sample_id"}
    assert not banned & set(X.columns)
    for c in X.columns:  # no feature perfectly reproduces the label
        assert abs(np.corrcoef(X[c], y)[0, 1]) < 0.95


def test_stratified_disjoint_split():
    X, y = D.load_dataset()
    Xtr, Xte, ytr, yte = D.split_dataset(X, y, seed=42)
    assert len(Xtr) == 455 and len(Xte) == 114
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert set(Xtr.index) | set(Xte.index) == set(range(569))
    assert abs(ytr.mean() - yte.mean()) < 0.01


def test_split_is_reproducible():
    X, y = D.load_dataset()
    a = D.split_dataset(X, y, seed=42)[1].index
    b = D.split_dataset(X, y, seed=42)[1].index
    c = D.split_dataset(X, y, seed=7)[1].index
    assert list(a) == list(b) and list(a) != list(c)
