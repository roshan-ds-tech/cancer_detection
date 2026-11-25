"""Dataset loading, schema definition and train/test partitioning.

Class orientation: scikit-learn's ``load_breast_cancer`` encodes 0 = malignant and
1 = benign. OncoSense re-encodes the target so that **1 = malignant (positive class)**
and 0 = benign. Every metric in the project uses this orientation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from app.core.config import RANDOM_SEED, TEST_SIZE

POSITIVE_LABEL = "malignant"
NEGATIVE_LABEL = "benign"
CLASS_MAPPING = {0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}
EXPECTED_ROWS = 569
EXPECTED_FEATURES = 30

BASE_MEASUREMENTS = {
    "radius": "Mean of distances from the nucleus centre to points on its perimeter.",
    "texture": "Standard deviation of grey-scale values inside the nucleus boundary.",
    "perimeter": "Length of the nucleus boundary.",
    "area": "Area enclosed by the nucleus boundary.",
    "smoothness": "Local variation in radius lengths.",
    "compactness": "Perimeter^2 / area - 1.0.",
    "concavity": "Severity of concave portions of the contour.",
    "concave_points": "Number of concave portions of the contour.",
    "symmetry": "Symmetry of the nucleus boundary.",
    "fractal_dimension": "'Coastline approximation' - 1.",
}
# Units are not specified in the source documentation. Size measures are computed
# on the digitised image and are reported in image-derived units; the rest are ratios
# or counts. We state this explicitly rather than invent physical units.
UNITS = {
    "radius": "image units (unspecified in source)",
    "texture": "grey-scale units",
    "perimeter": "image units (unspecified in source)",
    "area": "image units^2 (unspecified in source)",
    "smoothness": "dimensionless",
    "compactness": "dimensionless",
    "concavity": "dimensionless",
    "concave_points": "dimensionless",
    "symmetry": "dimensionless",
    "fractal_dimension": "dimensionless",
}
STAT_DESCRIPTIONS = {
    "mean": "mean across all nuclei in the image",
    "se": "standard error across all nuclei in the image",
    "worst": "mean of the three largest values in the image",
}


def _snake(sklearn_name: str) -> str:
    """'mean concave points' -> 'mean_concave_points'; 'radius error' -> 'radius_se'."""
    name = sklearn_name.replace(" error", " se")
    return name.replace(" ", "_")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    sklearn_name: str
    measurement: str
    statistic: str
    description: str
    unit: str


def feature_specs() -> list[FeatureSpec]:
    raw = load_breast_cancer()
    specs = []
    for sk_name in raw.feature_names:
        name = _snake(str(sk_name))
        if name.startswith("mean_"):
            stat, meas = "mean", name[len("mean_"):]
        elif name.startswith("worst_"):
            stat, meas = "worst", name[len("worst_"):]
        else:
            stat, meas = "se", name[: -len("_se")]
        specs.append(
            FeatureSpec(
                name=name,
                sklearn_name=str(sk_name),
                measurement=meas,
                statistic=stat,
                description=f"{BASE_MEASUREMENTS[meas]} Statistic: {STAT_DESCRIPTIONS[stat]}.",
                unit=UNITS[meas],
            )
        )
    return specs


FEATURE_NAMES: list[str] = [s.name for s in feature_specs()]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with snake_case columns and y = 1 for malignant."""
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = [_snake(c) for c in X.columns]
    # sklearn: 0 = malignant, 1 = benign  ->  OncoSense: 1 = malignant
    assert list(raw.target_names) == ["malignant", "benign"], "Unexpected sklearn label order"
    y = (raw.target == 0).astype(int).rename("malignant")
    validate_schema(X, y)
    return X, y


def validate_schema(X: pd.DataFrame, y: pd.Series) -> None:
    if X.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(f"Unexpected dataset shape {X.shape}")
    if list(X.columns) != FEATURE_NAMES:
        raise ValueError("Feature names do not match schema")
    if X.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Dataset contains non-finite values")
    if set(y.unique()) != {0, 1}:
        raise ValueError("Target must be binary")
    if int(y.sum()) != 212:
        raise ValueError("Expected 212 malignant samples")


def dataset_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def split_dataset(
    X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_SEED, test_size: float = TEST_SIZE
):
    """Stratified hold-out split. Row index (0..568) is preserved as the sample ID."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
