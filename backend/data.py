"""Dataset loading and exploratory data analysis (EDA)."""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2

# The 10 "mean" measurements of the dataset, with simple names for the app.
FEATURES = [
    {"key": "radius", "column": "mean radius", "label": "Cell radius",
     "help": "Average distance from the centre of the cell nucleus to its edge"},
    {"key": "texture", "column": "mean texture", "label": "Texture",
     "help": "How much the grey colour varies inside the cell"},
    {"key": "perimeter", "column": "mean perimeter", "label": "Cell perimeter",
     "help": "Length of the outline of the cell nucleus"},
    {"key": "area", "column": "mean area", "label": "Cell area",
     "help": "Size of the cell nucleus"},
    {"key": "smoothness", "column": "mean smoothness", "label": "Smoothness",
     "help": "How smooth the edge of the cell is"},
    {"key": "compactness", "column": "mean compactness", "label": "Compactness",
     "help": "How tightly packed the cell shape is"},
    {"key": "concavity", "column": "mean concavity", "label": "Concavity",
     "help": "How deep the inward curves of the edge are"},
    {"key": "concave_points", "column": "mean concave points", "label": "Concave points",
     "help": "How many inward curves the edge has"},
    {"key": "symmetry", "column": "mean symmetry", "label": "Symmetry",
     "help": "How symmetrical the cell is"},
    {"key": "fractal_dimension", "column": "mean fractal dimension", "label": "Edge roughness",
     "help": "How rough or irregular the edge of the cell is"},
]
FEATURE_KEYS = [f["key"] for f in FEATURES]
LABELS = {f["key"]: f["label"] for f in FEATURES}
CLASS_NAMES = {0: "Benign", 1: "Malignant"}


@lru_cache
def load_data():
    """Return X (10 features) and y (1 = malignant, 0 = benign)."""
    data = load_breast_cancer(as_frame=True)
    X = data.frame[[f["column"] for f in FEATURES]].copy()
    X.columns = FEATURE_KEYS
    # scikit-learn stores malignant as 0, so we flip it to make malignant the positive class
    y = (data.target == 0).astype(int)
    return X, y


@lru_cache
def split_data():
    """Stratified 80/20 train/test split. Always the same because the seed is fixed."""
    X, y = load_data()
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def summary():
    X, y = load_data()
    X_train, _, _, _ = split_data()
    features = []
    for f in FEATURES:
        col = X[f["key"]]
        features.append({
            "key": f["key"], "label": f["label"], "help": f["help"], "column": f["column"],
            "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4),
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "benign_mean": round(float(col[y == 0].mean()), 4),
            "malignant_mean": round(float(col[y == 1].mean()), 4),
            # typical values of each class (median), used for "fill example" buttons
            "benign_example": round(float(col[y == 0].median()), 4),
            "malignant_example": round(float(col[y == 1].median()), 4),
            "train_mean": round(float(X_train[f["key"]].mean()), 4),
        })
    return {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository (via scikit-learn)",
        "total": len(X),
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum()),
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "train_rows": len(X_train),
        "test_rows": len(X) - len(X_train),
        "features": features,
    }


def eda():
    """Histograms by class, correlation matrix and a scatter plot."""
    X, y = load_data()
    histograms = {}
    for key in FEATURE_KEYS:
        edges = np.histogram_bin_edges(X[key], bins=20)
        benign, _ = np.histogram(X[key][y == 0], bins=edges)
        malignant, _ = np.histogram(X[key][y == 1], bins=edges)
        histograms[key] = [
            {"bin": round(float((edges[i] + edges[i + 1]) / 2), 4),
             "benign": int(benign[i]), "malignant": int(malignant[i])}
            for i in range(len(benign))
        ]
    corr = X.corr().round(3)
    scatter = [{"radius": round(float(r), 2), "texture": round(float(t), 2), "diagnosis": CLASS_NAMES[d]}
               for r, t, d in zip(X["radius"], X["texture"], y)]
    return {
        "histograms": histograms,
        "correlation": {"features": [LABELS[k] for k in FEATURE_KEYS], "matrix": corr.values.tolist()},
        "scatter": scatter,
    }


def records(limit=20, offset=0):
    X, y = load_data()
    page = X.iloc[offset: offset + limit].round(4)
    rows = [{"id": int(i), **row, "diagnosis": CLASS_NAMES[int(y[i])]} for i, row in page.iterrows()]
    return {"total": len(X), "limit": limit, "offset": offset, "rows": rows}
