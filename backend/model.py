"""Machine learning part of the project.

Loads the Breast Cancer Wisconsin dataset, trains 6 classic models,
compares them and saves the best one so the API can make predictions.
"""
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SAVE_DIR = Path(__file__).parent / "saved"
MODEL_PATH = SAVE_DIR / "model.joblib"
RESULTS_PATH = SAVE_DIR / "results.json"
SEED = 42

# We only use the 10 "mean" measurements of the dataset and give them simple names.
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


@lru_cache
def load_data():
    """Return the features (X) and the label (y): 1 = malignant, 0 = benign."""
    data = load_breast_cancer(as_frame=True)
    X = data.frame[[f["column"] for f in FEATURES]].copy()
    X.columns = FEATURE_KEYS
    # In scikit-learn's copy of the dataset 0 means malignant, so we flip it.
    y = (data.target == 0).astype(int)
    return X, y


def get_models():
    """The 6 models we compare. Each one scales the data first."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        # CalibratedClassifierCV lets the SVM give a probability
        "Support Vector Machine": CalibratedClassifierCV(SVC(), ensemble=False),
        "Naive Bayes": GaussianNB(),
    }


def train_models():
    """Train all models, compare them and save the best one. Returns the results."""
    X, y = load_data()
    # 80% of the rows are used for training, 20% are kept aside for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    results = []
    trained = {}
    for name, classifier in get_models().items():
        model = make_pipeline(StandardScaler(), classifier)
        # 5-fold cross-validation on the training data is used to choose the best model
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained[name] = model
        results.append({
            "model": name,
            "cv_accuracy": round(float(cv_accuracy), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        })

    best = max(results, key=lambda r: r["cv_accuracy"])
    forest = trained["Random Forest"][-1]
    importance = sorted(
        ({"feature": f["label"], "importance": round(float(v), 4)}
         for f, v in zip(FEATURES, forest.feature_importances_)),
        key=lambda d: d["importance"], reverse=True)

    summary = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_model": best["model"],
        "results": results,
        "feature_importance": importance,
    }
    SAVE_DIR.mkdir(exist_ok=True)
    joblib.dump(trained[best["model"]], MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    return summary


def load_model():
    """Load the saved best model, training it first if it does not exist yet."""
    if not MODEL_PATH.exists() or not RESULTS_PATH.exists():
        train_models()
    return joblib.load(MODEL_PATH), json.loads(RESULTS_PATH.read_text())


def risk_level(probability):
    if probability >= 0.7:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


def predict(model, values):
    """Predict one sample. `values` is a dict with the 10 feature keys."""
    row = pd.DataFrame([[values[k] for k in FEATURE_KEYS]], columns=FEATURE_KEYS)
    probability = float(model.predict_proba(row)[0, 1])

    X, _ = load_data()
    unusual = [f["label"] for f in FEATURES
               if not X[f["key"]].min() <= values[f["key"]] <= X[f["key"]].max()]
    return {
        "prediction": "Malignant" if probability >= 0.5 else "Benign",
        "malignant_probability": round(probability * 100, 1),
        "level": risk_level(probability),
        "unusual_values": unusual,
    }


def dataset_summary():
    """Simple facts and chart data about the dataset."""
    X, y = load_data()
    features = []
    for f in FEATURES:
        col = X[f["key"]]
        features.append({
            "key": f["key"], "label": f["label"], "help": f["help"],
            "min": round(float(col.min()), 4), "max": round(float(col.max()), 4),
            "benign_avg": round(float(col[y == 0].mean()), 4),
            "malignant_avg": round(float(col[y == 1].mean()), 4),
            # typical (median) values, used for the "fill example" buttons
            "benign_example": round(float(col[y == 0].median()), 4),
            "malignant_example": round(float(col[y == 1].median()), 4),
        })
    points = [{"radius": round(float(r), 2), "texture": round(float(t), 2),
               "diagnosis": "Malignant" if d else "Benign"}
              for r, t, d in zip(X["radius"], X["texture"], y)]
    # 5 benign and 5 malignant rows to show in a table
    rows = pd.concat([X[y == 0].head(5), X[y == 1].head(5)]).sort_index()
    sample = rows.round(4).assign(diagnosis=["Malignant" if y[i] else "Benign" for i in rows.index])
    return {
        "total": len(X),
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum()),
        "features": features,
        "scatter": points,
        "sample_rows": sample.to_dict(orient="records"),
    }
