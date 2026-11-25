"""Train, tune and compare 6 machine learning models."""
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import FEATURE_KEYS, LABELS, SEED, split_data

SAVE_DIR = Path(__file__).parent / "saved"
MODEL_PATH = SAVE_DIR / "model.joblib"
RESULTS_PATH = SAVE_DIR / "results.json"
HISTORY_PATH = SAVE_DIR / "history.json"


def get_models():
    """Each model with a small grid of hyperparameters to try."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000), {"clf__C": [0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [1, 5]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=SEED),
                          {"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3]}),
        "K-Nearest Neighbors": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7, 11]}),
        # CalibratedClassifierCV gives the SVM a probability output
        "Support Vector Machine": (CalibratedClassifierCV(SVC(), ensemble=False),
                                   {"clf__estimator__C": [0.5, 1, 5, 10]}),
        "Naive Bayes": (GaussianNB(), {}),
    }


def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }, [[int(tn), int(fp)], [int(fn), int(tp)]], [
        {"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in zip(fpr, tpr)]


def train_models():
    """Tune every model with 5-fold cross-validation, test it, and save the best one."""
    started = time.time()
    X_train, X_test, y_train, y_test = split_data()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = [], {}
    for name, (classifier, grid) in get_models().items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
        search = GridSearchCV(pipeline, grid, cv=folds, scoring="accuracy")
        t0 = time.time()
        search.fit(X_train, y_train)
        fit_seconds = time.time() - t0
        model = search.best_estimator_
        fitted[name] = model

        test, cm, roc = evaluate(model, X_test, y_test)
        i = search.best_index_
        results.append({
            "model": name,
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
            "cv_accuracy": round(float(search.cv_results_["mean_test_score"][i]), 4),
            "cv_std": round(float(search.cv_results_["std_test_score"][i]), 4),
            **{k: round(float(v), 4) for k, v in test.items()},
            "confusion_matrix": cm,
            "roc_curve": roc,
            "train_seconds": round(fit_seconds, 2),
        })

    # The best model is the one with the highest cross-validation accuracy (the test set is not used to choose).
    best = max(results, key=lambda r: (r["cv_accuracy"], -r["cv_std"]))
    best_model = fitted[best["model"]]

    summary = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_model": best["model"],
        "results": results,
        "importance": feature_importance(fitted, best_model, X_test, y_test),
        "duration_seconds": round(time.time() - started, 1),
    }
    SAVE_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    add_to_history(summary)
    return summary


def feature_importance(fitted, best_model, X_test, y_test):
    perm = permutation_importance(best_model, X_test, y_test, scoring="accuracy",
                                  n_repeats=20, random_state=SEED)
    forest = fitted["Random Forest"].named_steps["clf"]
    logreg = fitted["Logistic Regression"].named_steps["clf"]

    def table(values, extra=None):
        rows = [{"feature": LABELS[k], "value": round(float(v), 4), **(extra[i] if extra else {})}
                for i, (k, v) in enumerate(zip(FEATURE_KEYS, values))]
        return sorted(rows, key=lambda r: abs(r["value"]), reverse=True)

    return {
        "permutation": table(perm.importances_mean, [{"std": round(float(s), 4)} for s in perm.importances_std]),
        "random_forest": table(forest.feature_importances_),
        "logistic_regression": table(logreg.coef_[0]),
    }


def add_to_history(summary):
    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    best = next(r for r in summary["results"] if r["model"] == summary["best_model"])
    history.insert(0, {
        "run": len(history) + 1,
        "trained_at": summary["trained_at"],
        "best_model": summary["best_model"],
        "cv_accuracy": best["cv_accuracy"],
        "test_accuracy": best["accuracy"],
        "test_recall": best["recall"],
        "duration_seconds": summary["duration_seconds"],
    })
    HISTORY_PATH.write_text(json.dumps(history[:50], indent=2))


def load_model():
    """Load the saved model, training it first if it does not exist yet."""
    if not MODEL_PATH.exists() or not RESULTS_PATH.exists():
        train_models()
    return joblib.load(MODEL_PATH), json.loads(RESULTS_PATH.read_text())


def load_history():
    return json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []

