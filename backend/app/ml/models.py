"""Candidate model definitions and (deliberately small) hyperparameter grids.

Every candidate is a scikit-learn Pipeline so that any fitted preprocessing (scaling)
is learned only from the data passed to ``fit`` — i.e. only from training folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class Candidate:
    key: str
    name: str
    pipeline: Pipeline
    param_grid: dict
    needs_scaling: bool
    complexity_rank: int  # lower = simpler / more interpretable (used for tie-breaks)
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def get_candidates(seed: int) -> list[Candidate]:
    return [
        Candidate(
            key="logistic_regression",
            name="Logistic Regression",
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
            ]),
            param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
            needs_scaling=True,
            complexity_rank=1,
            notes="L2-regularised linear model; native probabilities.",
        ),
        Candidate(
            key="svm",
            name="Support Vector Machine (RBF)",
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                # SVC(probability=True) is deprecated in sklearn 1.9; this is the documented
                # replacement: Platt (sigmoid) scaling fitted with internal 5-fold CV.
                ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed),
                                               method="sigmoid", cv=5, ensemble=False)),
            ]),
            param_grid={"clf__estimator__C": [0.1, 1.0, 10.0],
                        "clf__estimator__gamma": ["scale", 0.01]},
            needs_scaling=True,
            complexity_rank=4,
            notes="Probabilities from Platt (sigmoid) scaling via CalibratedClassifierCV(ensemble=False).",
        ),
        Candidate(
            key="random_forest",
            name="Random Forest",
            pipeline=Pipeline([
                ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)),
            ]),
            param_grid={"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3]},
            needs_scaling=False,
            complexity_rank=5,
            notes="Tree ensemble; scale-invariant, so no scaler.",
        ),
        Candidate(
            key="decision_tree",
            name="Decision Tree",
            pipeline=Pipeline([("clf", DecisionTreeClassifier(random_state=seed))]),
            param_grid={"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 5]},
            needs_scaling=False,
            complexity_rank=2,
            notes="Single CART tree; probabilities are leaf frequencies (coarse).",
        ),
        Candidate(
            key="knn",
            name="K-Nearest Neighbors",
            pipeline=Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            param_grid={"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]},
            needs_scaling=True,
            complexity_rank=3,
            notes="Distance-based; requires scaling. Probabilities are neighbour vote fractions.",
        ),
        Candidate(
            key="gaussian_nb",
            name="Gaussian Naive Bayes",
            pipeline=Pipeline([("clf", GaussianNB())]),
            param_grid={"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
            needs_scaling=False,
            complexity_rank=1,
            notes="Assumes conditional feature independence — strongly violated here "
                  "(radius/perimeter/area are near-collinear), so probabilities tend to be extreme.",
        ),
        Candidate(
            key="gradient_boosting",
            name="Gradient Boosting (optional)",
            pipeline=Pipeline([
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]),
            param_grid={"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
                        "clf__learning_rate": [0.1]},
            needs_scaling=False,
            complexity_rank=6,
            notes="Optional boosted-tree candidate (scikit-learn implementation).",
            tags=["optional"],
        ),
    ]
