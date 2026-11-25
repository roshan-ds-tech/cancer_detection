"""Model explanations.

Local explanations use SHAP's model-agnostic Permutation explainer applied to the
fitted pipeline's predicted probability of the malignant class, with an
interventional (independent) background drawn from the training split. Because the
explainer wraps the whole pipeline, it works for whichever candidate is selected.

Attributions describe how this statistical model's output responds to its inputs.
They are not evidence of biological causation. With strongly correlated features
(radius / perimeter / area) credit can be split among correlated inputs in ways
that look arbitrary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

BACKGROUND_SIZE = 50
MAX_EVALS = 2 * 30 + 1 + 200  # >= 2*n_features + 1 required by the permutation algorithm


def background_sample(X_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    return X_train.sample(n=min(BACKGROUND_SIZE, len(X_train)), random_state=seed)


class LocalExplainer:
    def __init__(self, pipeline, background: pd.DataFrame, seed: int):
        self.pipeline = pipeline
        self.features = list(background.columns)
        self.seed = seed
        bg = background.to_numpy(dtype=float)

        def f(arr: np.ndarray) -> np.ndarray:
            return self.pipeline.predict_proba(pd.DataFrame(arr, columns=self.features))[:, 1]

        self._explainer = shap.explainers.Permutation(
            f, shap.maskers.Independent(bg, max_samples=len(bg)),
            feature_names=self.features, seed=seed,
        )

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        return self._explainer(X.to_numpy(dtype=float), max_evals=MAX_EVALS, silent=True)

    def explain_row(self, x: pd.DataFrame, top_k: int = 10) -> dict:
        ex = self.explain(x)
        values = ex.values[0]
        base = float(np.atleast_1d(ex.base_values)[0])
        order = np.argsort(-np.abs(values))
        contribs = [
            {
                "feature": self.features[i],
                "value": float(x.iloc[0, i]),
                "contribution": float(values[i]),
                "direction": "towards malignant" if values[i] > 0 else "towards benign",
            }
            for i in order
        ]
        return {
            "method": "SHAP permutation explainer (model-agnostic, interventional background "
                      f"of {BACKGROUND_SIZE} training samples)",
            "output_space": "predicted probability of the 'malignant' dataset class",
            "base_value": base,
            "sum_of_contributions": float(values.sum()),
            "top_contributions": contribs[:top_k],
            "all_contributions": contribs,
        }


def permutation_importances(pipeline, X, y, seed: int, n_repeats: int = 20) -> list[dict]:
    r = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    out = [
        {"feature": f, "importance_mean": float(m), "importance_std": float(s)}
        for f, m, s in zip(X.columns, r.importances_mean, r.importances_std)
    ]
    return sorted(out, key=lambda d: -d["importance_mean"])


def native_importances(pipeline, feature_names: list[str]) -> dict | None:
    """Model-specific importances where they exist (coefficients or impurity)."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()
        items = [{"feature": f, "value": float(c)} for f, c in zip(feature_names, coefs)]
        return {
            "kind": "standardised_coefficient",
            "description": "Logistic-regression coefficients on standardised features "
                           "(log-odds change per 1 SD; positive pushes towards malignant).",
            "values": sorted(items, key=lambda d: -abs(d["value"])),
        }
    if hasattr(clf, "feature_importances_"):
        items = [{"feature": f, "value": float(v)}
                 for f, v in zip(feature_names, clf.feature_importances_)]
        return {
            "kind": "impurity_importance",
            "description": "Mean decrease in impurity from the tree model (biased towards "
                           "high-cardinality features; computed on training data).",
            "values": sorted(items, key=lambda d: -d["value"]),
        }
    return None
