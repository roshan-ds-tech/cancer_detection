"""Loads trusted artifacts once at startup and exposes them to the API layer."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.core import config
from app.ml.artifacts import load_trusted_model, read_json
from app.ml.explain import LocalExplainer
from app.ml.ood import OODChecker

log = logging.getLogger("oncosense")

PROBABILITY_NOTES = {
    "logistic_regression": "Native logistic-regression probability.",
    "svm": "Platt (sigmoid) calibrated SVM score, fitted with internal cross-validation.",
    "random_forest": "Fraction of tree votes; not separately calibrated.",
    "decision_tree": "Leaf class frequency; coarse and poorly calibrated.",
    "knn": "Fraction of neighbour votes; coarse.",
    "gaussian_nb": "Naive Bayes posterior; tends to be over-confident here.",
    "gradient_boosting": "Boosted-tree probability; not separately calibrated.",
}


class ArtifactsUnavailable(RuntimeError):
    pass


class Registry:
    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR):
        self.dir = artifacts_dir
        self.loaded = False
        self.load_error: str | None = None

    def load(self) -> None:
        try:
            self.pipeline, self.manifest = load_trusted_model(
                self.dir / "model_manifest.json", self.dir)
            self.metadata = read_json(self.dir / "metadata.json")
            self.metrics = read_json(self.dir / "metrics.json")
            self.explainability = read_json(self.dir / "explainability.json")
            self.ood = OODChecker(read_json(self.dir / "ood.json"))
            self.demo = read_json(self.dir / "demo_samples.json")
            self.train_records = read_json(self.dir / "train_records.json")
            eda_path = self.dir / "eda.json"
            self.eda = read_json(eda_path) if eda_path.exists() else None
            bg = read_json(self.dir / "background.json")
            self.feature_names: list[str] = self.metadata["feature_names"]
            if bg["features"] != self.feature_names:
                raise ValueError("Background feature order does not match model schema")
            self.explainer = LocalExplainer(
                self.pipeline, pd.DataFrame(bg["rows"], columns=bg["features"]), bg["seed"])
            if self.manifest["model_version"] != self.metadata["model_version"]:
                raise ValueError("Manifest and metadata model versions differ")
            self.demo_index = {sid: i for i, sid in enumerate(self.demo["sample_ids"])}
            self.loaded, self.load_error = True, None
            log.info("Loaded model %s", self.metadata["model_version"])
        except Exception as exc:  # app still starts; endpoints report 503
            self.loaded, self.load_error = False, f"{type(exc).__name__}: {exc}"
            log.error("Artifact load failed: %s", self.load_error)

    def require(self) -> "Registry":
        if not self.loaded:
            raise ArtifactsUnavailable(self.load_error or "Model artifacts not loaded")
        return self

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"]["operating_threshold"])

    @property
    def probability_note(self) -> str:
        return PROBABILITY_NOTES.get(self.metadata["selected_model"]["key"], "")


registry = Registry()
