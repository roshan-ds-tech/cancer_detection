import json

import numpy as np
import pytest
from sklearn.base import clone

from app.ml import data as D
from app.ml import metrics as M
from app.ml.artifacts import ArtifactIntegrityError, load_trusted_model
from app.ml.models import get_candidates
from app.ml.ood import OODChecker


def test_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.6, 0.1, 0.1, 0.3, 0.2])
    m = M.threshold_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tp": 2, "fn": 1, "fp": 1, "tn": 4}
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_malignant_orientation_in_metrics():
    y = np.array([1, 1, 0, 0])
    m = M.threshold_metrics(y, np.array([0.9, 0.9, 0.1, 0.1]))
    assert m["recall"] == 1.0 and m["roc_auc"] == 1.0  # high score <-> malignant


def test_scaler_fit_only_on_training_data():
    X, y = D.load_dataset()
    Xtr, Xte, ytr, _ = D.split_dataset(X, y)
    pipe = clone(get_candidates(42)[0].pipeline).fit(Xtr, ytr)
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, Xtr.mean().to_numpy())
    assert not np.allclose(pipe.named_steps["scaler"].mean_, X.mean().to_numpy())


def test_training_run_outputs(trained):
    out, res = trained
    md, mt = res["metadata"], res["metrics"]
    assert md["feature_names"] == D.FEATURE_NAMES
    assert md["class_mapping"] == {"0": "benign", "1": "malignant"}
    keys = {m["key"] for m in mt["models"]}
    assert {"logistic_regression", "svm", "random_forest", "decision_tree", "knn", "gaussian_nb"} <= keys
    for m in mt["models"]:
        assert m["status"] == "completed", m.get("error")
        assert len(m["cv"]["roc_auc"]["folds"]) == 5  # quick: 5x1 outer folds
        assert 0.5 < m["test"]["roc_auc"] <= 1.0
    for f in ["metadata.json", "metrics.json", "explainability.json", "ood.json", "model_manifest.json"]:
        assert (out / f).exists()


def test_selection_does_not_depend_on_test_set(trained):
    from app.ml.training import select_model

    _, res = trained
    stripped = [{k: v for k, v in m.items() if not k.startswith("test")} for m in res["metrics"]["models"]]
    assert select_model(stripped)[0] == res["metrics"]["selected_key"]


def test_artifact_loads_and_predicts_finite(trained):
    out, res = trained
    pipe, manifest = load_trusted_model(out / "model_manifest.json", out)
    X, y = D.load_dataset()
    p = pipe.predict_proba(X)[:, 1]
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert list(pipe.feature_names_in_) == D.FEATURE_NAMES
    # sanity: model ranks malignant above benign
    assert p[y == 1].mean() > p[y == 0].mean()


def test_tampered_artifact_is_refused(trained, tmp_path):
    out, _ = trained
    manifest = json.loads((out / "model_manifest.json").read_text())
    manifest["model"]["sha256"] = "0" * 64
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactIntegrityError):
        load_trusted_model(tmp_path / "m.json", out)
    manifest["model"]["path"] = "../../etc/passwd"
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactIntegrityError):
        load_trusted_model(tmp_path / "m.json", out)


def test_ood_checker_flags_extremes():
    X, y = D.load_dataset()
    Xtr, Xte, *_ = D.split_dataset(X, y)
    ood = OODChecker.fit(Xtr)
    assert ood.check(Xtr.median().to_numpy())["status"] == "ok"
    x = Xtr.median().to_numpy().copy()
    x[D.FEATURE_NAMES.index("mean_area")] = 50_000
    assert ood.check(x)["status"] == "abstain"
    # most real held-out samples should not be abstained on
    statuses = [ood.check(r)["status"] for r in Xte.to_numpy()]
    assert statuses.count("abstain") <= 2
