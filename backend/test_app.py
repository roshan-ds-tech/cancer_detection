"""Tests for the data, the models and the API. Run with:  pytest"""
from fastapi.testclient import TestClient

import data
import predictor
from main import app

client = TestClient(app)


def examples(kind):
    features = client.get("/api/dataset/summary").json()["features"]
    return {f["key"]: f[f"{kind}_example"] for f in features}


# ---------- data ----------
def test_dataset_loads():
    X, y = data.load_data()
    assert X.shape == (569, 10)
    assert y.sum() == 212  # malignant is the positive class


def test_split_is_stratified_and_separate():
    X_train, X_test, y_train, y_test = data.split_data()
    assert len(X_train) == 455 and len(X_test) == 114
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_eda_endpoint():
    eda = client.get("/api/dataset/eda").json()
    assert len(eda["correlation"]["matrix"]) == 10
    assert len(eda["histograms"]["radius"]) == 20


# ---------- models ----------
def test_all_six_models_trained():
    results = client.get("/api/models/results").json()
    assert len(results["results"]) == 6
    for r in results["results"]:
        assert 0.8 < r["accuracy"] <= 1 and 0.5 < r["roc_auc"] <= 1
        assert sum(map(sum, r["confusion_matrix"])) == 114


def test_best_model_has_highest_cv_accuracy():
    results = client.get("/api/models/results").json()
    best = max(r["cv_accuracy"] for r in results["results"])
    chosen = next(r for r in results["results"] if r["model"] == results["best_model"])
    assert chosen["cv_accuracy"] == best


def test_history_is_recorded():
    assert len(client.get("/api/models/history").json()) >= 1


# ---------- predictions ----------
def test_risk_levels():
    assert predictor.risk_level(0.1) == "Low"
    assert predictor.risk_level(0.5) == "Medium"
    assert predictor.risk_level(0.9) == "High"


def test_predict_examples():
    out = client.post("/api/predict", json=examples("benign")).json()
    assert out["prediction"] == "Benign" and out["level"] == "Low"
    assert len(out["explanation"]["items"]) == 10
    out = client.post("/api/predict", json=examples("malignant")).json()
    assert out["prediction"] == "Malignant" and out["level"] == "High"


def test_invalid_input_rejected():
    assert client.post("/api/predict", json={"radius": 10}).status_code == 422
    values = examples("benign")
    values["radius"] = -1
    assert client.post("/api/predict", json=values).status_code == 422


def test_unusual_values_flagged():
    values = examples("benign")
    values["area"] = 99999
    assert "Cell area" in client.post("/api/predict", json=values).json()["unusual_values"]


def test_test_sample_prediction():
    sample = client.get("/api/samples").json()[0]
    out = client.post(f"/api/predict/sample/{sample['id']}").json()
    assert out["actual"] in ("Benign", "Malignant")
    assert client.post("/api/predict/sample/999999").status_code == 404


def test_csv_batch():
    csv = client.get("/api/predict/template.csv").text
    csv += "bad-row," + ",".join(["abc"] + ["1"] * 9) + "\n"
    out = client.post("/api/predict/batch", files={"file": ("s.csv", csv, "text/csv")}).json()
    assert out["summary"] == {**out["summary"], "total": 4, "valid": 3, "invalid": 1}


def test_csv_wrong_columns_and_type():
    bad = client.post("/api/predict/batch", files={"file": ("s.csv", "id,age\n1,50\n", "text/csv")})
    assert bad.status_code == 422 and "Missing" in bad.json()["detail"]
    assert client.post("/api/predict/batch", files={"file": ("s.txt", "x", "text/plain")}).status_code == 415


def test_contributions_add_up():
    """Shapley values: average-sample prediction + contributions = this prediction."""
    out = client.post("/api/predict", json=examples("malignant")).json()
    total = out["explanation"]["base_value"] + sum(c["contribution"] for c in out["explanation"]["items"])
    assert abs(total - out["malignant_probability"]) < 0.002
