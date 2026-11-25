"""Simple tests for the model and the API. Run with:  pytest"""
from fastapi.testclient import TestClient

import model as ml
from main import app

client = TestClient(app)


def test_dataset_loads():
    X, y = ml.load_data()
    assert X.shape == (569, 10)
    assert y.sum() == 212  # 212 malignant, 357 benign


def test_all_models_are_trained():
    results = client.get("/api/results").json()
    assert len(results["results"]) == 6
    for r in results["results"]:
        assert 0.8 < r["accuracy"] <= 1.0


def test_risk_levels():
    assert ml.risk_level(0.1) == "Low"
    assert ml.risk_level(0.5) == "Medium"
    assert ml.risk_level(0.9) == "High"


def test_predict_benign_and_malignant_examples():
    features = client.get("/api/dataset").json()["features"]
    benign = {f["key"]: f["benign_example"] for f in features}
    malignant = {f["key"]: f["malignant_example"] for f in features}

    out = client.post("/api/predict", json=benign).json()
    assert out["prediction"] == "Benign" and out["level"] == "Low"

    out = client.post("/api/predict", json=malignant).json()
    assert out["prediction"] == "Malignant" and out["level"] == "High"


def test_invalid_input_is_rejected():
    assert client.post("/api/predict", json={"radius": 10}).status_code == 422
    features = client.get("/api/dataset").json()["features"]
    values = {f["key"]: f["benign_example"] for f in features}
    values["radius"] = -5
    assert client.post("/api/predict", json=values).status_code == 422


def test_unusual_values_are_flagged():
    features = client.get("/api/dataset").json()["features"]
    values = {f["key"]: f["benign_example"] for f in features}
    values["area"] = 99999
    assert "Cell area" in client.post("/api/predict", json=values).json()["unusual_values"]
