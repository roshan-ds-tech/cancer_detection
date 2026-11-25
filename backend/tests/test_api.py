import io


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True
    assert "Not validated for clinical use" in r.json()["disclaimer"]


def test_dataset_endpoints(client):
    s = client.get("/api/dataset/summary").json()
    assert s["dataset"]["n_samples"] == 569 and s["integrity"]["missing_values_total"] == 0
    f = client.get("/api/dataset/features").json()["features"]
    assert len(f) == 30 and f[0]["name"] == "mean_radius"
    assert client.get("/api/dataset/eda").status_code == 200
    rec = client.get("/api/dataset/records?limit=5").json()
    assert rec["split"] == "train" and len(rec["records"]) == 5


def test_model_endpoints(client):
    m = client.get("/api/models").json()
    assert sum(x["selected"] for x in m["models"]) == 1
    met = client.get("/api/models/metrics").json()
    assert all("cv" in x and "test" in x for x in met["models"])
    roc = client.get("/api/models/roc-data").json()
    assert roc["models"][0]["curves"]["roc"][0] == {"fpr": 0.0, "tpr": 0.0}
    fi = client.get("/api/models/feature-importance").json()
    assert len(fi["shap_global"]) == 30
    assert client.get("/api/research/methodology").json()["citations"]
    assert client.get("/api/experiments").json()["runs"]


def test_valid_prediction(client, demo_features):
    r = client.post("/api/predict", json={"features": demo_features})
    assert r.status_code == 200
    d = r.json()
    assert d["disclaimer"].startswith("Educational research prototype only")
    assert d["prediction"]["predicted_class"] in {"malignant", "benign"}
    assert 0 <= d["prediction"]["probability_malignant"] <= 1
    e = d["explanation"]
    assert abs(e["base_value"] + e["sum_of_contributions"] - d["prediction"]["probability_malignant"]) < 1e-6


def test_sample_prediction_and_404(client):
    sid = client.get("/api/demo/samples").json()["samples"][0]["sample_id"]
    d = client.post(f"/api/predict/sample/{sid}").json()
    assert d["ground_truth"] in {"malignant", "benign"}
    assert client.post("/api/predict/sample/999999").status_code == 404


def test_missing_and_unexpected_fields(client, demo_features):
    partial = dict(demo_features)
    partial.pop("mean_radius")
    r = client.post("/api/predict", json={"features": partial})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    extra = {**demo_features, "patient_age": 50}
    assert client.post("/api/predict", json={"features": extra}).status_code == 422


def test_invalid_values(client, demo_features):
    for bad in [-1, "abc", None]:
        r = client.post("/api/predict", json={"features": {**demo_features, "mean_area": bad}})
        assert r.status_code == 422
    # NaN / Infinity are not valid JSON numbers for this API
    body = '{"features": {' + ",".join(f'"{k}": {v}' for k, v in demo_features.items()).replace(
        str(demo_features["mean_area"]), "Infinity", 1) + "}}"
    assert client.post("/api/predict", content=body,
                       headers={"content-type": "application/json"}).status_code == 422


def test_extreme_input_abstains(client, demo_features):
    d = client.post("/api/predict", json={"features": {**demo_features, "mean_area": 90000}}).json()
    assert d["abstained"] is True and d["prediction"] is None


def _csv(client, rows, header=None, sid=True):
    feats = list(client.get("/api/demo/samples").json()["samples"][0]["features"].keys())
    header = header or ((["sample_id"] if sid else []) + feats)
    lines = [",".join(header)] + [",".join(map(str, r)) for r in rows]
    return "\n".join(lines).encode()


def test_batch_inference(client):
    samples = client.get("/api/demo/samples").json()["samples"][:3]
    rows = [[f"s{i}"] + list(s["features"].values()) for i, s in enumerate(samples)]
    bad = ["bad"] + list(samples[0]["features"].values())
    bad[3] = "oops"
    missing = ["miss"] + list(samples[0]["features"].values())
    missing[5] = ""
    data = _csv(client, rows + [bad, missing])
    r = client.post("/api/predict/batch", files={"file": ("x.csv", data, "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["summary"]["total"] == 5 and d["summary"]["invalid"] == 2
    assert d["rows"][0]["predicted_class"] in {"malignant", "benign"}
    rc = client.post("/api/predict/batch?format=csv", files={"file": ("x.csv", data, "text/csv")})
    assert rc.status_code == 200 and rc.headers["content-type"].startswith("text/csv")
    assert "Not validated for clinical use" in rc.text


def test_batch_schema_errors(client):
    samples = client.get("/api/demo/samples").json()["samples"][:2]
    feats = list(samples[0]["features"].keys())
    # unexpected column
    data = _csv(client, [], header=["sample_id"] + feats + ["patient_age"]) + b"\n" + b",".join(
        [b"a"] + [str(v).encode() for v in samples[0]["features"].values()] + [b"50"])
    r = client.post("/api/predict/batch", files={"file": ("x.csv", data, "text/csv")})
    assert r.status_code == 422 and "patient_age" in r.json()["error"]["details"]["unexpected_columns"]
    # missing column
    data = _csv(client, [], header=feats[1:])
    assert client.post("/api/predict/batch", files={"file": ("x.csv", data, "text/csv")}).status_code == 422
    # duplicate ids
    rows = [["dup"] + list(s["features"].values()) for s in samples]
    r = client.post("/api/predict/batch", files={"file": ("x.csv", _csv(client, rows), "text/csv")})
    assert r.status_code == 422 and "dup" in r.json()["error"]["details"]["duplicate_ids"]


def test_malformed_and_wrong_files(client):
    r = client.post("/api/predict/batch", files={"file": ("x.csv", b"\xff\xfe\x00\x01", "text/csv")})
    assert r.status_code == 422
    r = client.post("/api/predict/batch", files={"file": ("x.csv", b"", "text/csv")})
    assert r.status_code == 422
    r = client.post("/api/predict/batch", files={"file": ("x.exe", b"a,b\n1,2", "application/octet-stream")})
    assert r.status_code == 415


def test_oversized_csv(client):
    big = b"a" * 1_100_000
    r = client.post("/api/predict/batch", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413


def test_template_round_trip(client):
    t = client.get("/api/predict/template.csv")
    assert t.status_code == 200
    r = client.post("/api/predict/batch", files={"file": ("t.csv", t.content, "text/csv")})
    assert r.status_code == 200 and r.json()["summary"]["invalid"] == 0
