"""Full workflow: data -> training -> selection -> artifacts -> API load -> predict -> explain."""
from app.ml import data as D


def test_end_to_end_workflow(trained, client):
    out, res = trained
    # artifacts written by training are what the API serves
    health = client.get("/api/health").json()
    assert health["model_version"] == res["metadata"]["model_version"]
    # every held-out sample can be scored, explained and compared to its label
    samples = client.get("/api/demo/samples").json()["samples"]
    assert len(samples) == 114
    correct = 0
    for s in samples[:15]:
        d = client.post(f"/api/predict/sample/{s['sample_id']}").json()
        if d["abstained"]:
            continue
        assert d["explanation"]["top_contributions"]
        assert set(d["features"]) == set(D.FEATURE_NAMES)
        correct += d["prediction"]["predicted_class"] == d["ground_truth"]
    assert correct >= 12  # sanity floor, not a performance claim
    # metrics served by the API equal those produced by training
    sel = res["metrics"]["selected_key"]
    api_m = next(m for m in client.get("/api/models/metrics").json()["models"] if m["key"] == sel)
    tr_m = next(m for m in res["metrics"]["models"] if m["key"] == sel)
    assert api_m["test"]["roc_auc"] == tr_m["test"]["roc_auc"]
