"""Shared fixtures. A quick training run is executed once per session into a temporary
artifacts directory, so tests exercise the real pipeline without touching the committed
artifacts."""
import pytest
from fastapi.testclient import TestClient

from app.ml.artifacts import write_json
from app.ml.eda import compute_eda
from app.ml.training import run_training


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    res = run_training(seed=42, artifacts_dir=out, quick=True, make_figures=False, log=lambda *_: None)
    write_json(out / "eda.json", compute_eda(seed=42))
    return out, res


@pytest.fixture(scope="session")
def client(trained):
    from app.main import app
    from app.services.registry import registry

    registry.dir = trained[0]
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def demo_features(client):
    return client.get("/api/demo/samples").json()["samples"][0]["features"]
