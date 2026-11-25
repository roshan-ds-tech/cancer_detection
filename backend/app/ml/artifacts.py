"""Artifact persistence with integrity checks.

joblib/pickle files can execute code when loaded, so the backend only loads a model
file that (a) lives inside the configured artifacts directory and (b) matches the
SHA-256 digest recorded in the manifest written by our own training script.
Uploaded files are never deserialised with joblib/pickle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from app.core.config import ARTIFACTS_DIR, MANIFEST_PATH


class ArtifactIntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def read_json(path: Path):
    return json.loads(path.read_text())


def save_model(pipeline, path: Path, base: Path = ARTIFACTS_DIR) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    return {"path": str(path.relative_to(base)), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size}


def _resolve_inside(base: Path, relative: str) -> Path:
    p = (base / relative).resolve()
    if base.resolve() not in p.parents:
        raise ArtifactIntegrityError(f"Refusing to load artifact outside {base}: {p}")
    return p


def load_trusted_model(manifest_path: Path = MANIFEST_PATH, base: Path = ARTIFACTS_DIR):
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No model manifest at {manifest_path}. Run `python scripts/train_models.py` first."
        )
    manifest = read_json(manifest_path)
    path = _resolve_inside(base, manifest["model"]["path"])
    digest = sha256_file(path)
    if digest != manifest["model"]["sha256"]:
        raise ArtifactIntegrityError(
            "Model file hash does not match the manifest; refusing to load."
        )
    return joblib.load(path), manifest
