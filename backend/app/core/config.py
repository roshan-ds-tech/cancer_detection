"""Central configuration. Values can be overridden through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("ONCOSENSE_ARTIFACTS_DIR", BACKEND_DIR / "artifacts")).resolve()
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODELS_DIR = ARTIFACTS_DIR / "models"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
EXPERIMENT_DB = ARTIFACTS_DIR / "experiments.sqlite"
MANIFEST_PATH = ARTIFACTS_DIR / "model_manifest.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20

# Upload / processing limits for batch inference
MAX_UPLOAD_BYTES = int(os.environ.get("ONCOSENSE_MAX_UPLOAD_BYTES", 1_000_000))  # 1 MB
MAX_BATCH_ROWS = int(os.environ.get("ONCOSENSE_MAX_BATCH_ROWS", 2_000))

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ONCOSENSE_CORS_ORIGINS", "http://localhost:5180,http://127.0.0.1:5180"
    ).split(",")
    if o.strip()
]

DISCLAIMER = (
    "Educational research prototype only. Not validated for clinical use. "
    "This result is not a medical diagnosis."
)
