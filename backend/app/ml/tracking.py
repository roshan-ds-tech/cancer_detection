"""Minimal SQLite experiment tracker (no external MLOps service)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    selected_model TEXT,
    model_version TEXT,
    selection_rule TEXT,
    library_versions TEXT,
    duration_seconds REAL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    hyperparameters TEXT,
    seed INTEGER,
    cv_metrics TEXT,
    test_metrics TEXT,
    training_seconds REAL,
    dataset_version TEXT,
    artifact_path TEXT,
    created_at TEXT NOT NULL
);
"""


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_run(db: Path, run: dict) -> None:
    with closing(_connect(db)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run["run_id"], run["created_at"], run["dataset_version"], run["seed"],
             run.get("selected_model"), run.get("model_version"), run.get("selection_rule"),
             json.dumps(run.get("library_versions", {})), run.get("duration_seconds")),
        )


def log_experiment(db: Path, exp: dict) -> None:
    with closing(_connect(db)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (exp["experiment_id"], exp["run_id"], exp["algorithm"], exp["status"],
             exp.get("error"), json.dumps(exp.get("hyperparameters")), exp.get("seed"),
             json.dumps(exp.get("cv_metrics")), json.dumps(exp.get("test_metrics")),
             exp.get("training_seconds"), exp.get("dataset_version"),
             exp.get("artifact_path"), now_iso()),
        )


def list_runs(db: Path) -> list[dict]:
    if not db.exists():
        return []
    with closing(_connect(db)) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["library_versions"] = json.loads(d["library_versions"] or "{}")
        out.append(d)
    return out


def list_experiments(db: Path, run_id: str | None = None) -> list[dict]:
    if not db.exists():
        return []
    q = "SELECT * FROM experiments"
    args: tuple = ()
    if run_id:
        q += " WHERE run_id = ?"
        args = (run_id,)
    q += " ORDER BY created_at DESC, algorithm"
    with closing(_connect(db)) as conn:
        rows = conn.execute(q, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("hyperparameters", "cv_metrics", "test_metrics"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out
