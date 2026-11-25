"""Inference orchestration: validation -> distribution check -> prediction -> explanation."""
from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from app.core import config
from app.services.registry import Registry

LIMITATIONS = [
    "Trained on 569 samples from a single institution (University of Wisconsin, early 1990s); "
    "no external validation.",
    "Inputs are image-derived nuclear measurements from a specific FNA digitisation process; "
    "values from other processes may not be comparable.",
    "The output is a dataset-class prediction, not a diagnosis and not a person's probability "
    "of having cancer.",
    "Feature attributions describe model behaviour, not biological causation.",
]

ID_COLUMN = "sample_id"


class BatchValidationError(ValueError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def _frame(reg: Registry, features: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[float(features[f]) for f in reg.feature_names]], columns=reg.feature_names)


def predict_features(reg: Registry, features: dict[str, float], explain: bool = True) -> dict:
    x = _frame(reg, features)
    check = reg.ood.check(x.to_numpy()[0])
    base = {
        "disclaimer": config.DISCLAIMER,
        "model_version": reg.model_version,
        "model_name": reg.metadata["selected_model"]["name"],
        "input_validation": "valid",
        "distribution_check": check,
        "limitations": LIMITATIONS,
    }
    if check["status"] == "abstain":
        return {**base, "abstained": True, "abstain_reason": check["messages"],
                "prediction": None, "explanation": None}
    prob = float(reg.pipeline.predict_proba(x)[0, 1])
    if not np.isfinite(prob):
        return {**base, "abstained": True, "abstain_reason": ["Model produced a non-finite score"],
                "prediction": None, "explanation": None}
    pred = "malignant" if prob >= reg.threshold else "benign"
    return {
        **base,
        "abstained": False,
        "abstain_reason": [],
        "prediction": {"predicted_class": pred, "probability_malignant": prob,
                       "operating_threshold": reg.threshold,
                       "probability_note": reg.probability_note},
        "explanation": reg.explainer.explain_row(x) if explain else None,
    }


def _safe_cell(v: str) -> str:
    """Neutralise spreadsheet formula injection in echoed identifiers."""
    return "'" + v if v and v[0] in "=+-@\t\r" else v


def parse_batch_csv(reg: Registry, raw: bytes) -> tuple[list[str | None], list[dict], list[list[str]]]:
    """Return (ids, parsed feature dicts or None per row, per-row error lists)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BatchValidationError("File is not valid UTF-8 text")
    if "\x00" in text:
        raise BatchValidationError("File contains binary content")
    reader = csv.reader(io.StringIO(text))
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise BatchValidationError(f"Malformed CSV: {exc}")
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise BatchValidationError("CSV is empty")
    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    required = reg.feature_names
    dupes = sorted({h for h in header if header.count(h) > 1})
    missing = [f for f in required if f not in header]
    unexpected = [h for h in header if h not in required and h != ID_COLUMN]
    if dupes or missing or unexpected:
        raise BatchValidationError(
            "CSV columns do not match the model schema",
            {"missing_columns": missing, "unexpected_columns": unexpected,
             "duplicate_columns": dupes,
             "hint": f"Required: the 30 feature columns; optional: '{ID_COLUMN}'. "
                     "Download the template from /api/predict/template.csv."})
    if not body:
        raise BatchValidationError("CSV has a header but no data rows")
    if len(body) > config.MAX_BATCH_ROWS:
        raise BatchValidationError(f"Too many rows: {len(body)} > limit {config.MAX_BATCH_ROWS}")
    col = {h: i for i, h in enumerate(header)}
    ids: list[str | None] = []
    parsed: list[dict | None] = []
    errors: list[list[str]] = []
    for r in body:
        errs = []
        if len(r) != len(header):
            errs.append(f"Row has {len(r)} fields, expected {len(header)}")
            ids.append(None)
            parsed.append(None)
            errors.append(errs)
            continue
        sid = r[col[ID_COLUMN]].strip() if ID_COLUMN in col else None
        ids.append(sid or None)
        vals = {}
        for f in required:
            cell = r[col[f]].strip()
            if cell == "":
                errs.append(f"{f}: missing value (missing values are not imputed)")
                continue
            try:
                v = float(cell)
            except ValueError:
                errs.append(f"{f}: not a number ({cell[:20]!r})")
                continue
            if not np.isfinite(v):
                errs.append(f"{f}: non-finite value")
            elif v < 0:
                errs.append(f"{f}: negative value not allowed")
            else:
                vals[f] = v
        parsed.append(None if errs else vals)
        errors.append(errs)
    present = [i for i in ids if i]
    dup_ids = sorted({i for i in present if present.count(i) > 1})
    if dup_ids:
        raise BatchValidationError("Duplicate sample_id values", {"duplicate_ids": dup_ids[:50]})
    return ids, parsed, errors


def predict_batch(reg: Registry, raw: bytes) -> dict:
    ids, parsed, errors = parse_batch_csv(reg, raw)
    out_rows = []
    valid_idx = [i for i, p in enumerate(parsed) if p is not None]
    checks = {i: reg.ood.check(np.array([parsed[i][f] for f in reg.feature_names])) for i in valid_idx}
    to_score = [i for i in valid_idx if checks[i]["status"] != "abstain"]
    probs = {}
    if to_score:
        X = pd.DataFrame([[parsed[i][f] for f in reg.feature_names] for i in to_score],
                         columns=reg.feature_names)
        for i, p in zip(to_score, reg.pipeline.predict_proba(X)[:, 1]):
            probs[i] = float(p)
    for i in range(len(parsed)):
        row = {"row": i + 1, "sample_id": _safe_cell(ids[i]) if ids[i] else None,
               "errors": errors[i], "warnings": []}
        if parsed[i] is None:
            row["status"] = "invalid"
        elif checks[i]["status"] == "abstain":
            row["status"] = "abstained"
            row["warnings"] = checks[i]["messages"]
        elif not np.isfinite(probs[i]):
            row["status"] = "abstained"
            row["warnings"] = ["Model produced a non-finite score"]
        else:
            p = probs[i]
            row["probability_malignant"] = p
            row["predicted_class"] = "malignant" if p >= reg.threshold else "benign"
            row["status"] = "warning" if checks[i]["status"] == "warning" else "predicted"
            row["warnings"] = checks[i]["messages"]
        out_rows.append(row)
    summary = {s: sum(r["status"] == s for r in out_rows)
               for s in ["predicted", "warning", "abstained", "invalid"]}
    summary["total"] = len(out_rows)
    return {"disclaimer": config.DISCLAIMER, "model_version": reg.model_version,
            "operating_threshold": reg.threshold, "summary": summary, "rows": out_rows}


def batch_to_csv(result: dict) -> str:
    buf = io.StringIO()
    buf.write(f"# {config.DISCLAIMER}\n")
    buf.write(f"# model_version={result['model_version']} "
              f"operating_threshold={result['operating_threshold']}\n")
    w = csv.writer(buf)
    w.writerow(["row", "sample_id", "status", "predicted_dataset_class",
                "probability_malignant", "errors", "warnings"])
    for r in result["rows"]:
        w.writerow([r["row"], r["sample_id"] or "", r["status"], r.get("predicted_class") or "",
                    "" if r.get("probability_malignant") is None else f"{r['probability_malignant']:.6f}",
                    "; ".join(r["errors"]), "; ".join(r["warnings"])])
    return buf.getvalue()


def template_csv(reg: Registry, n_examples: int = 3) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([ID_COLUMN] + reg.feature_names)
    for sid, feats in list(zip(reg.demo["sample_ids"], reg.demo["features"]))[:n_examples]:
        w.writerow([f"demo-{sid}"] + [feats[f] for f in reg.feature_names])
    return buf.getvalue()
