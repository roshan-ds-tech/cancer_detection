"""Making predictions: single sample, test samples and CSV batch."""
import csv
import io
import itertools
import math

import numpy as np
import pandas as pd

from data import CLASS_NAMES, FEATURE_KEYS, LABELS, load_data, split_data

MAX_CSV_BYTES = 1_000_000
MAX_CSV_ROWS = 1000


def risk_level(probability):
    """Turn the malignant probability into an easy level."""
    if probability >= 0.7:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


def _probability(model, rows):
    frame = pd.DataFrame(rows, columns=FEATURE_KEYS)
    return model.predict_proba(frame)[:, 1]


def unusual_values(values):
    """Features whose value is outside the range seen in the dataset."""
    X, _ = load_data()
    return [LABELS[k] for k in FEATURE_KEYS if not X[k].min() <= values[k] <= X[k].max()]


def contributions(model, values):
    """Shapley values: how much each measurement moved the prediction.

    We compare the sample with an "average sample" (the mean of the training data). Every
    combination of measurements (2^10 = 1024) is tried, switching each one between the average
    value and the real value. A measurement's contribution is its fair (Shapley) share of the
    difference between this prediction and the prediction for the average sample.
    Positive = pushed towards malignant. The contributions add up exactly to that difference.
    """
    X_train, _, _, _ = split_data()
    n = len(FEATURE_KEYS)
    x = np.array([values[k] for k in FEATURE_KEYS], dtype=float)
    average = X_train[FEATURE_KEYS].mean().to_numpy()

    masks = np.array(list(itertools.product([0, 1], repeat=n)))  # row i is the binary form of i
    f = _probability(model, np.where(masks == 1, x, average))
    sizes = masks.sum(axis=1)
    weights = np.array([math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)])

    result = []
    for j, key in enumerate(FEATURE_KEYS):
        without = np.where(masks[:, j] == 0)[0]
        with_j = without + (1 << (n - 1 - j))
        phi = float(np.sum(weights[sizes[without]] * (f[with_j] - f[without])))
        result.append({"feature": LABELS[key], "value": values[key], "contribution": round(phi, 4)})
    result.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return {"base_value": round(float(f[0]), 4), "items": result}


def predict_one(model, values, explain=True):
    probability = float(_probability(model, [[values[k] for k in FEATURE_KEYS]])[0])
    return {
        "prediction": "Malignant" if probability >= 0.5 else "Benign",
        "malignant_probability": round(probability, 4),
        "level": risk_level(probability),
        "unusual_values": unusual_values(values),
        "explanation": contributions(model, values) if explain else None,
    }


def test_samples():
    """The 114 test samples (without their labels, so users can guess first)."""
    _, X_test, _, _ = split_data()
    return [{"id": int(i), "values": {k: round(float(v), 4) for k, v in row.items()}}
            for i, row in X_test.iterrows()]


def predict_test_sample(model, sample_id):
    _, X_test, _, y_test = split_data()
    if sample_id not in X_test.index:
        return None
    values = {k: float(v) for k, v in X_test.loc[sample_id].items()}
    result = predict_one(model, values)
    actual = CLASS_NAMES[int(y_test.loc[sample_id])]
    return {**result, "id": sample_id, "values": values, "actual": actual,
            "correct": actual == result["prediction"]}


class CSVError(ValueError):
    pass


def predict_csv(model, content: bytes):
    """Validate an uploaded CSV file and predict every valid row."""
    if len(content) > MAX_CSV_BYTES:
        raise CSVError("File is larger than 1 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CSVError("File must be a UTF-8 text CSV file.")

    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    missing = [k for k in FEATURE_KEYS if k not in header]
    unknown = [h for h in header if h not in FEATURE_KEYS and h != "id"]
    if missing or unknown:
        raise CSVError("Wrong columns. "
                       + (f"Missing: {', '.join(missing)}. " if missing else "")
                       + (f"Not allowed: {', '.join(unknown)}." if unknown else ""))

    rows = list(reader)
    if not rows:
        raise CSVError("The file has no data rows.")
    if len(rows) > MAX_CSV_ROWS:
        raise CSVError(f"Too many rows (maximum {MAX_CSV_ROWS}).")

    output = []
    for n, raw in enumerate(rows, start=1):
        raw = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        row_id = raw.get("id") or str(n)
        values, errors = {}, []
        for k in FEATURE_KEYS:
            try:
                v = float(raw[k])
                if v < 0 or not math.isfinite(v):
                    raise ValueError
                values[k] = v
            except ValueError:
                errors.append(f"{LABELS[k]}: '{raw[k]}' is not a valid number")
        if errors:
            output.append({"row": n, "id": row_id, "status": "invalid", "errors": errors})
        else:
            result = predict_one(model, values, explain=False)
            output.append({"row": n, "id": row_id, "status": "ok", "errors": [],
                           "prediction": result["prediction"],
                           "malignant_probability": result["malignant_probability"],
                           "level": result["level"], "unusual_values": result["unusual_values"]})

    ok = [r for r in output if r["status"] == "ok"]
    return {
        "summary": {
            "total": len(output),
            "valid": len(ok),
            "invalid": len(output) - len(ok),
            "benign": sum(r["prediction"] == "Benign" for r in ok),
            "malignant": sum(r["prediction"] == "Malignant" for r in ok),
        },
        "rows": output,
    }


def template_csv():
    """A CSV template with the right header and 3 example rows from the test set."""
    _, X_test, _, _ = split_data()
    lines = [",".join(["id"] + FEATURE_KEYS)]
    for i, row in X_test.head(3).iterrows():
        lines.append(",".join([f"sample-{i}"] + [str(round(float(v), 4)) for v in row]))
    return "\n".join(lines) + "\n"
