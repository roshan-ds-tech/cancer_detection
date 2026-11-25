# API Documentation

Base URL: `http://localhost:8000`. Interactive OpenAPI docs are at `/docs`, and the schema is at `/openapi.json`.
All responses are JSON unless noted otherwise. Prediction responses include `disclaimer` and `model_version`.

**Error envelope (all errors)**

```json
{"error": {"code": "validation_error", "message": "Input failed validation; no prediction was made.",
           "details": [{"loc": ["body","features","mean_area"], "msg": "Input should be greater than or equal to 0", "type": "greater_than_equal"}]}}
```

| Status | `code` | When |
|---|---|---|
| 404 | `request_error` | Unknown demo sample, missing EDA artifact |
| 413 | `payload_too_large` / `request_error` | Body > 1 MB (+ multipart overhead) |
| 415 | `request_error` | Uploaded file is not `.csv` |
| 422 | `validation_error` / `request_error` | Schema violation, malformed or mismatched CSV |
| 503 | `model_unavailable` | Artifacts missing or failed integrity check (run training) |
| 500 | `internal_error` | Unexpected error (no stack trace exposed) |

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | `status` (`ok`/`degraded`), `model_loaded`, `model_version`, `load_error`, `disclaimer` |
| GET | `/api/dataset/summary` | Dataset metadata, integrity checks, per-feature statistics (training split), scaling note |
| GET | `/api/dataset/features` | The 30 feature specs: name, description, statistic, unit, training min/max/median; missing-value policy |
| GET | `/api/dataset/eda` | Histograms, box plots, correlation matrix, high-correlation pairs, VIF, PCA points |
| GET | `/api/dataset/records?limit=25&offset=0` | Paginated **training** records with labels (limit 1–100) |
| GET | `/api/models` | All candidates: status, error, best params, grid, fit time, size, inference timing, `selected`; selection rationale, calibration, threshold, config, versions |
| GET | `/api/models/metrics` | Per model: `cv` (mean/std/se/ci95/folds per metric), `test`, `test_ci95`, classification report; selected model at the operating threshold; test errors; learning curve |
| GET | `/api/models/roc-data` | Per model: ROC, PR and calibration points plus a threshold sweep (test); OOF sweep and calibration for the selected model |
| GET | `/api/models/feature-importance` | Permutation (all models), native, global SHAP, correlation, caveat |
| GET | `/api/experiments?run_id=` | Tracker runs and experiments |
| GET | `/api/demo/samples` | 114 held-out samples **without labels** |
| POST | `/api/predict` | Single prediction (below) |
| POST | `/api/predict/sample/{sample_id}` | Predict a held-out sample; response adds `sample_id`, `features`, `ground_truth` (UI hides it until revealed) |
| POST | `/api/predict/batch?format=json\|csv` | CSV batch inference (multipart field `file`) |
| GET | `/api/predict/template.csv` | CSV template with header + 3 example rows |
| GET | `/api/research/methodology` | Methodology sections (run-specific) + citations |
| GET | `/static/figures/{name}.png` | Static figures |

### POST /api/predict

Request: all 30 features are required. Each must be a finite number ≥ 0. Extra keys are rejected.

```json
{"features": {"mean_radius": 11.41, "mean_texture": 10.82, "...": "...", "worst_fractal_dimension": 0.08523},
 "explain": true}
```

Response (abridged, real output for held-out sample 120):

```json
{
  "disclaimer": "Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.",
  "model_version": "oncosense-svm-20261125T114504Z-d23509",
  "model_name": "Support Vector Machine (RBF)",
  "input_validation": "valid",
  "abstained": false,
  "abstain_reason": [],
  "prediction": {"predicted_class": "benign", "probability_malignant": 0.000397,
                 "operating_threshold": 0.39,
                 "probability_note": "Platt (sigmoid) calibrated SVM score, fitted with internal cross-validation."},
  "distribution_check": {"status": "ok", "mahalanobis_distance": 3.5, "mahalanobis_p99_train": 12.75,
                         "mahalanobis_max_train": 16.75, "flagged_features": [], "messages": [], "note": "..."},
  "explanation": {"method": "SHAP permutation explainer ...", "output_space": "predicted probability of the 'malignant' dataset class",
                  "base_value": 0.2384, "sum_of_contributions": -0.2380,
                  "top_contributions": [{"feature": "worst_texture", "value": 15.97, "contribution": -0.0700, "direction": "towards benign"}],
                  "all_contributions": ["... 30 items ..."]},
  "limitations": ["..."]
}
```

When `abstained` is `true`, `prediction` and `explanation` are `null`, and `abstain_reason` lists why.
`distribution_check.status` is `ok`, `warning` or `abstain`.

### POST /api/predict/batch

```bash
curl -F "file=@samples.csv" http://localhost:8000/api/predict/batch
curl -F "file=@samples.csv" "http://localhost:8000/api/predict/batch?format=csv" -o predictions.csv
```

- The header must contain exactly the 30 feature names, plus an optional `sample_id`. Order does not matter.
- File-level 422 errors: missing, unexpected or duplicate columns (listed in `details`); duplicate `sample_id`; empty file; non-UTF-8 or binary content; more than 2,000 rows.
- Row-level errors are returned per row with `status: "invalid"`: missing, non-numeric, negative or non-finite cells, or a wrong field count. Values are never imputed.
- Row statuses are `predicted`, `warning` (predicted, but the input is unusual), `abstained` or `invalid`.

Response: `{disclaimer, model_version, operating_threshold, summary: {predicted, warning, abstained, invalid, total}, rows: [{row, sample_id, status, predicted_class, probability_malignant, errors, warnings}]}`.

The CSV format adds the disclaimer and model version as `#` comment lines.
