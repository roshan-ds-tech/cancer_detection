# System Design

## Goals and constraints

| Requirement | Design response |
|---|---|
| ₹0 budget, offline after setup | Only open-source libraries. The dataset ships with scikit-learn. No cloud services, paid APIs or LLMs |
| 8 GB laptop | Small grids; nested CV takes about 45 s; peak memory is far below 1 GB; SHAP uses a 50-row background |
| No fake numbers | The UI only renders API data read from training artifacts. Test fixtures are captured from the real API |
| Reproducible | Fixed seed, pinned versions, dataset fingerprint, SQLite run log |
| Safe | Strict input schema, upload limits, hash-verified model loading, no stored uploads, disclaimers |

## Artifact contract (`backend/artifacts/`)

| File | Content | Consumer |
|---|---|---|
| `models/oncosense-<key>-<run>.joblib` | Fitted final pipeline (compressed) | Registry |
| `model_manifest.json` | Model path (relative) + SHA-256 + version | Registry (integrity gate) |
| `metadata.json` | Feature names and specs, class mapping, selection rationale, calibration, threshold (+ OOF sweep), dataset info, config, library versions | `/api/models`, `/api/research/*` |
| `metrics.json` | Per-model CV/test metrics, CIs, curves, sweeps, reports, inference timing, sizes, errors, learning curve | `/api/models/metrics`, `/api/models/roc-data` |
| `explainability.json` | Permutation (all models), native, global SHAP, correlation | `/api/models/feature-importance` |
| `ood.json` | Training min/max/median/IQR, scaling, LW precision, Mahalanobis thresholds | OOD checker |
| `background.json` | 50 training rows for SHAP | LocalExplainer |
| `demo_samples.json` | Test rows (features + labels) | Demo endpoints |
| `train_records.json` | Training rows | `/api/dataset/records` |
| `eda.json` | EDA statistics | `/api/dataset/*` |
| `figures/*.png` | Static figures | `/static/figures` |
| `candidates/*.joblib` | Each tuned candidate (for size/reproducibility, not served) | — |
| `experiments.sqlite` | `runs`, `experiments` tables | `/api/experiments` |

Only the model files are pickles, and they are only loaded after verification. Everything else is JSON.

## Experiment tracker schema (SQLite)

```
runs(run_id PK, created_at, dataset_version, seed, selected_model, model_version,
     selection_rule, library_versions JSON, duration_seconds)
experiments(experiment_id PK, run_id FK, algorithm, status, error, hyperparameters JSON, seed,
            cv_metrics JSON, test_metrics JSON, training_seconds, dataset_version,
            artifact_path, created_at)
```

Failed candidates are recorded with `status='failed'` and the error string. The UI displays them; they are never hidden.

## Security design

| Control | Implementation |
|---|---|
| Input validation | Pydantic model built from the 30 feature names: `extra="forbid"`, `allow_inf_nan=False`, `ge=0`, all required |
| Upload limits | `.csv` extension only (415 otherwise). Content-Length guard (413) before parsing. Read capped at 1 MB + 1 byte (413). Maximum 2,000 rows (422) |
| Content validation | UTF-8 only, NUL bytes rejected, `csv` module parsing, exact header match (missing/unexpected/duplicate columns → 422), duplicate `sample_id` → 422, per-cell numeric/finite/non-negative checks |
| No temp files | Uploads are parsed from memory; nothing is written to disk |
| CSV injection | Echoed identifiers that start with `= + - @` are prefixed with `'` in outputs |
| Safe model loading | Manifest SHA-256 must match; path must resolve inside the artifacts dir. Tested with a tampered hash and a `../` path |
| Error handling | Standard envelope; generic 500 without stack traces; `debug=False` |
| CORS | Explicit origin allow-list (env `ONCOSENSE_CORS_ORIGINS`), GET/POST only, no credentials |
| Headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` |
| Secrets | None required; none in the code |
| Dependencies | Pinned in `requirements.txt` / `package-lock.json`; `npm install` reported 0 vulnerabilities |

## Frontend design

- Seven routes are lazy-loaded (code-split), with a shared `Layout` (sidebar on desktop, collapsible menu on mobile) and a persistent disclaimer banner.
- The `useApi` hook provides loading, error, data and reload. Every page has loading, error-with-retry and empty states.
- The palette is deep navy, teal, soft blue and purple. Benign is blue and malignant is purple, deliberately **not red**, to avoid an emergency framing.
- Accessibility: skip link, landmarks, `aria-current` via NavLink, labelled form controls, `aria-invalid` with described-by hints, `role=alert` and `role=status`, a focus ring, and reduced-motion support. Chart animations are disabled.
- Metric provenance badges ("Nested CV · training split", "Held-out test set", "Out-of-fold") sit next to every metric group.

## Trade-offs

- **SHAP permutation over KernelExplainer.** Permutation is model-agnostic and exactly additive, and with a 50-row background it takes about 30–40 ms per explanation. KernelExplainer would be slower.
- **Explanations not computed in batch mode.** This keeps batch latency bounded. Use single prediction for explanations.
- **JSON artifacts over a database for results.** They are simple, diffable and need no server. SQLite is used only where history accumulates.
- **No authentication.** The app is a local educational tool. If it is deployed publicly, add authentication and rate limiting at a reverse proxy (see the deployment guide).
