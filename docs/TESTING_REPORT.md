# Testing Report

Executed on 2026-11-25 on macOS (Apple Silicon, 16 GB) with Python 3.12.15 and Node 26. **All suites passed.** The commands and actual
outcomes are recorded below.

| Suite | Command | Result |
|---|---|---|
| Backend (pytest) | `cd backend && pytest` | **27 passed** in about 19 s (1 deprecation warning from Starlette's TestClient about `httpx`) |
| Frontend unit/UI (Vitest + Testing Library) | `cd frontend && npm test` | **19 passed** in about 2.7 s. Stable over 12 consecutive runs after one race was fixed (see §5) |
| Type check / lint / build | `npx tsc -b`, `npx oxlint src`, `npm run build` | No errors and no lint warnings; production build succeeds |
| End-to-end (headless Chrome) | `cd frontend && npm run e2e` (both servers running) | **31/31 checks passed** |
| Manual API security checks | `curl` (see §4) | All behaved as expected |

## 1. Backend tests (`backend/tests/`)

The session fixture runs the **real training pipeline** in quick mode (5×1 nested CV) in a temporary artifacts directory. It then points the API registry at that directory, so the API tests exercise freshly trained artifacts and never touch the committed ones.

**Data (`test_data.py`)**

- `test_dataset_shape_and_features`: 569×30, schema order, no NaN or Inf.
- `test_target_orientation_malignant_is_positive`: 212 malignant; y equals `(sklearn_target == 0)`.
- `test_no_target_or_id_leakage_in_features`: no ID or label columns; no feature with |r(label)| ≥ 0.95.
- `test_stratified_disjoint_split`: 455/114, disjoint, covers all rows, class ratios within 1%.
- `test_split_is_reproducible`: the same seed gives the same split, and a different seed gives a different one.

**ML (`test_ml.py`)**

- `test_metrics_hand_computed`: TP/FN/FP/TN, recall, specificity, precision and accuracy against a hand-worked example.
- `test_malignant_orientation_in_metrics`: a high score on malignant gives recall 1 and AUC 1.
- `test_scaler_fit_only_on_training_data`: the pipeline scaler mean equals the training mean and not the full-data mean.
- `test_training_run_outputs`: all 6 mandatory models plus the optional one complete; the CV fold count is correct; artifacts exist; the feature schema and class mapping are saved.
- `test_selection_does_not_depend_on_test_set`: selection re-run with every test field removed gives the same model.
- `test_artifact_loads_and_predicts_finite`: the hash-verified load works, probabilities lie in [0, 1] and are finite, the feature names match, and malignant scores exceed benign scores on average.
- `test_tampered_artifact_is_refused`: a wrong SHA-256 and a `../` path both raise `ArtifactIntegrityError`.
- `test_ood_checker_flags_extremes`: the training median is `ok`, mean_area = 50,000 gives `abstain`, and at most 2 held-out samples are abstained on.

**API (`test_api.py`)**

- `test_health`, `test_dataset_endpoints`, `test_model_endpoints`: shapes and contents, exactly one selected model, ROC data, 30 SHAP features, citations, tracker runs.
- `test_valid_prediction`: the disclaimer is present, the class is valid, the probability lies in [0, 1], and **SHAP additivity** holds (base + Σ = probability, within 1e-6).
- `test_sample_prediction_and_404`.
- `test_missing_and_unexpected_fields`: a missing feature gives 422; an extra `patient_age` gives 422.
- `test_invalid_values`: −1, `"abc"`, `null` and `Infinity` each give 422.
- `test_extreme_input_abstains`: `abstained=true` with `prediction=null`.
- `test_batch_inference`: a mix of valid, non-numeric and missing rows gives the correct summary; the CSV export contains the disclaimer.
- `test_batch_schema_errors`: unexpected column, missing column and duplicate IDs each give 422 with details.
- `test_malformed_and_wrong_files`: binary content gives 422, an empty file 422, and `.exe` 415.
- `test_oversized_csv`: 1.1 MB gives 413.
- `test_template_round_trip`: the template CSV is accepted with 0 invalid rows.

**Integration (`test_integration.py`)**

- `test_end_to_end_workflow`: data → training → selection → artifacts → API load (the version matches the training output) → predictions on held-out samples with explanations → the API serves exactly the metrics that training produced.

## 2. Frontend tests (`frontend/src/test/`)

The fixtures are **real API responses** captured from the running backend (`src/test/fixtures/*.json`), so tests never use invented metrics.

- Page rendering: each of the 7 routes renders its heading and the full disclaimer text (7 tests).
- The landing page shows a loading state, then the selected model name taken from the API.
- The model page renders the comparison table from the metrics API, with provenance badges.
- A failed experiment is displayed as "Experiment failed: …" and is not hidden.
- Error states: a network failure gives a meaningful message and a Retry button; a 503 shows the backend message.
- Prediction lab: the ground truth stays hidden until "Reveal" is clicked; the result includes the inline disclaimer and an explanation.
- Manual form: an empty submit blocks the API call and shows "30 field(s) need attention". A negative value is flagged with `aria-invalid`, and a valid form calls `/api/predict`.
- CSV: a `.txt` file is rejected client-side; batch results and row errors render; the download button is present. Server schema errors (unexpected columns) are displayed.
- Responsive navigation: the mobile menu toggles `aria-expanded`.
- `validateFeatureInputs` unit tests: missing, non-numeric, negative and non-finite values are blocking errors; out-of-range values are non-blocking warnings.

## 3. End-to-end checks (`frontend/e2e/smoke.mjs`, real backend + real browser)

- All 7 pages at 1440×900 and 390×844: the disclaimer is present, there are no error alerts, and charts render (Dataset: 7 Recharts surfaces, Models: 16, Explain: 3).
- On mobile, there is no horizontal page scroll on any page.
- Sample explorer: the label is hidden, then revealed; the explanation renders.
- Manual: an empty submit is blocked; training medians produce a prediction; mean area = 90,000 makes the model **abstain**.
- CSV: the template plus a corrupted row gives results with one invalid row; the CSV download works; a `patient_age` column is rejected with details.
- The CV AUC shown on the Models page matches `/api/models/metrics`, which shows that the charts use real data.
- There are no browser console errors or warnings and no 5xx responses. The only console entry is the expected 422 from the deliberate bad-CSV upload, which is whitelisted.

Screenshots are written to `frontend/e2e/screenshots/` (git-ignored).

## 4. Manual security checks against the running API

| Check | Result |
|---|---|
| CORS from `http://localhost:5180` | `access-control-allow-origin` echoed |
| CORS from `http://evil.example` | No allow-origin header |
| Security headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` present |
| 1.2 MB multipart upload | 413 before parsing |
| Unknown demo sample | 404 in the standard error envelope |
| Backend log after the full e2e run | No errors or tracebacks |

## 5. Defects found by testing and fixed

1. **Artifact path bug.** `save_model` computed paths relative to the default artifacts dir, so training into another directory failed. The pytest fixture caught this, and it was fixed by passing the base dir.
2. **scikit-learn 1.9 deprecation.** `SVC(probability=True)` is deprecated. It was replaced by `CalibratedClassifierCV(SVC(), ensemble=False)`, and training now runs warning-free.
3. **Mobile overflow.** Wide tables inside grid items stretched the Dataset page (228 px) and the Models page (31 px) at 390 px width. The e2e check caught this, and it was fixed with `min-w-0` on cards and on the threshold-explorer select.
4. **Garbled chart ticks.** Error bars above 1.0 produced unreadable axis labels. Fixed with clamped domains and tick formatters.
5. **Flaky UI test.** A page heading renders during loading and again after data arrives, so a node found early could be detached. The test now re-queries inside `waitFor`. Verified with 12/12 clean runs.
6. **Dev port collision.** Another local project already used Vite's default port 5173. The frontend is now pinned to 5180 with `strictPort`.

## 6. Not covered

- Cross-browser testing beyond Chrome.
- Load or performance testing beyond single-request latency (about 40 ms per prediction with explanation).
- Automated accessibility auditing (axe/Lighthouse). Accessibility was implemented and checked manually with ARIA roles and labels, which the tests also exercise.
