# OncoSense AI

> **Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.**

OncoSense AI is an end-to-end, explainable machine-learning platform that classifies breast-mass samples from the
public **Breast Cancer Wisconsin (Diagnostic)** dataset as *benign* or *malignant* **dataset classes**. It demonstrates
leakage-aware ML methodology, transparent evaluation, explainability, and professional full-stack engineering, all
free and offline-capable after setup.

It is **not** a medical device, a screening service or a diagnostic tool. Its outputs are dataset-specific model
predictions. They are not a person's probability of having cancer. Do not enter real patient data.

![Overview page](docs/screenshots/01-home.png)

## Contents

1. [Tech stack](#tech-stack)
2. [Results](#results-run-20261125t114504z-d23509-seed-42)
3. [Installation and setup](#installation-and-setup)
4. [How to use the app (step-by-step with screenshots)](#how-to-use-the-app)
5. [Using the REST API directly](#using-the-rest-api-directly)
6. [Running the tests](#running-the-tests)
7. [Retraining and configuration](#retraining-and-configuration)
8. [Troubleshooting](#troubleshooting)
9. [Repository layout](#repository-layout)
10. [Documentation](#documentation)
11. [Limitations](#limitations)
12. [Data attribution](#data-attribution)

## Tech stack

| Layer | Stack |
|---|---|
| ML | Python 3.12, scikit-learn 1.9.1, SHAP 0.54.1, pandas, NumPy, Matplotlib |
| API | FastAPI 0.165, Pydantic v2, Uvicorn |
| Frontend | React 19, TypeScript, Vite 8, Tailwind CSS 4, React Router 7, Recharts 3, Lucide |
| Storage | JSON artifacts, hash-verified joblib model, SQLite experiment tracker |
| Tests | pytest (27), Vitest + Testing Library (19), headless-Chrome e2e smoke (31 checks) |

## Results (run `20261125T114504Z-d23509`, seed 42)

All numbers come from `backend/artifacts/metrics.json`, produced by `scripts/train_models.py`.
Positive class = **malignant**. Split: 455 training / 114 held-out test samples (stratified).

**Nested cross-validation on the training split.** These were used for model selection. Values are mean ± SD over 10 outer folds at threshold 0.5.

| Model | CV ROC-AUC | CV recall | CV specificity | CV Brier |
|---|---|---|---|---|
| Logistic Regression | 0.9950 ± 0.0059 | 0.950 ± 0.034 | 0.991 | 0.0228 |
| **SVM (RBF), selected** | 0.9948 ± 0.0063 | 0.950 ± 0.039 | 0.982 | **0.0214** |
| Gradient Boosting (optional) | 0.9924 ± 0.0066 | 0.941 ± 0.044 | 0.981 | 0.0270 |
| Gaussian Naive Bayes | 0.9882 ± 0.0074 | 0.897 ± 0.081 | 0.963 | 0.0565 |
| Random Forest | 0.9878 ± 0.0095 | 0.929 ± 0.058 | 0.970 | 0.0349 |
| K-Nearest Neighbors | 0.9870 ± 0.0135 | 0.900 ± 0.070 | 0.995 | 0.0335 |
| Decision Tree | 0.9534 ± 0.0332 | 0.897 ± 0.061 | 0.949 | 0.0571 |

**Why the SVM was selected.** Logistic Regression had the highest mean CV ROC-AUC. The SVM was within one standard error of it, so the two were treated as tied. The tie was broken by the lower CV Brier score, which measures probability quality. The test set played no part in this choice.

**Selected model on the untouched test set.** The operating threshold of 0.39 was chosen from out-of-fold training predictions. The 95% bootstrap CIs are in brackets.

| Accuracy | Recall (sensitivity) | Specificity | Precision | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| 0.982 [0.956, 1.0] | 0.952 [0.875, 1.0] | 1.000 [1.0, 1.0] | 1.000 | 0.976 | 0.996 [0.985, 1.0] | 0.018 |

Confusion matrix: TN 72, FP 0, **FN 2**, TP 40. Two malignant test samples were missed (IDs 99 and 73).
With 114 test samples, one error moves recall by about 2.4 points. These results do **not** establish clinical
reliability. See [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md).

## Installation and setup

### Prerequisites

| Tool | Version | Check with |
|---|---|---|
| Python | 3.11 or 3.12 | `python3 --version` |
| Node.js + npm | Node 20 or later | `node --version` |
| Git | any | `git --version` |
| Google Chrome | optional, only for `npm run e2e` / `npm run screenshots` | — |

A 4–8 GB RAM laptop is enough. No GPU, paid API, cloud account or API key is needed. The dataset ships with
scikit-learn, so after installing dependencies everything works **offline**.

### Step 1: Clone the repository

```bash
git clone https://github.com/roshan-ds-tech/cancer_detection.git
cd cancer_detection
```

### Step 2: Set up the Python backend

macOS / Linux:

```bash
python3.12 -m venv .venv                 # or python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### Step 3: Generate the EDA and train the models

The repository already contains a trained model and its artifacts in `backend/artifacts/`, so this step is optional.
Run it to reproduce the results from scratch:

```bash
cd backend
python scripts/run_eda.py              # EDA statistics + figures -> artifacts/eda.json, artifacts/figures/
python scripts/train_models.py         # nested CV for 7 models, selection, test evaluation (~45 s)
python scripts/build_eda_notebook.py   # optional: regenerates and executes notebooks/EDA.ipynb
```

`train_models.py` prints each model's CV metrics, the selection rationale, and the one-time held-out test results.
Options: `--quick` (faster 5×1 CV, for smoke tests), `--seed 42`, `--no-figures`.

### Step 4: Start the API (terminal 1)

```bash
cd backend                              # with the virtual environment activated
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check it is healthy: open <http://localhost:8000/api/health>. You should see `"status": "ok"` and `"model_loaded": true`.
Interactive API docs are at <http://localhost:8000/docs>.

### Step 5: Start the frontend (terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5180>**. The Vite dev server proxies `/api` and `/static` to the API on port 8000.
The sidebar should show a green **API connected** badge with the model version.

## How to use the app

All pages are reachable from the left sidebar (or the menu button on mobile). The research-only disclaimer is
shown at the top of every page and next to every prediction.

### 1. Overview

The landing page summarises the project goal, the methodology, the live headline results of the selected model
(read from the API, never hard-coded) and the research-only boundaries. Use the buttons at the top to jump to the Model Comparison Lab or the Prediction Lab.

![Overview](docs/screenshots/01-home.png)

### 2. Dataset Explorer: understand the data

1. Open **Dataset Explorer**. The top cards show integrity checks (569 rows, 30 features, no missing values, no
   duplicates), the class balance and a PCA projection of the training split.
2. Scroll to **Feature distributions by class** and choose a feature from the dropdown to compare benign and malignant
   histograms and box plots.
3. Scroll to **Feature correlations** to see the 30×30 correlation heatmap and the highly correlated pairs (|r| ≥ 0.9).
   **Multicollinearity** shows the variance inflation factors.
4. Further down are a per-feature statistics table, paginated sample records and the static EDA figures.

![Dataset overview](docs/screenshots/02-dataset-overview.png)
![Feature distributions](docs/screenshots/03-dataset-distributions.png)
![Feature correlations](docs/screenshots/04-dataset-correlations.png)

### 3. Model Performance & Comparison Lab: compare the models

1. Open **Model Performance**. The first card explains *which* model was selected and *why*, with the calibration
   decision and the operating threshold.
2. **Algorithm comparison** lists all 7 models with CV metrics (mean ± SD), test metrics, fit and inference time,
   model size and best hyperparameters. Badges show where each number came from
   ("Nested CV · training split", "Held-out test set", "Out-of-fold").
3. **Curves** has ROC, precision–recall and calibration tabs. Tick or untick models to compare them.
4. **Threshold explorer**: drag the slider to see how recall, specificity and precision trade off, and how the
   confusion matrix changes. Switch the data source between out-of-fold training predictions and the test set.
5. Below that are per-model confusion matrices, a CV ROC-AUC box plot, inference latency, the learning curve and an
   error analysis listing the misclassified test samples.

![Selected model](docs/screenshots/05-models-selected.png)
![Algorithm comparison](docs/screenshots/06-models-comparison.png)
![ROC / PR / calibration curves](docs/screenshots/07-models-curves.png)
![Threshold explorer](docs/screenshots/08-models-threshold.png)
![Confusion matrices](docs/screenshots/09-models-confusion.png)

### 4. Prediction Lab: run the model

The Prediction Lab has three tabs.

#### a) Held-out sample explorer

1. Pick a sample from **Sample ID** (or press **Random**). These are the 114 untouched test records. Their labels are
   **not** shown in the list.
2. Press **Run model**. The result card shows the predicted dataset class, the model probability for the
   "malignant" class, the operating threshold, input validation and a distribution check.
3. Press **Reveal known dataset label** to compare the prediction with the dataset's label.
4. **Why the model produced this output** shows a SHAP explanation: purple bars push towards malignant,
   blue bars towards benign. The baseline plus the sum of the contributions equals the model output.

![Choose a sample](docs/screenshots/10-predict-choose-sample.png)
![Sample prediction](docs/screenshots/11-predict-sample-result.png)
![Label revealed and explanation](docs/screenshots/12-predict-sample-explanation.png)

#### b) Manual input

1. Switch to **Manual input**. The form has all 30 features grouped as mean values, standard errors and "worst"
   values. Each field shows its unit and the training range.
2. Use **Fill with training medians** or **Fill with a random held-out sample** to start from realistic values,
   or **Clear** to type your own.
3. Press **Run model**. Missing, non-numeric or negative values block the request and are highlighted.
   Values outside the training range are allowed but flagged.
4. If the input is far outside the training data (for example, `mean area = 90000`), the model **abstains**:
   it shows no class or probability, and lists the reasons instead.

![Manual form](docs/screenshots/13-predict-manual-form.png)
![Manual prediction](docs/screenshots/14-predict-manual-result.png)
![Model abstains on extreme input](docs/screenshots/15-predict-abstain.png)

#### c) CSV batch

1. Switch to **CSV batch** and press **Template CSV** to download a file with the correct header and 3 example rows.
2. Fill in rows. The header must contain exactly the 30 snake_case feature columns, plus an optional `sample_id`.
   Column order does not matter. Limits: 1 MB and 2,000 rows.
3. Choose the file and press **Upload & predict**. Each row is marked `predicted`, `warning` (unusual input),
   `abstained` or `invalid` (with the reason). Nothing is imputed.
4. Press **Download CSV** to save the results. The export includes the disclaimer and the model version.
5. A file with wrong columns (for example, a `patient_age` column) is rejected as a whole, with the offending
   columns listed. Uploaded files are processed in memory and never stored.

![CSV upload](docs/screenshots/16-predict-csv-upload.png)
![CSV batch results](docs/screenshots/17-predict-csv-results.png)
![CSV schema error](docs/screenshots/18-predict-csv-schema-error.png)

### 5. Explainable AI: see what drives the model

1. **Global SHAP importance** ranks features by mean |SHAP| for the selected model on the test set.
2. **Permutation importance** shows the drop in test ROC-AUC when a feature is shuffled. Use the dropdown to pick
   any of the 7 models.
3. **Sample-level explanation**: choose a held-out sample to see all 30 SHAP contributions, computed live by the API.
4. **Model-native importances** shows logistic-regression coefficients and tree-based impurity importances, followed by
   a glossary of the most influential features.

Attributions describe the model's behaviour, not biology. Correlated features (radius, perimeter, area) can
share credit unpredictably.

![Global explanations](docs/screenshots/19-explain-global.png)
![Sample-level explanation](docs/screenshots/20-explain-local.png)

### 6. Experiments: training history

Every training run is logged to `backend/artifacts/experiments.sqlite`. This page lists the runs (dataset version,
seed, selected model, library versions, duration), lets you **compare two runs** side by side, and shows every logged
experiment, including failed ones.

![Experiments](docs/screenshots/21-experiments.png)

### 7. Research & Methods

This page covers dataset provenance, preprocessing and leakage control, the algorithms, the evaluation design,
explainability, the regulatory context, limitations, references and the exact commands to reproduce the run.

![Research & Methods](docs/screenshots/22-research.png)

### Mobile

The layout is responsive. On small screens the sidebar collapses into a menu button.

<img src="docs/screenshots/24-mobile-predict.png" alt="Mobile view" width="300">

## Using the REST API directly

Full reference: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md). Interactive docs are at <http://localhost:8000/docs>.

![API docs](docs/screenshots/23-api-docs.png)

```bash
# health and model version
curl http://localhost:8000/api/health

# predict a held-out test sample (the response includes the SHAP explanation)
curl -X POST http://localhost:8000/api/predict/sample/120

# download the CSV template, then run batch inference on it
curl -o template.csv http://localhost:8000/api/predict/template.csv
curl -F "file=@template.csv" http://localhost:8000/api/predict/batch
curl -F "file=@template.csv" "http://localhost:8000/api/predict/batch?format=csv" -o predictions.csv
```

A single prediction (`POST /api/predict`) takes a JSON body with all 30 features:

```json
{"features": {"mean_radius": 11.41, "mean_texture": 10.82, "...": "...", "worst_fractal_dimension": 0.08523}, "explain": true}
```

Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model loaded, model version |
| GET | `/api/dataset/summary`, `/api/dataset/features`, `/api/dataset/eda`, `/api/dataset/records` | Dataset information |
| GET | `/api/models`, `/api/models/metrics`, `/api/models/roc-data`, `/api/models/feature-importance` | Models and evaluation |
| GET | `/api/experiments` | Experiment tracker |
| GET | `/api/demo/samples` | Held-out samples (without labels) |
| POST | `/api/predict`, `/api/predict/sample/{id}`, `/api/predict/batch` | Predictions |
| GET | `/api/predict/template.csv` | CSV template |
| GET | `/api/research/methodology` | Methodology and citations |

## Running the tests

```bash
# backend: data, ML, API and integration tests (trains a quick model in a temporary directory)
cd backend && pytest

# frontend: page rendering, validation, error states, CSV upload, disclaimer
cd frontend && npm test

# type check, lint and production build
cd frontend && npx tsc -b && npx oxlint src && npm run build

# end-to-end: headless Chrome against the running app (needs both servers running + Google Chrome)
cd frontend && npm run e2e

# regenerate the README screenshots (same requirements as e2e)
cd frontend && npm run screenshots
```

Latest results: 27 backend tests passed, 19 frontend tests passed, 31/31 e2e checks passed.
Details are in [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md).

## Retraining and configuration

Running `python scripts/train_models.py` again writes a new versioned model (`oncosense-<model>-<run_id>.joblib`),
updates the manifest and metadata, and adds a run to the experiment tracker. **Restart Uvicorn** to load the new model.
The API only loads a model whose SHA-256 hash matches `model_manifest.json`.

| Environment variable | Default | Purpose |
|---|---|---|
| `ONCOSENSE_ARTIFACTS_DIR` | `backend/artifacts` | Where trusted artifacts are read from |
| `ONCOSENSE_CORS_ORIGINS` | `http://localhost:5180,http://127.0.0.1:5180` | Allowed browser origins |
| `ONCOSENSE_MAX_UPLOAD_BYTES` | `1000000` | CSV size limit |
| `ONCOSENSE_MAX_BATCH_ROWS` | `2000` | CSV row limit |
| `VITE_API_PROXY` (dev) | `http://127.0.0.1:8000` | Where the Vite dev server forwards `/api` |
| `VITE_API_BASE` (build) | `''` (same origin) | API base URL for a production build |

For a production-style build and hosting notes, see [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar shows **API unreachable** (or **Model not loaded**), or pages show a network error | Start the API (step 4) and check <http://localhost:8000/api/health>. |
| `/api/health` returns `"degraded"` with a `load_error` | The artifacts are missing or were modified. Run `python scripts/train_models.py` in `backend/`, then restart Uvicorn. |
| `npm run dev` fails with **Port 5180 is already in use** | Stop the other process on that port, or change `server.port` in `frontend/vite.config.ts` and add the new origin to `ONCOSENSE_CORS_ORIGINS`. |
| `ModuleNotFoundError` when running scripts | Activate the virtual environment (`source .venv/bin/activate`) and run the commands from `backend/`. |
| `npm run e2e` cannot launch the browser | It uses your installed Google Chrome via `playwright-core`. Install Chrome, or skip the e2e step. |
| CSV upload rejected | Use the template header exactly (30 snake_case columns, plus optional `sample_id`), UTF-8 encoding, under 1 MB and 2,000 rows. |

## Repository layout

```
backend/
  app/
    main.py            FastAPI app: CORS, security headers, size guard, error envelope, lifespan model load
    api/routes.py      HTTP endpoints (thin)
    core/config.py     paths, limits, CORS, disclaimer
    schemas/           Pydantic request/response models (strict 30-feature schema)
    services/          registry (trusted artifact loading), prediction/batch, research content
    ml/                data, models, metrics, training, eda, explain, ood, artifacts, tracking, figures
  scripts/             run_eda.py, train_models.py, build_eda_notebook.py
  notebooks/EDA.ipynb  executed EDA notebook
  artifacts/           model + metadata + metrics + figures + experiments.sqlite (generated)
  tests/               pytest suite
frontend/
  src/pages/           7 pages
  src/components/      layout, disclaimer, charts, prediction result, UI primitives
  src/lib/             typed API client, hooks, formatting, validation
  src/test/            Vitest suites + fixtures captured from the real API
  e2e/                 smoke.mjs (headless Chrome checks), readme_screenshots.mjs
docs/                  research, methodology, evaluation, architecture, API, ethics, testing, deployment
docs/screenshots/      screenshots used in this README
```

## Documentation

| Topic | File |
|---|---|
| Research summary | [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) |
| Dataset analysis (EDA findings) | [docs/DATASET_ANALYSIS.md](docs/DATASET_ANALYSIS.md) |
| Related work | [docs/RELATED_WORK.md](docs/RELATED_WORK.md) |
| Study methodology (TRIPOD+AI-oriented) | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| ML implementation methodology | [docs/ML_METHODOLOGY.md](docs/ML_METHODOLOGY.md) |
| Model evaluation | [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) |
| Ethics and limitations | [docs/ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md) (also [LIMITATIONS_AND_ETHICS](docs/LIMITATIONS_AND_ETHICS.md)) |
| Architecture / system design | [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md), [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) |
| API reference | [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) |
| Testing report | [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md) |
| Deployment | [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) |

## Limitations

- 569 samples from a single institution, collected before 1995. There is no external validation.
- No demographic data, so fairness across groups cannot be assessed.
- Inputs are image-derived measurements from one specific digitisation process. Values from other processes may not be comparable.
- The held-out test set has only 114 samples, so the confidence intervals are wide.
- The out-of-distribution check is a heuristic, not a validated detector.

See [docs/ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md).

## Data attribution

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Licensed under **CC BY 4.0**.
