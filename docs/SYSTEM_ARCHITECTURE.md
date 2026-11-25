# System Architecture

## 1. Overall architecture

```mermaid
flowchart LR
    subgraph Offline["Offline (developer laptop)"]
        S1[scripts/run_eda.py] --> A[(artifacts/)]
        S2[scripts/train_models.py] --> A
        S3[scripts/build_eda_notebook.py] --> NB[notebooks/EDA.ipynb]
    end
    subgraph Backend["FastAPI backend :8000"]
        R[api/routes.py] --> SV[services/<br/>registry · prediction · research]
        SV --> ML[ml/<br/>ood · explain · metrics · tracking]
        REG[Registry<br/>loaded once at startup] --> SV
    end
    A -- sha256-verified load --> REG
    A -- /static/figures --> R
    subgraph Frontend["React + Vite :5180"]
        P[7 pages] --> C[lib/api.ts typed client]
    end
    C -- "/api/* (Vite proxy)" --> R
    U((User / browser)) --> P
```

Key properties:

- **Training never runs in a request.** Artifacts are created offline, and the API only loads them.
- **ML code is separated from HTTP code.** `app/ml` has no FastAPI imports, and `app/api` contains no modelling logic.
- **Single source of truth.** Every number in the UI comes from artifacts written by the training and EDA scripts.

## 2. Inference workflow

```mermaid
sequenceDiagram
    participant UI as Prediction Lab
    participant API as POST /api/predict
    participant V as Pydantic schema
    participant O as OOD checker
    participant M as Pipeline (SVM)
    participant X as SHAP explainer
    UI->>API: {features: {30 values}}
    API->>V: validate (30 required, finite, >= 0, no extras)
    alt invalid
        V-->>UI: 422 validation_error (no prediction)
    end
    API->>O: check(x) vs training stats
    alt abstain (far outside training data)
        O-->>UI: abstained=true, reasons, no class/probability
    else ok / warning
        API->>M: predict_proba -> P(malignant)
        API->>API: class = P >= operating threshold (0.39)
        API->>X: permutation SHAP on P(malignant)
        API-->>UI: class, probability, warnings, explanation, disclaimer, model_version
    end
```

## 3. Frontend–backend communication

```mermaid
flowchart TB
    subgraph Browser
        Pg[Page component] --> H[useApi hook<br/>loading / error / data / reload]
        H --> Cl[api.ts request&lt;T&gt;]
        Cl -->|ApiError status, code, details| H
    end
    Cl -->|fetch /api/...| VP[Vite dev proxy :5180]
    VP --> F[FastAPI :8000]
    F -->|"200 JSON or {error:{code,message,details}}"| Cl
    Img[img src=/static/figures/*.png] --> VP
```

- All errors use the same envelope: `{"error": {"code", "message", "details"}}`. The client turns it into `ApiError`, and pages render `ErrorState` with a Retry button.
- In production, set `VITE_API_BASE` or serve the frontend behind the same origin (see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)).

## 4. Model evaluation pipeline

See [MODEL_EVALUATION.md](MODEL_EVALUATION.md) §0 and [ML_METHODOLOGY.md](ML_METHODOLOGY.md) for the training flow.

```mermaid
flowchart LR
    Tr[Train split] --> NCV[Nested CV per model] --> Sel[Selection rule]
    Tr --> OOF[OOF predictions] --> Cal[Calibration decision] --> Thr[Threshold]
    Sel --> Fin[Final pipeline]
    Cal --> Fin
    Te[Test split] --> Ev[Evaluate once]
    Fin --> Ev --> Out[metrics.json · figures · tracker]
```

## 5. Components

| Module | Responsibility |
|---|---|
| `app/ml/data.py` | Load, re-encode, validate schema, fingerprint, split, feature specs |
| `app/ml/models.py` | Candidate pipelines and grids |
| `app/ml/training.py` | Full workflow: nested CV, selection, calibration, threshold, test evaluation, explainability, persistence, tracking |
| `app/ml/metrics.py` | Metrics with malignant = positive, bootstrap CIs, curves, threshold sweep |
| `app/ml/explain.py` | SHAP local explainer, permutation importance, native importance |
| `app/ml/ood.py` | Distribution-shift heuristics (JSON-serialisable) |
| `app/ml/artifacts.py` | JSON I/O, model save, hash-verified trusted load |
| `app/ml/tracking.py` | SQLite experiment tracker |
| `app/ml/eda.py`, `figures.py` | EDA statistics, static figures |
| `app/services/registry.py` | Loads all artifacts at startup; reports a degraded state if missing |
| `app/services/prediction.py` | Single and batch inference, CSV parsing and validation, CSV export |
| `app/services/research.py` | Methodology content with run-specific values injected |
| `app/api/routes.py` | Endpoints |
| `app/main.py` | App factory, CORS, security headers, body-size guard, error handlers, static figures |
