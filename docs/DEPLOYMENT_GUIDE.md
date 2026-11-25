# Deployment Guide

> OncoSense AI is an educational prototype. Deploy it only as a teaching or demonstration tool, never in a clinical workflow.

## 1. Local development (recommended)

Requirements: Python 3.11/3.12, Node.js ≥ 20, about 1.5 GB of disk for dependencies. No GPU. Internet is needed only for installing dependencies.

```bash
# from the repository root
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cd backend
python scripts/run_eda.py              # -> artifacts/eda.json, figures
python scripts/train_models.py         # -> model, metrics, explainability, tracker (~45 s)
python scripts/build_eda_notebook.py   # optional: executes notebooks/EDA.ipynb
uvicorn app.main:app --host 127.0.0.1 --port 8000

# second terminal
cd frontend
npm install
npm run dev                            # http://localhost:5180
```

Vite proxies `/api` and `/static` to `http://127.0.0.1:8000`. Override the target with `VITE_API_PROXY`.
The port is fixed at 5180 (`strictPort`). If 5180 is busy, change it in `vite.config.ts` and in `ONCOSENSE_CORS_ORIGINS`.

After the scikit-learn dataset is cached and dependencies are installed, everything runs **offline**.

## 2. Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ONCOSENSE_ARTIFACTS_DIR` | `backend/artifacts` | Where trusted artifacts are read from |
| `ONCOSENSE_CORS_ORIGINS` | `http://localhost:5180,http://127.0.0.1:5180` | Comma-separated allowed origins |
| `ONCOSENSE_MAX_UPLOAD_BYTES` | `1000000` | CSV size limit |
| `ONCOSENSE_MAX_BATCH_ROWS` | `2000` | CSV row limit |
| `VITE_API_BASE` (build time) | `''` (same origin) | API base URL for the built frontend |
| `VITE_API_PROXY` (dev) | `http://127.0.0.1:8000` | Dev proxy target |

## 3. Production-style build (single host)

```bash
cd frontend && npm run build           # -> frontend/dist (static files)
cd ../backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2
```

Serve `frontend/dist` with any static server or reverse proxy, and forward `/api` and `/static` to Uvicorn on the same origin.
Example nginx snippet:

```nginx
server {
  listen 80;
  root /srv/oncosense/frontend/dist;
  location / { try_files $uri /index.html; }
  location /api/    { proxy_pass http://127.0.0.1:8000; client_max_body_size 1100k; }
  location /static/ { proxy_pass http://127.0.0.1:8000; }
}
```

If the app is exposed beyond localhost, add authentication and rate limiting at the proxy, keep the disclaimer visible, and do not
accept real patient data.

## 4. Free hosting options

The app needs no paid services. Possible free tiers include any static host for `dist/` combined with a free Python host
for FastAPI. Free tiers and their limits change, so check them before relying on one. Ship the `artifacts/` directory
with the backend, or run the training script during the build. Keep `model_manifest.json` and the `.joblib` together,
otherwise the integrity check will (correctly) refuse to load.

## 5. Retraining and versioning

`python scripts/train_models.py` writes a new versioned model (`oncosense-<model>-<run_id>.joblib`), updates the manifest and
metadata, and appends to `experiments.sqlite`. Restart Uvicorn to load it. Older model files can be deleted. The
tracker keeps their metrics.

## 6. Verification after deployment

```bash
curl http://localhost:8000/api/health          # status "ok", model_loaded true
cd backend && pytest                            # 27 tests
cd frontend && npm test && npm run e2e          # 19 unit/UI tests, 31 e2e checks (needs Chrome + running servers)
```
