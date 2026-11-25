"""HTTP endpoints. Thin layer: validation + delegation to services."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response

from app.core import config
from app.ml import tracking
from app.ml.data import feature_specs
from app.schemas.prediction import BatchResponse, PredictRequest, PredictResponse, SamplePredictResponse
from app.services import prediction as P
from app.services.registry import registry
from app.services.research import methodology

router = APIRouter(prefix="/api")

_MODEL_SUMMARY_KEYS = ["key", "name", "status", "error", "notes", "tags", "best_params", "param_grid",
                       "training_seconds", "cv_seconds", "model_size_bytes", "inference",
                       "complexity_rank", "experiment_id", "cv_scheme"]


@router.get("/health")
def health():
    return {
        "status": "ok" if registry.loaded else "degraded",
        "model_loaded": registry.loaded,
        "model_version": registry.metadata["model_version"] if registry.loaded else None,
        "load_error": registry.load_error,
        "disclaimer": config.DISCLAIMER,
    }


@router.get("/dataset/summary")
def dataset_summary():
    reg = registry.require()
    md = reg.metadata["dataset"]
    eda = reg.eda or {}
    return {"dataset": md, "integrity": eda.get("integrity"),
            "feature_stats": eda.get("feature_stats"), "scope_note": eda.get("scope_note"),
            "scaling": eda.get("scaling")}


@router.get("/dataset/features")
def dataset_features():
    reg = registry.require()
    p = reg.ood.p
    out = []
    for i, s in enumerate(feature_specs()):
        out.append({**s.__dict__, "train_min": p["min"][i], "train_max": p["max"][i],
                    "train_median": p["median"][i], "input_min": 0.0})
    return {"features": out,
            "missing_value_policy": "All 30 features are required. Missing values are rejected, "
                                    "never imputed.",
            "measurement_note": "These are image-derived measurements from a specific research "
                                "digitisation process. They cannot be obtained without that process."}


@router.get("/dataset/eda")
def dataset_eda():
    reg = registry.require()
    if not reg.eda:
        raise HTTPException(404, "EDA artifact missing. Run `python scripts/run_eda.py`.")
    return {k: reg.eda[k] for k in ["histograms", "boxplots", "correlation", "high_correlation_pairs",
                                    "vif", "pca", "scope_note"]}


@router.get("/dataset/records")
def dataset_records(limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0)):
    reg = registry.require()
    tr = reg.train_records
    n = len(tr["sample_ids"])
    sl = slice(offset, min(offset + limit, n))
    return {"split": "train", "total": n, "offset": offset, "limit": limit,
            "records": [{"sample_id": sid, "label": lab, "features": f}
                        for sid, lab, f in zip(tr["sample_ids"][sl], tr["labels"][sl], tr["features"][sl])]}


@router.get("/models")
def models():
    reg = registry.require()
    sel = reg.metrics["selected_key"]
    return {"selected_key": sel, "model_version": reg.model_version,
            "selection_rationale": reg.metadata["selection_rationale"],
            "selection_rule": reg.metadata["selection_rule"],
            "calibration": reg.metadata["calibration"],
            "threshold": {k: v for k, v in reg.metadata["threshold"].items() if k != "oof_sweep"},
            "training_config": reg.metadata["training_config"],
            "library_versions": reg.metadata["library_versions"],
            "models": [{**{k: m.get(k) for k in _MODEL_SUMMARY_KEYS}, "selected": m["key"] == sel}
                       for m in reg.metrics["models"]]}


@router.get("/models/metrics")
def model_metrics():
    reg = registry.require()
    keep = ["key", "name", "status", "error", "cv", "test", "test_ci95", "classification_report",
            "test_at_operating_threshold", "test_at_operating_threshold_ci95",
            "classification_report_operating"]
    return {"notes": reg.metrics["notes"], "selected_key": reg.metrics["selected_key"],
            "operating_threshold": reg.threshold,
            "models": [{k: m.get(k) for k in keep} for m in reg.metrics["models"]],
            "selected_test_errors": reg.metrics["selected_test_errors"],
            "learning_curve": reg.metrics["learning_curve"]}


@router.get("/models/roc-data")
def roc_data():
    reg = registry.require()
    return {"split": "test", "selected_key": reg.metrics["selected_key"],
            "oof_threshold_sweep": reg.metadata["threshold"]["oof_sweep"],
            "oof_calibration_curve": reg.metadata["calibration"]["oof_calibration_curve"],
            "models": [{"key": m["key"], "name": m["name"], "status": m["status"],
                        "curves": m.get("test_curves"), "threshold_sweep": m.get("test_threshold_sweep")}
                       for m in reg.metrics["models"]]}


@router.get("/models/feature-importance")
def feature_importance():
    reg = registry.require()
    return reg.explainability


@router.get("/experiments")
def experiments(run_id: str | None = None):
    db = registry.dir / "experiments.sqlite"
    return {"runs": tracking.list_runs(db), "experiments": tracking.list_experiments(db, run_id)}


@router.get("/demo/samples")
def demo_samples():
    """Held-out test records for the demonstration explorer. Labels are NOT included."""
    reg = registry.require()
    return {"split": "test", "note": "Ground-truth labels are withheld here; they are returned "
                                     "only with a prediction so the UI can reveal them afterwards.",
            "samples": [{"sample_id": sid, "features": f}
                        for sid, f in zip(reg.demo["sample_ids"], reg.demo["features"])]}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    reg = registry.require()
    return P.predict_features(reg, req.features.model_dump(), explain=req.explain)


@router.post("/predict/sample/{sample_id}", response_model=SamplePredictResponse)
def predict_sample(sample_id: int):
    reg = registry.require()
    i = reg.demo_index.get(sample_id)
    if i is None:
        raise HTTPException(404, f"Sample {sample_id} is not in the held-out demonstration set")
    feats = reg.demo["features"][i]
    res = P.predict_features(reg, feats, explain=True)
    return {**res, "sample_id": sample_id, "features": feats, "ground_truth": reg.demo["labels"][i]}


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: Request, file: UploadFile = File(...),
                        format: str = Query("json", pattern="^(json|csv)$")):
    reg = registry.require()
    name = (file.filename or "").lower()
    if not name.endswith(".csv"):
        raise HTTPException(415, "Only .csv files are accepted")
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_BYTES} bytes")
    try:
        result = P.predict_batch(reg, raw)
    except P.BatchValidationError as exc:
        raise HTTPException(422, {"message": str(exc), **exc.details})
    if format == "csv":
        return Response(P.batch_to_csv(result), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="oncosense_predictions.csv"'})
    return result


@router.get("/predict/template.csv", response_class=PlainTextResponse)
def template():
    reg = registry.require()
    return Response(P.template_csv(reg), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="oncosense_template.csv"'})


@router.get("/research/methodology")
def research():
    return methodology(registry.require())
