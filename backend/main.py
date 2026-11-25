"""FastAPI server for the Breast Cancer Prediction project.

Run with:  uvicorn main:app --port 8000      (API docs at http://localhost:8000/docs)
"""
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import data
import predictor
import training

app = FastAPI(title="OncoSense AI – Breast Cancer Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the best model once when the server starts (it is trained automatically the first time).
state = {}
state["model"], state["results"] = training.load_model()


class Measurements(BaseModel):
    """The 10 cell measurements. Every value must be a number >= 0."""
    radius: float = Field(ge=0, allow_inf_nan=False)
    texture: float = Field(ge=0, allow_inf_nan=False)
    perimeter: float = Field(ge=0, allow_inf_nan=False)
    area: float = Field(ge=0, allow_inf_nan=False)
    smoothness: float = Field(ge=0, allow_inf_nan=False)
    compactness: float = Field(ge=0, allow_inf_nan=False)
    concavity: float = Field(ge=0, allow_inf_nan=False)
    concave_points: float = Field(ge=0, allow_inf_nan=False)
    symmetry: float = Field(ge=0, allow_inf_nan=False)
    fractal_dimension: float = Field(ge=0, allow_inf_nan=False)


@app.get("/api/health")
def health():
    return {"status": "ok", "best_model": state["results"]["best_model"],
            "trained_at": state["results"]["trained_at"]}


# ---------- dataset ----------
@app.get("/api/dataset/summary")
def dataset_summary():
    return data.summary()


@app.get("/api/dataset/eda")
def dataset_eda():
    return data.eda()


@app.get("/api/dataset/records")
def dataset_records(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return data.records(limit, offset)


# ---------- models ----------
@app.get("/api/models/results")
def model_results():
    return state["results"]


@app.post("/api/models/train")
def run_training():
    """Train and compare all models again, then use the new best model."""
    training.train_models()
    state["model"], state["results"] = training.load_model()
    return state["results"]


@app.get("/api/models/history")
def model_history():
    return training.load_history()


# ---------- predictions ----------
@app.post("/api/predict")
def predict(values: Measurements):
    return {**predictor.predict_one(state["model"], values.model_dump()),
            "model": state["results"]["best_model"]}


@app.get("/api/samples")
def samples():
    return predictor.test_samples()


@app.post("/api/predict/sample/{sample_id}")
def predict_sample(sample_id: int):
    result = predictor.predict_test_sample(state["model"], sample_id)
    if result is None:
        raise HTTPException(404, "This sample is not in the test set.")
    return {**result, "model": state["results"]["best_model"]}


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Please upload a .csv file.")
    content = await file.read(predictor.MAX_CSV_BYTES + 1)
    try:
        result = predictor.predict_csv(state["model"], content)
    except predictor.CSVError as e:
        raise HTTPException(422, str(e))
    return {**result, "model": state["results"]["best_model"]}


@app.get("/api/predict/template.csv", response_class=PlainTextResponse)
def csv_template():
    return PlainTextResponse(predictor.template_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=template.csv"})
