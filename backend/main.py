"""FastAPI server for the Breast Cancer Prediction project.

Run with:  uvicorn main:app --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import model as ml

app = FastAPI(title="Breast Cancer Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the best model once when the server starts (it is trained the first time).
best_model, results = ml.load_model()


class PatientValues(BaseModel):
    """The 10 measurements entered in the form. All must be numbers >= 0."""
    radius: float = Field(ge=0)
    texture: float = Field(ge=0)
    perimeter: float = Field(ge=0)
    area: float = Field(ge=0)
    smoothness: float = Field(ge=0)
    compactness: float = Field(ge=0)
    concavity: float = Field(ge=0)
    concave_points: float = Field(ge=0)
    symmetry: float = Field(ge=0)
    fractal_dimension: float = Field(ge=0)


@app.get("/api/dataset")
def get_dataset():
    return ml.dataset_summary()


@app.get("/api/results")
def get_results():
    return results


@app.post("/api/train")
def train():
    """Run all models on the dataset again and use the new best model."""
    global best_model, results
    ml.train_models()
    best_model, results = ml.load_model()
    return results


@app.post("/api/predict")
def predict(values: PatientValues):
    output = ml.predict(best_model, values.model_dump())
    output["model"] = results["best_model"]
    return output
