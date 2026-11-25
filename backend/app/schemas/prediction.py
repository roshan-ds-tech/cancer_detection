"""Pydantic request/response schemas for inference."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.ml.data import FEATURE_NAMES

# Strict 30-field schema: every feature required, finite, non-negative; extra keys rejected.
FeatureVector = create_model(
    "FeatureVector",
    __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    **{name: (float, Field(..., ge=0)) for name in FEATURE_NAMES},
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: FeatureVector  # type: ignore[valid-type]
    explain: bool = True


class Contribution(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class Explanation(BaseModel):
    method: str
    output_space: str
    base_value: float
    sum_of_contributions: float
    top_contributions: list[Contribution]
    all_contributions: list[Contribution]


class FlaggedFeature(BaseModel):
    feature: str
    value: float
    train_min: float
    train_max: float
    robust_z: float
    outside_training_range: bool
    severity: Literal["unusual", "extreme"]


class DistributionCheck(BaseModel):
    status: Literal["ok", "warning", "abstain"]
    mahalanobis_distance: float
    mahalanobis_p99_train: float
    mahalanobis_max_train: float
    flagged_features: list[FlaggedFeature]
    messages: list[str]
    note: str = ("Heuristic check against the training distribution; not a validated "
                 "out-of-distribution detector.")


class PredictionBlock(BaseModel):
    predicted_class: Literal["malignant", "benign"]
    probability_malignant: float = Field(
        description="Model-estimated probability of the 'malignant' dataset class. "
                    "Dataset-specific; not a person's probability of having cancer.")
    operating_threshold: float
    probability_note: str


class PredictResponse(BaseModel):
    disclaimer: str
    model_version: str
    model_name: str
    input_validation: Literal["valid"]
    abstained: bool
    abstain_reason: list[str] = []
    prediction: PredictionBlock | None
    distribution_check: DistributionCheck
    explanation: Explanation | None
    limitations: list[str]


class SamplePredictResponse(PredictResponse):
    sample_id: int
    features: dict[str, float]
    ground_truth: Literal["malignant", "benign"] = Field(
        description="Known dataset label. The UI keeps this hidden until the user reveals it.")


class BatchRow(BaseModel):
    row: int
    sample_id: str | None
    status: Literal["predicted", "warning", "abstained", "invalid"]
    predicted_class: Literal["malignant", "benign"] | None = None
    probability_malignant: float | None = None
    errors: list[str] = []
    warnings: list[str] = []


class BatchResponse(BaseModel):
    disclaimer: str
    model_version: str
    operating_threshold: float
    summary: dict[str, int]
    rows: list[BatchRow]
