"""Lightweight distribution-shift heuristics.

This is NOT a validated out-of-distribution detector. It flags inputs that look
unlike the training data so the application can warn, or abstain, instead of
returning a confident-looking prediction for an implausible input.

Two signals, both fitted on the training split only:
  1. Per-feature range checks (training min/max, robust z-score using median/IQR).
  2. A multivariate Mahalanobis distance on standardised features using a
     Ledoit-Wolf shrunk covariance. This catches combinations that are individually
     plausible but jointly inconsistent (e.g. a large radius with a tiny area).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROBUST_Z_WARN = 4.0
RANGE_ABSTAIN_FRACTION = 0.5  # beyond training range by > 50% of that range
MAHAL_ABSTAIN_FACTOR = 2.0  # > 2x the largest training distance


class OODChecker:
    def __init__(self, params: dict):
        self.p = params
        self.features = params["features"]
        self._mean = np.array(params["mean"])
        self._scale = np.array(params["scale"])
        self._precision = np.array(params["precision"])

    @classmethod
    def fit(cls, X_train: pd.DataFrame) -> "OODChecker":
        Xv = X_train.to_numpy(dtype=float)
        mean = Xv.mean(axis=0)
        scale = Xv.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        Z = (Xv - mean) / scale
        lw = LedoitWolf().fit(Z)
        precision = lw.precision_
        d = np.sqrt(np.einsum("ij,jk,ik->i", Z - lw.location_, precision, Z - lw.location_))
        q1, med, q3 = np.percentile(Xv, [25, 50, 75], axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1e-12
        params = {
            "features": list(X_train.columns),
            "min": Xv.min(axis=0).tolist(),
            "max": Xv.max(axis=0).tolist(),
            "median": med.tolist(),
            "iqr": iqr.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "precision": precision.tolist(),
            "lw_location": lw.location_.tolist(),
            "mahalanobis_p99": float(np.percentile(d, 99)),
            "mahalanobis_max": float(d.max()),
        }
        return cls(params)

    def to_dict(self) -> dict:
        return self.p

    def mahalanobis(self, x: np.ndarray) -> float:
        z = (x - self._mean) / self._scale - np.array(self.p["lw_location"])
        return float(np.sqrt(z @ self._precision @ z))

    def check(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        lo, hi = np.array(self.p["min"]), np.array(self.p["max"])
        rng = hi - lo
        robust_z = (x - np.array(self.p["median"])) / (np.array(self.p["iqr"]) / 1.349)
        flags, abstain_reasons = [], []
        for i, f in enumerate(self.features):
            outside = x[i] < lo[i] or x[i] > hi[i]
            extreme_z = abs(robust_z[i]) > ROBUST_Z_WARN
            if outside or extreme_z:
                far = (x[i] < lo[i] - RANGE_ABSTAIN_FRACTION * rng[i]) or (
                    x[i] > hi[i] + RANGE_ABSTAIN_FRACTION * rng[i]
                )
                flags.append({
                    "feature": f,
                    "value": float(x[i]),
                    "train_min": float(lo[i]),
                    "train_max": float(hi[i]),
                    "robust_z": float(robust_z[i]),
                    "outside_training_range": bool(outside),
                    "severity": "extreme" if far else "unusual",
                })
                if far:
                    abstain_reasons.append(
                        f"{f}={x[i]:.4g} is far outside the training range "
                        f"[{lo[i]:.4g}, {hi[i]:.4g}]"
                    )
        d = self.mahalanobis(x)
        if d > MAHAL_ABSTAIN_FACTOR * self.p["mahalanobis_max"]:
            abstain_reasons.append(
                f"Joint feature pattern is far from the training data (Mahalanobis {d:.1f} "
                f"> {MAHAL_ABSTAIN_FACTOR:g} x training max {self.p['mahalanobis_max']:.1f})"
            )
        multivariate_unusual = d > self.p["mahalanobis_p99"]
        if abstain_reasons:
            status = "abstain"
        elif flags or multivariate_unusual:
            status = "warning"
        else:
            status = "ok"
        messages = list(abstain_reasons)
        if status == "warning":
            if flags:
                messages.append(
                    f"{len(flags)} feature(s) are unusual relative to the training data."
                )
            if multivariate_unusual:
                messages.append(
                    "The combination of feature values is less typical than 99% of training samples."
                )
        return {
            "status": status,
            "mahalanobis_distance": d,
            "mahalanobis_p99_train": self.p["mahalanobis_p99"],
            "mahalanobis_max_train": self.p["mahalanobis_max"],
            "flagged_features": flags,
            "messages": messages,
        }
