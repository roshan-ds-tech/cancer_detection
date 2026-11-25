"""End-to-end training workflow.

Leakage controls
----------------
* The 20 % stratified hold-out test split is created first and is not touched until
  a model has been selected (step 4). Its results never feed back into selection,
  tuning, calibration or threshold choice.
* Hyperparameters are tuned with GridSearchCV *inside* each outer CV fold (nested CV),
  so the CV estimates used for selection are not optimistically biased by tuning.
* All fitted preprocessing lives inside sklearn Pipelines, so scalers are fit only on
  the training portion of each fold.
* The decision threshold and the calibration choice are made from out-of-fold
  predictions on the training split.
"""
from __future__ import annotations

import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
)

from app.core import config
from app.ml import data as D
from app.ml import metrics as M
from app.ml.artifacts import save_model, write_json
from app.ml.explain import LocalExplainer, background_sample, native_importances, permutation_importances
from app.ml.models import get_candidates
from app.ml.ood import OODChecker
from app.ml.tracking import log_experiment, log_run, now_iso

TARGET_OOF_RECALL = 0.97
THRESHOLD_BOUNDS = (0.05, 0.50)
CV_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "brier"]


def library_versions() -> dict:
    import fastapi  # noqa: F401  (recorded for reproducibility; optional at train time)
    import shap

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "shap": shap.__version__,
        "joblib": joblib.__version__,
        "fastapi": fastapi.__version__,
    }


def _summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else 0.0
    return {"mean": float(a.mean()), "std": sd, "se": float(se),
            "ci95": [float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se)],
            "folds": [float(v) for v in a]}


def nested_cv(cand, X, y, seed, n_repeats, inner_splits=5):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    per_metric = {k: [] for k in CV_METRICS}
    chosen_params = []
    for tr, va in outer.split(X, y):
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        gs = GridSearchCV(clone(cand.pipeline), cand.param_grid, cv=inner,
                          scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X.iloc[tr], y.iloc[tr])
        prob = gs.predict_proba(X.iloc[va])[:, 1]
        m = M.threshold_metrics(y.iloc[va], prob, 0.5)
        for k in CV_METRICS:
            per_metric[k].append(m[k])
        chosen_params.append({k: str(v) for k, v in gs.best_params_.items()})
    return {k: _summary(v) for k, v in per_metric.items()}, chosen_params


def _measure_inference(pipe, X_test: pd.DataFrame) -> dict:
    single = X_test.iloc[[0]]
    pipe.predict_proba(single)  # warm-up
    t = []
    for _ in range(30):
        s = time.perf_counter()
        pipe.predict_proba(single)
        t.append(time.perf_counter() - s)
    s = time.perf_counter()
    pipe.predict_proba(X_test)
    batch = time.perf_counter() - s
    return {"single_row_ms_median": float(np.median(t) * 1e3),
            "batch_per_row_ms": float(batch / len(X_test) * 1e3),
            "batch_rows": int(len(X_test))}


def select_model(results: list[dict]) -> tuple[str, str]:
    ok = [r for r in results if r["status"] == "completed"]
    if not ok:
        raise RuntimeError("All candidate models failed")
    best = max(ok, key=lambda r: r["cv"]["roc_auc"]["mean"])
    bar = best["cv"]["roc_auc"]["mean"] - best["cv"]["roc_auc"]["se"]
    pool = [r for r in ok if r["cv"]["roc_auc"]["mean"] >= bar]
    chosen = min(pool, key=lambda r: (round(r["cv"]["brier"]["mean"], 4), r["complexity_rank"]))
    rationale = (
        f"Highest nested-CV ROC-AUC: {best['name']} ({best['cv']['roc_auc']['mean']:.4f} ± "
        f"{best['cv']['roc_auc']['se']:.4f} SE). Candidates within one standard error of that "
        f"value ({', '.join(r['name'] for r in pool)}) were treated as statistically "
        f"indistinguishable; among them the one with the lowest mean CV Brier score (best "
        f"probability quality) was chosen, with model simplicity as the final tie-break. "
        f"Selected: {chosen['name']} (CV ROC-AUC {chosen['cv']['roc_auc']['mean']:.4f}, "
        f"CV Brier {chosen['cv']['brier']['mean']:.4f}, CV recall "
        f"{chosen['cv']['recall']['mean']:.4f}). The held-out test set was not used."
    )
    return chosen["key"], rationale


def choose_threshold(y, oof_prob) -> dict:
    lo, hi = THRESHOLD_BOUNDS
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    feasible = [t for t in grid if M.threshold_metrics(y, oof_prob, t)["recall"] >= TARGET_OOF_RECALL]
    thr = float(max(feasible)) if feasible else float(lo)
    m = M.threshold_metrics(y, oof_prob, thr)
    return {
        "operating_threshold": thr,
        "rule": f"Largest threshold in [{lo}, {hi}] whose out-of-fold training recall "
                f">= {TARGET_OOF_RECALL} (5-fold CV on the training split only).",
        "target_recall": TARGET_OOF_RECALL,
        "achieved_oof_recall": m["recall"],
        "achieved_oof_specificity": m["specificity"],
        "feasible": bool(feasible),
    }


def run_training(seed: int = config.RANDOM_SEED, artifacts_dir: Path | None = None,
                 quick: bool = False, make_figures: bool = True, log=print) -> dict:
    t_start = time.perf_counter()
    out = Path(artifacts_dir or config.ARTIFACTS_DIR)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    db = out / "experiments.sqlite"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    np.random.seed(seed)

    # 1-3. load, validate, split ------------------------------------------------
    X, y = D.load_dataset()
    fingerprint = D.dataset_fingerprint(X, y)
    X_train, X_test, y_train, y_test = D.split_dataset(X, y, seed=seed)
    log(f"[data] {X.shape}, train={len(X_train)}, test={len(X_test)}, fingerprint={fingerprint}")

    # 4-7. candidates: nested CV + final tuning on the full training split --------
    n_repeats = 1 if quick else 2
    results, fitted = [], {}
    for cand in get_candidates(seed):
        exp_id = f"{run_id}-{cand.key}"
        rec = {"key": cand.key, "name": cand.name, "notes": cand.notes, "tags": cand.tags,
               "complexity_rank": cand.complexity_rank, "experiment_id": exp_id,
               "param_grid": {k: [str(v) for v in vs] for k, vs in cand.param_grid.items()}}
        try:
            t0 = time.perf_counter()
            cv, fold_params = nested_cv(cand, X_train, y_train, seed, n_repeats)
            cv_seconds = time.perf_counter() - t0
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            gs = GridSearchCV(clone(cand.pipeline), cand.param_grid, cv=inner,
                              scoring="roc_auc", n_jobs=-1, refit=False)
            gs.fit(X_train, y_train)
            final = clone(cand.pipeline).set_params(**gs.best_params_)
            t1 = time.perf_counter()
            final.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t1
            art = save_model(final, out / "candidates" / f"{cand.key}.joblib", out)
            rec.update({
                "status": "completed", "error": None, "cv": cv,
                "cv_scheme": f"nested: outer RepeatedStratifiedKFold(5x{n_repeats}), "
                             "inner StratifiedKFold(5) GridSearchCV on ROC-AUC",
                "outer_fold_selected_params": fold_params,
                "best_params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                                for k, v in gs.best_params_.items()},
                "inner_cv_best_roc_auc": float(gs.best_score_),
                "training_seconds": fit_seconds, "cv_seconds": cv_seconds,
                "model_size_bytes": art["size_bytes"], "artifact": art,
            })
            fitted[cand.key] = final
            log(f"[cv] {cand.name:32s} AUC={cv['roc_auc']['mean']:.4f}±{cv['roc_auc']['se']:.4f} "
                f"recall={cv['recall']['mean']:.3f} brier={cv['brier']['mean']:.4f}")
        except Exception as exc:  # failure is recorded and surfaced, never hidden
            rec.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            log(f"[cv] {cand.name} FAILED: {exc}")
        results.append(rec)

    # 8. model selection (training evidence only) --------------------------------
    selected_key, rationale = select_model(results)
    sel = next(r for r in results if r["key"] == selected_key)
    log(f"[select] {sel['name']}")
    best_pipe = fitted[selected_key]

    # calibration decision + threshold from out-of-fold training predictions
    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_raw = cross_val_predict(clone(best_pipe), X_train, y_train, cv=oof_cv,
                                method="predict_proba")[:, 1]
    calibrated_tmpl = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=5)
    oof_cal = cross_val_predict(calibrated_tmpl, X_train, y_train, cv=oof_cv,
                                method="predict_proba")[:, 1]
    brier_raw = brier_score_loss(y_train, oof_raw)
    brier_cal = brier_score_loss(y_train, oof_cal)
    use_cal = brier_cal < brier_raw - 0.002
    calibration_info = {
        "oof_brier_uncalibrated": float(brier_raw),
        "oof_brier_sigmoid_calibrated": float(brier_cal),
        "applied": "sigmoid (Platt) via CalibratedClassifierCV(cv=5)" if use_cal else "none",
        "rule": "Apply sigmoid calibration only if it lowers out-of-fold training Brier by > 0.002.",
        "oof_calibration_curve": M.curves(y_train, oof_cal if use_cal else oof_raw)["calibration"],
    }
    if use_cal:
        best_pipe = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=5).fit(X_train, y_train)
    oof = oof_cal if use_cal else oof_raw
    threshold_info = choose_threshold(y_train, oof)
    threshold_info["oof_sweep"] = M.threshold_sweep(y_train, oof)
    thr = threshold_info["operating_threshold"]
    log(f"[threshold] {thr} (calibration: {calibration_info['applied']})")

    # 9. single evaluation on the untouched test set ------------------------------
    test_probs = {}
    for r in results:
        if r["status"] != "completed":
            continue
        pipe = best_pipe if r["key"] == selected_key else fitted[r["key"]]
        prob = pipe.predict_proba(X_test)[:, 1]
        if not np.all(np.isfinite(prob)):
            r["status"], r["error"] = "failed", "Non-finite probabilities on test set"
            continue
        test_probs[r["key"]] = prob
        r["test"] = M.threshold_metrics(y_test, prob, 0.5)
        r["test_ci95"] = M.bootstrap_ci(y_test, prob, 0.5, n_boot=300 if quick else 1000, seed=seed)
        r["test_curves"] = M.curves(y_test, prob)
        r["test_threshold_sweep"] = M.threshold_sweep(y_test, prob)
        r["classification_report"] = M.report(y_test, prob, 0.5)
        r["inference"] = _measure_inference(pipe, X_test)
        if r["key"] == selected_key:
            r["test_at_operating_threshold"] = M.threshold_metrics(y_test, prob, thr)
            r["test_at_operating_threshold_ci95"] = M.bootstrap_ci(
                y_test, prob, thr, n_boot=300 if quick else 1000, seed=seed)
            r["classification_report_operating"] = M.report(y_test, prob, thr)

    sel_prob = test_probs[selected_key]
    errors = []
    for idx, p, t in zip(X_test.index, sel_prob, y_test):
        pred = int(p >= thr)
        if pred != t:
            errors.append({"sample_id": int(idx), "true_label": D.CLASS_MAPPING[int(t)],
                           "predicted_label": D.CLASS_MAPPING[pred], "probability_malignant": float(p),
                           "error_type": "false_negative" if t == 1 else "false_positive"})

    # learning curve (training split only)
    lc_sizes, lc_train, lc_val = learning_curve(
        clone(fitted[selected_key]), X_train, y_train, cv=oof_cv, scoring="roc_auc",
        train_sizes=np.linspace(0.2, 1.0, 5 if quick else 8), n_jobs=-1)
    learning = [{"train_size": int(s), "train_auc_mean": float(a.mean()), "train_auc_std": float(a.std()),
                 "val_auc_mean": float(b.mean()), "val_auc_std": float(b.std())}
                for s, a, b in zip(lc_sizes, lc_train, lc_val)]

    # explainability ---------------------------------------------------------------
    perm = {}
    for key, prob in test_probs.items():
        pipe = best_pipe if key == selected_key else fitted[key]
        perm[key] = permutation_importances(pipe, X_test, y_test, seed, n_repeats=5 if quick else 20)
    native = {k: native_importances(fitted[k], D.FEATURE_NAMES) for k in test_probs}
    bg = background_sample(X_train, seed)
    expl = LocalExplainer(best_pipe, bg, seed)
    shap_X = X_test if not quick else X_test.iloc[:20]
    shap_ex = expl.explain(shap_X)
    mean_abs = np.abs(shap_ex.values).mean(axis=0)
    shap_global = sorted([{"feature": f, "mean_abs_shap": float(v)}
                          for f, v in zip(D.FEATURE_NAMES, mean_abs)], key=lambda d: -d["mean_abs_shap"])
    corr = X_train.corr().round(3)

    # 10-12. persist -----------------------------------------------------------------
    model_version = f"oncosense-{selected_key}-{run_id}"
    model_art = save_model(best_pipe, out / "models" / f"{model_version}.joblib", out)
    ood = OODChecker.fit(X_train)
    versions = library_versions()
    specs = [s.__dict__ for s in D.feature_specs()]

    metadata = {
        "project": "OncoSense AI",
        "disclaimer": config.DISCLAIMER,
        "run_id": run_id, "model_version": model_version, "created_at": now_iso(),
        "selected_model": {"key": selected_key, "name": sel["name"], "best_params": sel["best_params"]},
        "selection_rationale": rationale,
        "selection_rule": "max nested-CV ROC-AUC; within-1-SE pool -> min CV Brier; then simplicity",
        "calibration": calibration_info,
        "threshold": threshold_info,
        "feature_names": D.FEATURE_NAMES, "feature_specs": specs,
        "class_mapping": {"0": "benign", "1": "malignant"}, "positive_class": "malignant",
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)", "source": "sklearn.datasets.load_breast_cancer",
            "uci_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "doi": "10.24432/C5DW2B", "license": "CC BY 4.0", "fingerprint": fingerprint,
            "n_samples": int(len(X)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_class_counts": {"malignant": int(y_train.sum()), "benign": int((1 - y_train).sum())},
            "test_class_counts": {"malignant": int(y_test.sum()), "benign": int((1 - y_test).sum())},
        },
        "training_config": {"seed": seed, "test_size": config.TEST_SIZE, "quick": quick,
                            "outer_cv": f"RepeatedStratifiedKFold(5 x {n_repeats})",
                            "inner_cv": "StratifiedKFold(5)", "tuning_metric": "roc_auc"},
        "library_versions": versions,
        "model_artifact": model_art,
    }
    metrics_doc = {"run_id": run_id, "model_version": model_version, "selected_key": selected_key,
                   "operating_threshold": thr, "models": results, "selected_test_errors": errors,
                   "learning_curve": learning,
                   "notes": {"cv": "Mean ± SD over outer folds of nested CV on the training split "
                                   "(threshold 0.5). Used for model selection.",
                             "test": "Single evaluation on the untouched 20% hold-out split after "
                                     "selection. Reported for every model for transparency; not "
                                     "used for selection."}}
    explain_doc = {"model_version": model_version, "selected_key": selected_key,
                   "permutation_importance": perm, "native_importance": native,
                   "shap_global": shap_global,
                   "shap_method": "SHAP permutation explainer on P(malignant), background of "
                                  f"{len(bg)} training rows, evaluated on {len(shap_X)} test rows",
                   "correlation": {"features": D.FEATURE_NAMES, "matrix": corr.to_numpy().tolist()},
                   "caveat": "Attributions describe model behaviour, not biological causation."}
    demo = {"sample_ids": [int(i) for i in X_test.index],
            "features": X_test.round(6).to_dict(orient="records"),
            "labels": [D.CLASS_MAPPING[int(v)] for v in y_test]}
    train_records = {"sample_ids": [int(i) for i in X_train.index],
                     "features": X_train.round(6).to_dict(orient="records"),
                     "labels": [D.CLASS_MAPPING[int(v)] for v in y_train]}

    write_json(out / "metadata.json", metadata)
    write_json(out / "metrics.json", metrics_doc)
    write_json(out / "explainability.json", explain_doc)
    write_json(out / "ood.json", ood.to_dict())
    write_json(out / "demo_samples.json", demo)
    write_json(out / "train_records.json", train_records)
    write_json(out / "background.json", {"features": D.FEATURE_NAMES,
                                         "rows": bg.to_numpy().tolist(), "seed": seed})
    write_json(out / "model_manifest.json", {"model_version": model_version, "model": model_art,
                                             "run_id": run_id, "created_at": now_iso()})

    if make_figures:
        from app.ml.figures import training_figures
        training_figures(out / "figures", results, selected_key, y_test, test_probs, learning,
                         perm[selected_key], shap_ex, shap_X, thr)

    # experiment tracker -------------------------------------------------------------
    duration = time.perf_counter() - t_start
    for r in results:
        log_experiment(db, {
            "experiment_id": r["experiment_id"], "run_id": run_id, "algorithm": r["name"],
            "status": r["status"], "error": r.get("error"), "hyperparameters": r.get("best_params"),
            "seed": seed,
            "cv_metrics": {k: v["mean"] for k, v in r.get("cv", {}).items()} or None,
            "test_metrics": {k: v for k, v in r.get("test", {}).items() if k in M.METRIC_KEYS} or None,
            "training_seconds": r.get("training_seconds"), "dataset_version": fingerprint,
            "artifact_path": (r.get("artifact") or {}).get("path"),
        })
    log_run(db, {"run_id": run_id, "created_at": now_iso(), "dataset_version": fingerprint,
                 "seed": seed, "selected_model": sel["name"], "model_version": model_version,
                 "selection_rule": metadata["selection_rule"], "library_versions": versions,
                 "duration_seconds": duration})
    log(f"[done] {model_version} in {duration:.1f}s")
    return {"metadata": metadata, "metrics": metrics_doc, "explain": explain_doc}
