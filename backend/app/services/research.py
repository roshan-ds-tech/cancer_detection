"""Structured methodology content served at /api/research/methodology.

Run-specific values (split sizes, selected model, threshold) are injected from the
training metadata so the page never shows stale or invented numbers.
"""
from __future__ import annotations

from app.services.registry import Registry

CITATIONS = [
    {"id": "uci", "text": "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer "
                          "Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. CC BY 4.0.",
     "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
     "doi": "10.24432/C5DW2B"},
    {"id": "street1993", "text": "Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature "
                                 "extraction for breast tumor diagnosis. Proc. SPIE 1905, 861–870.",
     "url": "https://doi.org/10.1117/12.148698", "doi": "10.1117/12.148698"},
    {"id": "mangasarian1995", "text": "Mangasarian, O.L., Street, W.N., & Wolberg, W.H. (1995). Breast cancer "
                                      "diagnosis and prognosis via linear programming. Operations Research "
                                      "43(4), 570–577.",
     "url": "https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html", "doi": "10.1287/opre.43.4.570"},
    {"id": "wolberg1994", "text": "Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1994). Machine learning "
                                  "techniques to diagnose breast cancer from fine-needle aspirates. Cancer "
                                  "Letters 77, 163–171.", "url": None, "doi": "10.1016/0304-3835(94)90099-X"},
    {"id": "sklearn_ds", "text": "scikit-learn: sklearn.datasets.load_breast_cancer documentation.",
     "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html"},
    {"id": "sklearn", "text": "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, "
                              "2825–2830.", "url": "https://scikit-learn.org/stable/"},
    {"id": "tripod_ai", "text": "Collins, G.S., et al. (2024). TRIPOD+AI statement: updated guidance for "
                                "reporting clinical prediction models that use regression or machine "
                                "learning methods. BMJ 385, e078378.",
     "url": "https://www.tripod-statement.org/", "doi": "10.1136/bmj-2023-078378"},
    {"id": "fda_cds", "text": "U.S. FDA (2022). Clinical Decision Support Software — Guidance for Industry "
                              "and Food and Drug Administration Staff.",
     "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software"},
    {"id": "shap", "text": "Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model "
                           "Predictions. NeurIPS 30.", "url": "https://github.com/shap/shap"},
    {"id": "varma2006", "text": "Varma, S. & Simon, R. (2006). Bias in error estimation when using "
                                "cross-validation for model selection. BMC Bioinformatics 7, 91.",
     "url": "https://doi.org/10.1186/1471-2105-7-91", "doi": "10.1186/1471-2105-7-91"},
]


def methodology(reg: Registry) -> dict:
    md = reg.metadata
    ds = md["dataset"]
    return {
        "disclaimer": md["disclaimer"],
        "model_version": md["model_version"],
        "sections": [
            {"id": "provenance", "title": "Dataset provenance", "body": [
                f"Breast Cancer Wisconsin (Diagnostic): {ds['n_samples']} samples, 30 real-valued "
                "features, binary label (malignant / benign). Created by Dr William H. Wolberg, "
                "W. Nick Street and Olvi L. Mangasarian (University of Wisconsin); donated to UCI in "
                "November 1995; licensed CC BY 4.0.",
                "Features were computed from digitised images of fine-needle aspirates (FNA) of breast "
                "masses. An operator outlined cell nuclei with an interactive 'snake' contour tool; ten "
                "nuclear characteristics were measured and their mean, standard error and 'worst' "
                "(mean of the three largest) values recorded per image [street1993].",
                "The dataset contains no images, no mammograms and no patient demographics. Loaded "
                "via scikit-learn, which mirrors the UCI copy [sklearn_ds].",
            ]},
            {"id": "preprocessing", "title": "Preprocessing and leakage control", "body": [
                "No missing values or duplicate rows exist; nothing is imputed. Sample IDs are not "
                "used as features (the scikit-learn copy does not include them).",
                "Labels re-encoded so 1 = malignant (positive class). scikit-learn's native encoding "
                "is the reverse (0 = malignant).",
                f"Stratified hold-out split: {ds['n_train']} training / {ds['n_test']} test "
                f"(seed {md['training_config']['seed']}). The test split is not used for tuning, "
                "selection, calibration, threshold choice or EDA statistics.",
                "Standardisation is fitted inside scikit-learn Pipelines, i.e. on training folds only. "
                "Tree models and Naive Bayes are not scaled.",
            ]},
            {"id": "models", "title": "Algorithms", "body": [
                "Logistic Regression, SVM (RBF, Platt-calibrated), Random Forest, Decision Tree, "
                "K-Nearest Neighbours, Gaussian Naive Bayes and (optional) Gradient Boosting, each with "
                "a small hyperparameter grid suitable for an 8 GB laptop.",
                "No deep learning: with 455 training rows of tabular features, conventional models are "
                "appropriate and easier to audit.",
            ]},
            {"id": "evaluation", "title": "Evaluation methodology", "body": [
                f"Nested cross-validation ({md['training_config']['outer_cv']} outer, "
                f"{md['training_config']['inner_cv']} inner grid search on ROC-AUC) on the training split "
                "gives approximately unbiased performance estimates for selection [varma2006].",
                f"Selection rule: {md['selection_rule']}. {md['selection_rationale']}",
                f"Calibration: {md['calibration']['rule']} Applied: {md['calibration']['applied']}.",
                f"Operating threshold: {md['threshold']['operating_threshold']} — {md['threshold']['rule']}",
                "The selected model was then evaluated once on the untouched test split; percentile "
                "bootstrap 95% confidence intervals (1000 resamples) accompany test metrics. Test results "
                "for non-selected models are shown for transparency only.",
            ]},
            {"id": "explainability", "title": "Explainability", "body": [
                "Global: permutation importance (drop in test ROC-AUC), standardised coefficients or "
                "impurity importance where native, mean |SHAP|, and feature correlations.",
                "Local: SHAP permutation explainer on P(malignant) with a 50-row training background "
                "[shap]. Attributions describe model behaviour, not biology; correlated features "
                "(radius/perimeter/area) share credit unpredictably.",
            ]},
            {"id": "reporting", "title": "Reporting and regulatory context", "body": [
                "Documentation follows the spirit of TRIPOD+AI [tripod_ai] where applicable (data source, "
                "predictors, outcome, sample size, missing data, model development, performance, "
                "limitations). This is a development study only: no external validation exists.",
                "Software that analyses medical images or signals, or provides patient-specific "
                "diagnostic output, may fall under device regulation [fda_cds]. OncoSense is an "
                "educational prototype and must not be used for clinical decisions.",
            ]},
        ],
        "citations": CITATIONS,
    }
