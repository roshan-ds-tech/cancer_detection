# Study Methodology (TRIPOD+AI-oriented)

> Educational research prototype only. Not validated for clinical use.

This document is organised loosely around TRIPOD+AI reporting items (Collins et al., 2024) for a **model development
study with internal validation only**. Implementation details are in [ML_METHODOLOGY.md](ML_METHODOLOGY.md).

| TRIPOD+AI theme | This study |
|---|---|
| **Objective** | Develop and internally validate a classifier that separates benign from malignant dataset labels using 30 FNA-derived nuclear features, as an *educational demonstration* of sound ML methodology. Intended use: teaching and research only. There is no clinical intended use. |
| **Data source** | Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (doi:10.24432/C5DW2B), CC BY 4.0, loaded via scikit-learn 1.9.1. Retrospective. Single centre (University of Wisconsin). Collected before November 1995. |
| **Participants** | 569 FNA samples of breast masses. Eligibility criteria and patient characteristics are not documented in the dataset and are not available. |
| **Outcome** | Binary label: malignant (positive, 1) or benign (0), as recorded in the dataset. The dataset documentation does not define how the reference standard was obtained, so it is treated as given. |
| **Predictors** | 30 continuous features: the mean, SE and "worst" of 10 nuclear measurements computed from a digitised FNA image. No other predictors exist or were added. Predictors are measured from the same sample that defines the outcome. |
| **Sample size** | 569 samples with 212 events (malignant). Train: 455 (170 events); test: 114 (42 events). No formal sample-size calculation was done (fixed public dataset). With 42 test events, the confidence intervals are wide. |
| **Missing data** | None in the source. At inference time, missing inputs are **rejected**, never imputed. |
| **Analytical methods** | Stratified 80/20 split with seed 42. Seven candidate algorithms, each in a scikit-learn Pipeline. Nested CV: outer RepeatedStratifiedKFold(5 × 2), inner StratifiedKFold(5) grid search on ROC-AUC. Selection: highest mean CV AUC, then the pool within 1 SE, then lowest CV Brier, then simplest. Calibration: sigmoid applied only if out-of-fold Brier improves by > 0.002. Threshold: the largest t ∈ [0.05, 0.5] with out-of-fold training recall ≥ 0.97. |
| **Model evaluation** | A single evaluation of the selected model on the untouched test split. Metrics: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC (average precision), Brier, confusion matrix and classification report. Percentile bootstrap 95% CIs from 1,000 resamples. Calibration curves, threshold sweep, learning curve and error listing. |
| **Class imbalance** | Moderate (37% positive). Stratified splits are used, with no resampling. Metrics are chosen so they are not dominated by the majority class. |
| **Fairness** | Not assessable: the dataset has no demographic attributes. This is stated as a limitation. |
| **Explainability** | Global: SHAP (permutation explainer, mean \|SHAP\| on the test split), permutation importance (Δ ROC-AUC), native coefficients and impurity importances, correlations. Local: per-prediction SHAP. |
| **Results** | See [MODEL_EVALUATION.md](MODEL_EVALUATION.md). |
| **Limitations** | See [ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md). There is no external validation, so results describe this dataset only. |
| **Reproducibility** | Fixed seed (42). Library versions are recorded in `artifacts/metadata.json` and pinned in `requirements.txt`. Each run is logged in `artifacts/experiments.sqlite`. Dataset fingerprint: `a0c1a195d2fafbb7`. |
| **Code availability** | This repository. |

## Leakage prevention checklist

- [x] Test split created before any analysis that could influence modelling. EDA statistics use the training split only.
- [x] No identifiers or target-derived columns among the features. A unit test checks that no feature correlates with the label at |r| ≥ 0.95.
- [x] Scaling is fitted inside Pipelines, so it is learned on training folds only. A unit test verifies the scaler means equal the training means.
- [x] Hyperparameters are tuned in inner folds; performance is estimated in outer folds (nested CV).
- [x] Model selection, calibration and threshold are chosen from training-split evidence only. A unit test re-runs selection with all test fields removed and gets the same choice.
- [x] The test set is evaluated once, after selection. Results for non-selected models are reported for transparency but are not used.
- [x] The OOD reference statistics and the SHAP background are fitted on the training split only.
