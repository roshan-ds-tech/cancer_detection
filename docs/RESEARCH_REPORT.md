# Research Report — OncoSense AI

> Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.

This report summarises the research done before and during implementation. Each statement is labelled with its type:

- **[Source]** is a fact taken from the cited documentation or publication.
- **[Experiment]** is a result produced by this project's code, from run `20261125T114504Z-d23509` with seed 42.
- **[Assumption/Judgement]** is a design decision or interpretation made by this project.

## 1. Research questions

1. What exactly is the Breast Cancer Wisconsin (Diagnostic) dataset, and what are its limits?
2. Which preprocessing, models and evaluation protocol suit a small (n = 569), clean, tabular, two-class dataset?
3. How can model selection be made free of leakage and reproducible on a laptop at no cost?
4. How should performance, uncertainty and explanations be shown so that they are not mistaken for clinical claims?

## 2. Dataset facts (verified)

| Fact | Value | Type |
|---|---|---|
| Instances | 569 | [Source] UCI; sklearn `DESCR`; [Experiment] loader assertion |
| Features | 30 real-valued (10 measurements × mean / SE / worst) | [Source] |
| Classes | 212 malignant, 357 benign | [Source]; [Experiment] |
| Missing values | None | [Source]; [Experiment] 0 missing, 0 non-finite |
| Duplicate rows | — | [Experiment] 0 |
| Creators | Dr William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (Univ. of Wisconsin) | [Source] |
| Donated | November 1995 | [Source] |
| Licence | CC BY 4.0 | [Source] UCI page |
| DOI | 10.24432/C5DW2B | [Source] |
| scikit-learn label encoding | 0 = malignant, 1 = benign (`target_names = ['malignant','benign']`) | [Experiment] verified in code; re-encoded to 1 = malignant |

**Feature extraction [Source].** Each sample comes from a fine-needle aspirate (FNA) of a breast mass. The aspirate was digitised and viewed as an image. An operator placed approximate outlines around the cell nuclei, and an active-contour ("snake") algorithm refined them. Ten characteristics were computed for each nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each image the authors recorded three values per characteristic: the mean, the standard error, and the "worst" value, which is the mean of the three largest. This gives 30 features (Street et al., 1993).

**What the dataset is not [Source/Judgement].** It holds no images, no mammograms, no patient demographics, symptoms or history, and no sample identifiers in the scikit-learn copy. The UCI CSV does include an ID column, which this project never uses.

## 3. Methodological findings that shaped the design

| Finding | Basis | Consequence in OncoSense |
|---|---|---|
| Feature scales span about 5.4 orders of magnitude | [Experiment] EDA | `StandardScaler` inside pipelines for LR, SVM and KNN only |
| Severe multicollinearity: r(mean radius, mean perimeter) = 0.998; VIF ≈ 4,000 | [Experiment] EDA | Regularised models; attributions read with caution |
| 11 of 30 features have univariate AUC > 0.9 | [Experiment] EDA | Simple models are expected to be competitive, so deep learning is not justified |
| Tuning and evaluating on the same CV folds biases estimates upward | [Source] Varma & Simon (2006) | Nested CV is used for selection |
| Test-set reuse for selection leaks information | [Source] standard practice; TRIPOD+AI | Test set is used once, after selection |
| Probabilities can be miscalibrated (SVM margins, NB) | [Source] sklearn calibration docs | Platt scaling for the SVM; Brier score reported; calibration curves shown |
| A threshold of 0.5 is arbitrary when costs are asymmetric | [Judgement] | Threshold chosen from out-of-fold training predictions with a recall target of 0.97 |
| SHAP and permutation importance describe model behaviour | [Source] Lundberg & Lee (2017); sklearn docs | Every explanation carries a "not causal" note |
| Software giving patient-specific diagnostic output from images or signals can be a regulated device | [Source] FDA CDS guidance (2022) | Disclaimers everywhere; no clinical claims; no real patient data accepted |

## 4. Summary of experimental results [Experiment]

- Nested CV (5×2 outer, 5 inner) ranked the models by mean ROC-AUC: LR 0.9950, SVM 0.9948, GB 0.9924, GNB 0.9882, RF 0.9878, KNN 0.9870, DT 0.9534.
- Selection rule: take the pool within 1 SE of the best mean AUC, then the lowest CV Brier, then the simplest model. Result: **SVM (RBF, C = 10, γ = 0.01, Platt-calibrated)**.
- Sigmoid re-calibration of the selected pipeline did not lower out-of-fold Brier (0.0182 → 0.0189), so none was applied.
- Operating threshold **0.39**, giving out-of-fold training recall 0.971 and specificity 0.972.
- Held-out test at 0.39: recall 0.952 (95% CI 0.875–1.0), specificity 1.000, ROC-AUC 0.996, Brier 0.018. There were 2 false negatives and 0 false positives.
- Global SHAP ranking: worst texture, mean concave points, mean concavity, worst concavity, worst radius, worst area.

Full tables are in [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

## 5. Interpretation [Judgement]

The results agree with published work on this dataset. Linear and kernel methods reach very high discrimination, and the ranking among the top models is within noise. High performance on 114 retrospective, single-centre samples says nothing about performance on other populations, imaging processes or clinical workflows. No external validation exists, so no clinical conclusion can be drawn.

## 6. Sources

1. UCI ML Repository — Breast Cancer Wisconsin (Diagnostic). https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (doi:10.24432/C5DW2B)
2. Original dataset description file `wdbc.names` / WDBC.doc, UW CS FTP mirror: https://ftp1.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.doc
3. scikit-learn `load_breast_cancer`. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
4. Street, W.N., Wolberg, W.H., Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Proc. SPIE* 1905, 861–870. https://doi.org/10.1117/12.148698
5. Wolberg, W.H., Street, W.N., Mangasarian, O.L. (1994). Machine learning techniques to diagnose breast cancer from fine-needle aspirates. *Cancer Letters* 77, 163–171.
6. Mangasarian, O.L., Street, W.N., Wolberg, W.H. (1995). Breast cancer diagnosis and prognosis via linear programming. *Operations Research* 43(4), 570–577. https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html
7. Collins, G.S. et al. (2024). TRIPOD+AI statement. *BMJ* 385:e078378. https://www.tripod-statement.org/ ; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/
8. U.S. FDA (2022). Clinical Decision Support Software: Guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
9. Varma, S., Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 7:91. https://doi.org/10.1186/1471-2105-7-91
10. Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*. https://github.com/shap/shap
11. scikit-learn User Guide: probability calibration, permutation importance, nested CV. https://scikit-learn.org/stable/
