# Dataset Analysis

> Educational research prototype only. Not validated for clinical use.

Reproduce with `python backend/scripts/run_eda.py`, which writes `artifacts/eda.json` and `artifacts/figures/eda_*.png`, or
with the executed notebook `backend/notebooks/EDA.ipynb`. The analysis code lives in `backend/app/ml/eda.py`.

**Scope rule.** The dataset-level integrity checks use all 569 rows. All distributional statistics use only the **455-row
training split**: correlations, class comparisons, outliers, PCA and VIF. The 114-row test split therefore never
influences any design decision.

## 1. Dimensions and schema

- 569 rows × 30 numeric features plus a binary target. The positive class is malignant (1). See `app/ml/data.py` for the re-encoding.
- Feature names use snake_case: `mean_<m>`, `<m>_se` and `worst_<m>`, where `m` ∈ {radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave_points, symmetry, fractal_dimension}.
- **Units.** The source documentation does not give physical units. Size features are image-derived. Smoothness, compactness, concavity, concave points, symmetry and fractal dimension are dimensionless ratios, counts or indices. The app labels them "unspecified in source" rather than inventing units.

## 2. Integrity

| Check | Result |
|---|---|
| Missing values | 0 |
| Non-finite values | 0 |
| Duplicate rows | 0 |
| Negative values | 0 |
| Zero values | 13 rows each in the six concavity / concave-points features. A zero means a contour with no concavities, which is plausible. |
| Class counts | 357 benign / 212 malignant (37.3% malignant) |
| Split | Train 455 (170 malignant) / Test 114 (42 malignant), stratified, seed 42 |

## 3. Class separation (training split)

Univariate ROC-AUC measures how well one feature alone ranks malignant above benign:

| Strongest | AUC | Weakest | AUC |
|---|---|---|---|
| worst_perimeter | 0.974 | symmetry_se | 0.443 |
| worst_radius | 0.969 | smoothness_se | 0.464 |
| worst_area | 0.968 | mean_fractal_dimension | 0.493 |
| mean_concave_points | 0.964 | texture_se | 0.514 |
| worst_concave_points | 0.963 | fractal_dimension_se | 0.621 |

Eleven features have a univariate AUC above 0.9. Size and concavity measures are higher in the malignant class. An AUC below 0.5, as for symmetry_se and smoothness_se, means the feature runs slightly the other way and is weak on its own. Cohen's d and class-wise medians are in `eda.json` and on the Dataset Explorer page.

## 4. Multicollinearity

- 21 feature pairs have |r| ≥ 0.9. The strongest are mean_radius–mean_perimeter (0.998), worst_radius–worst_perimeter (0.994) and mean_radius–mean_area (0.987).
- Variance inflation factors: mean_perimeter about 4,111, mean_radius about 4,042, worst_radius about 800. In total, 24 of 30 features have VIF > 10.
- **Implications.** Individual linear coefficients are unstable. Permutation importance understates shared information. SHAP credit is split arbitrarily among correlated inputs. Regularisation (L2-LR, the SVM margin) is preferred.

## 5. Distributions and outliers

- Many size and SE features are right-skewed, area_se most strongly.
- IQR-rule outliers on the training split: area_se has 53, perimeter_se 34, radius_se 31, compactness_se 24, concavity_se 24.
- **Policy.** Outliers are **retained**. They are plausible measurements, and removing them would make the data look cleaner than real-world data. For inference, the out-of-distribution heuristic flags inputs beyond the training range.

## 6. Scaling

Feature means differ by about 5.4 orders of magnitude (area in the hundreds, SE of fractal dimension around 0.004). Distance-based and margin-based models
need standardisation, while tree models and Gaussian NB do not. Scaling is fitted inside each pipeline, on training folds only.

## 7. PCA (exploratory only)

On standardised training data, PC1 explains 44.6% of variance, PC2 18.5% and PC3 9.6%. The first five components together explain 84.9%. In PC1–PC2 space the classes are largely separable along PC1. PCA is **not** part of any model pipeline.

## 8. Figures

`artifacts/figures/`: eda_class_distribution, eda_correlation_heatmap, eda_histograms_mean_features,
eda_boxplots_mean_features, eda_pca, eda_univariate_auc. The Dataset Explorer page renders the same statistics
interactively from `/api/dataset/eda`.

## 9. Known dataset limitations

See [ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md). In short, the data come from a single centre in a historic period. They depend on an operator-assisted segmentation process, the class prevalence reflects a referral population, and the dataset contains no demographic variables, so subgroup fairness cannot be assessed.
