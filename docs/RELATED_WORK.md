# Related Work

Published findings appear in the **Published** column. This project's own numbers are in the final section and are
labelled as **this project**. No published accuracy figure has been invented. Where a number is quoted, its source is given.

## 1. The original work by the dataset creators

| Work | What it did | Published finding |
|---|---|---|
| Street, Wolberg & Mangasarian (1993), *Proc. SPIE* 1905:861–870 | Introduced the interactive image-analysis system (snake contours) and the 30 nuclear features; used a linear-programming classifier | Describes the feature-extraction method used to create the dataset |
| Dataset description (`wdbc.names` / WDBC.doc, UW CS) | Multisurface Method-Tree (MSM-T, Bennett 1992); exhaustive search over 1–4 features and 1–3 separating planes | The best single separating plane used **3 features (mean texture, worst area, worst smoothness)**, with an estimated accuracy of **97.5%** from repeated 10-fold CV. The same file reports that the diagnostic system correctly classified **176 consecutive new patients** as of November 1995. |
| Wolberg, Street & Mangasarian (1994), *Cancer Letters* 77:163–171 | Machine-learning techniques for FNA diagnosis | Clinical-application report by the same group |
| Mangasarian, Street & Wolberg (1995), *Operations Research* 43(4):570–577 | LP-based diagnosis and prognosis | Extends the approach to prognosis (recurrence) |

Sources: https://ftp1.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.doc ;
https://pages.cs.wisc.edu/~olvi/uwmp/cancer.html ; https://doi.org/10.1117/12.148698

**Relevance.** The original authors found that **a very small number of features in a linear decision surface** already
gives high cross-validated accuracy. That is a strong baseline, and it explains why complex models offer little extra benefit here.

## 2. The dataset as an ML benchmark

The dataset ships with scikit-learn as `load_breast_cancer` and is one of the most widely used teaching benchmarks for binary
classification. Many tutorials and papers report accuracies in the mid- to high-90s. These reports often differ in:

- the split (a single split or CV), and whether hyperparameters were tuned on the evaluation data;
- the metric (accuracy alone hides false negatives when classes are imbalanced);
- the class orientation (scikit-learn's 0 = malignant is a frequent source of inverted precision and recall).

**[Judgement]** Because of this heterogeneity, headline accuracies from different sources cannot be compared directly.
This project therefore reports nested-CV estimates, a single held-out evaluation with confidence intervals, and explicit
class orientation.

## 3. Methodological references used

| Topic | Reference | How it is used |
|---|---|---|
| Reporting of prediction-model studies | Collins et al. (2024), TRIPOD+AI, *BMJ* 385:e078378 | Structure of [METHODOLOGY.md](METHODOLOGY.md) |
| Selection bias from tuning in CV | Varma & Simon (2006), *BMC Bioinformatics* 7:91 | Nested CV |
| Model explanation | Lundberg & Lee (2017), SHAP | Local and global attributions |
| Calibration and permutation importance | scikit-learn User Guide | Platt scaling, calibration curves, permutation importance |
| Regulatory context | FDA (2022), Clinical Decision Support Software guidance | Scope limits and disclaimers |

## 4. Where this project's numbers sit (this project)

- Nested CV accuracy of the selected SVM: 0.970 (mean over 10 outer folds). Logistic Regression: 0.976.
- Held-out test accuracy of the selected SVM: 0.982, from 114 samples (95% bootstrap CI 0.956–1.0).

These figures are in the same range as the original authors' 97.5% CV estimate. Every interval overlaps, so the
project makes **no claim** of improving on prior work.
