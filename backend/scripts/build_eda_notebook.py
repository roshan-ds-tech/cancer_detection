"""Generate and execute notebooks/EDA.ipynb from the reusable analysis module.

Usage (from backend/):  python scripts/build_eda_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# OncoSense AI — Exploratory Data Analysis\n\n"
       "> **Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.**\n\n"
       "Dataset: Breast Cancer Wisconsin (Diagnostic), Wolberg, Street & Mangasarian, UCI ML Repository, "
       "CC BY 4.0, doi:10.24432/C5DW2B. Loaded via `sklearn.datasets.load_breast_cancer`.\n\n"
       "Integrity checks use all 569 rows. **Distributional statistics use the training split only** so that the "
       "held-out test set stays unseen. All code lives in `app/ml/eda.py`; this notebook only presents it."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from app.ml import data as D\nfrom app.ml.eda import compute_eda\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)\n"
         "X, y = D.load_dataset()\nprint(X.shape, '| malignant =', int(y.sum()), '| benign =', int((1-y).sum()))"),
    md("## 1. Dimensions, feature names and class orientation\n"
       "scikit-learn encodes 0 = malignant; OncoSense re-encodes so **1 = malignant (positive class)**."),
    code("pd.DataFrame([s.__dict__ for s in D.feature_specs()])[['name','measurement','statistic','unit']]"),
    md("## 2. Integrity: missing values, duplicates, non-finite and zero values"),
    code("eda = compute_eda()\ni = eda['integrity']\n{k: i[k] for k in ['n_rows','n_features','missing_values_total','non_finite_values','duplicate_rows','negative_values','class_counts','split']}"),
    code("pd.Series(i['zero_values_by_feature'], name='rows with value 0')  # concavity = 0 occurs for smooth contours"),
    md("## 3. Descriptive statistics and class-wise comparison (training split)"),
    code("st = pd.DataFrame(eda['feature_stats']).set_index('feature')\n"
         "st[['mean','std','min','50%','max','skew','iqr_outliers','benign_mean','malignant_mean','cohens_d','univariate_auc']].sort_values('univariate_auc', ascending=False).round(4)"),
    md("## 4. Class distribution"),
    code("cc = i['class_counts']; plt.bar(cc.keys(), cc.values(), color=['#7c3aed','#3b82f6']); plt.title('Class distribution'); plt.show()"),
    md("## 5. Correlation and multicollinearity"),
    code("C = pd.DataFrame(eda['correlation']['matrix'], index=eda['correlation']['features'], columns=eda['correlation']['features'])\n"
         "fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)\n"
         "ax.set_xticks(range(30), C.columns, rotation=90, fontsize=6); ax.set_yticks(range(30), C.index, fontsize=6); fig.colorbar(im); plt.show()"),
    code("pd.DataFrame(eda['high_correlation_pairs']).head(15)"),
    code("pd.DataFrame(eda['vif']).head(10).round(1)  # VIF > 10 indicates severe multicollinearity"),
    md("## 6. Distributions and box plots (mean features)"),
    code("Xtr, Xte, ytr, yte = D.split_dataset(X, y)\nfeats = [f for f in D.FEATURE_NAMES if f.startswith('mean_')]\n"
         "fig, axes = plt.subplots(2, 5, figsize=(16, 6))\n"
         "for ax, f in zip(axes.ravel(), feats):\n"
         "    ax.hist(Xtr.loc[ytr==0, f], bins=25, alpha=.6, label='benign', color='#3b82f6')\n"
         "    ax.hist(Xtr.loc[ytr==1, f], bins=25, alpha=.6, label='malignant', color='#7c3aed'); ax.set_title(f, fontsize=9)\n"
         "axes[0,0].legend(); plt.tight_layout(); plt.show()"),
    code("fig, axes = plt.subplots(2, 5, figsize=(16, 6))\n"
         "for ax, f in zip(axes.ravel(), feats):\n"
         "    ax.boxplot([Xtr.loc[ytr==0, f], Xtr.loc[ytr==1, f]], tick_labels=['benign','malignant']); ax.set_title(f, fontsize=9)\n"
         "plt.tight_layout(); plt.show()"),
    md("## 7. Outliers\nIQR-rule counts per feature (training split). Outliers are **kept**: they are plausible "
       "biological/measurement extremes, and removing them would bias evaluation. Scaling is learned inside pipelines."),
    code("st['iqr_outliers'].sort_values(ascending=False).head(10)"),
    md("## 8. Feature scaling requirement"),
    code("print(eda['scaling']['note']); print('orders of magnitude between feature means:', round(eda['scaling']['orders_of_magnitude_between_feature_means'], 2))"),
    md("## 9. PCA — exploratory only (not used by any model)"),
    code("P = pd.DataFrame(eda['pca']['points']); evr = eda['pca']['explained_variance_ratio']\n"
         "for lab, c in [('benign','#3b82f6'), ('malignant','#7c3aed')]:\n"
         "    q = P[P.label==lab]; plt.scatter(q.pc1, q.pc2, s=8, alpha=.6, c=c, label=lab)\n"
         "plt.xlabel(f'PC1 ({evr[0]:.1%})'); plt.ylabel(f'PC2 ({evr[1]:.1%})'); plt.legend(); plt.title('PCA (exploratory)'); plt.show()\n"
         "print('cumulative variance, first 5 PCs:', np.cumsum(evr[:5]).round(3))"),
    md("## 10. Model-based feature importance\nSee the trained-model artifacts (`artifacts/explainability.json`), "
       "Explainable AI page, and `docs/MODEL_EVALUATION.md`. Univariate AUCs above are a model-free complement."),
    code("import json\nexp = json.load(open('../artifacts/explainability.json'))\npd.DataFrame(exp['shap_global']).head(10).round(4)"),
    md("## Observations\n"
       "* No missing values, no duplicate rows, no negative values; zeros occur only in concavity / concave-points features.\n"
       "* Moderate class imbalance (37.3% malignant) → report recall, specificity, AUC, not accuracy alone.\n"
       "* Severe multicollinearity among size features (radius/perimeter/area; VIF in the thousands) → prefer regularised models; interpret attributions cautiously.\n"
       "* Size, concavity and concave-point features separate the classes strongly on their own (univariate AUC > 0.9); "
       "fractal dimension, smoothness SE and texture SE barely do.\n"
       "* Feature scales differ by several orders of magnitude → standardisation needed for LR/SVM/KNN (done inside pipelines)."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out_dir = ROOT / "notebooks"
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out_dir)}})
nbf.write(nb, out_dir / "EDA.ipynb")
print("wrote", out_dir / "EDA.ipynb")
