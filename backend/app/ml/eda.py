"""Reproducible exploratory data analysis.

Dataset-level integrity checks (shape, missingness, duplicates, class balance) use the
full dataset. Every *distributional* statistic that could plausibly inform modelling
(correlations, class-wise comparisons, PCA, outliers) is computed on the TRAINING split
only, so the hold-out test set stays unseen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.core.config import RANDOM_SEED
from app.ml import data as D


def compute_eda(seed: int = RANDOM_SEED) -> dict:
    X, y = D.load_dataset()
    X_train, X_test, y_train, y_test = D.split_dataset(X, y, seed=seed)
    labels = y_train.map(D.CLASS_MAPPING)

    integrity = {
        "n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "missing_values_total": int(X.isna().sum().sum()),
        "missing_by_feature": {c: int(v) for c, v in X.isna().sum().items()},
        "non_finite_values": int((~np.isfinite(X.to_numpy())).sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "negative_values": int((X < 0).sum().sum()),
        "zero_values_by_feature": {c: int(v) for c, v in (X == 0).sum().items() if v > 0},
        "class_counts": {"malignant": int(y.sum()), "benign": int((1 - y).sum())},
        "class_proportion_malignant": float(y.mean()),
        "split": {"train": int(len(X_train)), "test": int(len(X_test)),
                  "train_malignant": int(y_train.sum()), "test_malignant": int(y_test.sum())},
    }

    describe = X_train.describe().T
    by_class = X_train.groupby(labels).agg(["mean", "std", "median"])
    stats = []
    for f in D.FEATURE_NAMES:
        x = X_train[f]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        m_b, m_m = by_class.loc["benign", (f, "mean")], by_class.loc["malignant", (f, "mean")]
        s_b, s_m = by_class.loc["benign", (f, "std")], by_class.loc["malignant", (f, "std")]
        pooled = np.sqrt((s_b ** 2 + s_m ** 2) / 2)
        auc = roc_auc_score(y_train, x)
        stats.append({
            "feature": f,
            **{k: float(describe.loc[f, k]) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]},
            "skew": float(x.skew()),
            "iqr_outliers": n_out,
            "benign_mean": float(m_b), "malignant_mean": float(m_m),
            "benign_median": float(by_class.loc["benign", (f, "median")]),
            "malignant_median": float(by_class.loc["malignant", (f, "median")]),
            "cohens_d": float((m_m - m_b) / pooled) if pooled else 0.0,
            "univariate_auc": float(auc),
        })

    hist = {}
    for f in D.FEATURE_NAMES:
        edges = np.histogram_bin_edges(X_train[f], bins=20)
        hb, _ = np.histogram(X_train.loc[labels == "benign", f], bins=edges)
        hm, _ = np.histogram(X_train.loc[labels == "malignant", f], bins=edges)
        hist[f] = [{"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]),
                    "benign": int(hb[i]), "malignant": int(hm[i])} for i in range(len(hb))]

    box = {}
    for f in D.FEATURE_NAMES:
        box[f] = {}
        for cls in ["benign", "malignant"]:
            x = X_train.loc[labels == cls, f]
            q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            lo = x[x >= q1 - 1.5 * iqr].min()
            hi = x[x <= q3 + 1.5 * iqr].max()
            box[f][cls] = {"min": float(x.min()), "whisker_low": float(lo), "q1": float(q1),
                           "median": float(med), "q3": float(q3), "whisker_high": float(hi),
                           "max": float(x.max()), "n": int(len(x))}

    corr = X_train.corr()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.9:
                pairs.append({"a": cols[i], "b": cols[j], "r": float(r)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    Z = StandardScaler().fit_transform(X_train)
    inv = np.linalg.pinv(np.corrcoef(Z, rowvar=False))
    vif = sorted([{"feature": f, "vif": float(v)} for f, v in zip(cols, np.diag(inv))],
                 key=lambda d: -d["vif"])

    pca = PCA(n_components=10, random_state=seed).fit(Z)
    coords = pca.transform(Z)[:, :2]
    pca_doc = {
        "note": "Exploratory only. PCA is fit on standardised TRAINING data and is not part of "
                "any model pipeline.",
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "points": [{"pc1": float(a), "pc2": float(b), "label": lab, "sample_id": int(i)}
                   for (a, b), lab, i in zip(coords, labels, X_train.index)],
    }

    ranges = X.max() / X.replace(0, np.nan).min()
    scaling = {
        "min_feature_mean": float(describe["mean"].min()),
        "max_feature_mean": float(describe["mean"].max()),
        "orders_of_magnitude_between_feature_means": float(
            np.log10(describe["mean"].max() / describe["mean"].min())),
        "note": "Feature means span several orders of magnitude (e.g. area vs fractal "
                "dimension), so distance- and margin-based models (KNN, SVM) and regularised "
                "logistic regression need standardisation; tree models do not.",
        "max_to_min_ratio_by_feature": {k: float(v) for k, v in ranges.items()},
    }

    return {
        "integrity": integrity, "feature_stats": stats, "histograms": hist, "boxplots": box,
        "correlation": {"features": cols, "matrix": corr.round(3).to_numpy().tolist()},
        "high_correlation_pairs": pairs, "vif": vif, "pca": pca_doc, "scaling": scaling,
        "scope_note": "Integrity checks use all 569 rows. All distributional statistics use the "
                      f"{len(X_train)}-row training split only.",
    }


def eda_figures(eda: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made.append(name)

    cc = eda["integrity"]["class_counts"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["benign", "malignant"], [cc["benign"], cc["malignant"]], color=["#3b82f6", "#7c3aed"])
    for i, v in enumerate([cc["benign"], cc["malignant"]]):
        ax.text(i, v + 5, str(v), ha="center")
    ax.set(title="Class distribution (all 569 samples)", ylabel="count")
    save(fig, "eda_class_distribution.png")

    m = np.array(eda["correlation"]["matrix"])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    f = eda["correlation"]["features"]
    ax.set_xticks(range(len(f)), f, rotation=90, fontsize=6)
    ax.set_yticks(range(len(f)), f, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Pearson correlation (training split)")
    save(fig, "eda_correlation_heatmap.png")

    mean_feats = [s["feature"] for s in eda["feature_stats"] if s["feature"].startswith("mean_")]
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.5))
    for ax, feat in zip(axes.ravel(), mean_feats):
        bins = eda["histograms"][feat]
        x = [b["bin_start"] for b in bins]
        w = bins[0]["bin_end"] - bins[0]["bin_start"]
        ax.bar(x, [b["benign"] for b in bins], width=w, align="edge", alpha=0.6, color="#3b82f6", label="benign")
        ax.bar(x, [b["malignant"] for b in bins], width=w, align="edge", alpha=0.6, color="#7c3aed", label="malignant")
        ax.set_title(feat, fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Feature histograms by class (training split)")
    save(fig, "eda_histograms_mean_features.png")

    fig, axes = plt.subplots(2, 5, figsize=(15, 5.5))
    for ax, feat in zip(axes.ravel(), mean_feats):
        bx = eda["boxplots"][feat]
        stats = [{"med": bx[c]["median"], "q1": bx[c]["q1"], "q3": bx[c]["q3"],
                  "whislo": bx[c]["whisker_low"], "whishi": bx[c]["whisker_high"], "fliers": [],
                  "label": c} for c in ["benign", "malignant"]]
        ax.bxp(stats, showfliers=False)
        ax.set_title(feat, fontsize=8)
    fig.suptitle("Box plots by class (training split, whiskers = 1.5 IQR)")
    save(fig, "eda_boxplots_mean_features.png")

    pts = eda["pca"]["points"]
    evr = eda["pca"]["explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, c in [("benign", "#3b82f6"), ("malignant", "#7c3aed")]:
        p = [q for q in pts if q["label"] == lab]
        ax.scatter([q["pc1"] for q in p], [q["pc2"] for q in p], s=10, alpha=0.6, c=c, label=lab)
    ax.set(xlabel=f"PC1 ({evr[0]:.1%})", ylabel=f"PC2 ({evr[1]:.1%})",
           title="PCA — exploratory only (training split)")
    ax.legend()
    save(fig, "eda_pca.png")

    st = sorted(eda["feature_stats"], key=lambda s: s["univariate_auc"])
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh([s["feature"] for s in st], [s["univariate_auc"] for s in st], color="#0d9488")
    ax.axvline(0.5, color="#94a3b8", ls="--")
    ax.set(xlabel="Univariate ROC-AUC (training split)", title="Single-feature separability")
    save(fig, "eda_univariate_auc.png")
    return made
