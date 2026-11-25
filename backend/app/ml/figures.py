"""Static figures (PNG) generated from real training/evaluation outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

NAVY, TEAL, BLUE, PURPLE, GRAY = "#1e2a4a", "#0d9488", "#3b82f6", "#7c3aed", "#94a3b8"
PALETTE = [NAVY, TEAL, BLUE, PURPLE, "#f59e0b", "#64748b", "#db2777"]


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def training_figures(fig_dir: Path, results, selected_key, y_test, test_probs, learning,
                     perm_sel, shap_ex, shap_X, thr):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    ok = [r for r in results if r["key"] in test_probs]

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, r in zip(PALETTE, ok):
        fpr, tpr, _ = roc_curve(y_test, test_probs[r["key"]])
        ax.plot(fpr, tpr, color=c, lw=2 if r["key"] == selected_key else 1,
                label=f"{r['name']} (AUC {r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall, malignant)",
           title="ROC — held-out test set")
    ax.legend(fontsize=7)
    _save(fig, fig_dir / "roc_test.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, r in zip(PALETTE, ok):
        p, rc, _ = precision_recall_curve(y_test, test_probs[r["key"]])
        ax.plot(rc, p, color=c, label=f"{r['name']} (AP {r['test']['pr_auc']:.3f})")
    ax.set(xlabel="Recall (malignant)", ylabel="Precision", title="Precision–recall — test set")
    ax.legend(fontsize=7)
    _save(fig, fig_dir / "pr_test.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, r in zip(PALETTE, ok):
        fp, mp = calibration_curve(y_test, test_probs[r["key"]], n_bins=8, strategy="quantile")
        ax.plot(mp, fp, marker="o", color=c, label=f"{r['name']} (Brier {r['test']['brier']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY)
    ax.set(xlabel="Mean predicted P(malignant)", ylabel="Observed fraction malignant",
           title="Calibration — test set (n=114, wide uncertainty)")
    ax.legend(fontsize=7)
    _save(fig, fig_dir / "calibration_test.png")

    n = len(ok)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8))
    for ax, r in zip(np.atleast_1d(axes), ok):
        cm = r["test"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, v, ha="center", va="center", color="black")
        ax.set_xticks([0, 1], ["benign", "malig."])
        ax.set_yticks([0, 1], ["benign", "malig."])
        ax.set_title(r["name"].split(" (")[0], fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
    _save(fig, fig_dir / "confusion_matrices_test.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([r["cv"]["roc_auc"]["folds"] for r in ok], tick_labels=[r["name"].split(" (")[0] for r in ok])
    ax.set(ylabel="ROC-AUC (outer folds)", title="Nested CV ROC-AUC distribution (training split)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    _save(fig, fig_dir / "cv_auc_boxplot.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    s = [p["train_size"] for p in learning]
    for key, c, lab in [("train", NAVY, "training"), ("val", TEAL, "validation")]:
        m = np.array([p[f"{key}_auc_mean"] for p in learning])
        sd = np.array([p[f"{key}_auc_std"] for p in learning])
        ax.plot(s, m, color=c, marker="o", label=lab)
        ax.fill_between(s, m - sd, m + sd, color=c, alpha=0.15)
    ax.set(xlabel="Training samples", ylabel="ROC-AUC", title="Learning curve — selected model")
    ax.legend()
    _save(fig, fig_dir / "learning_curve.png")

    top = perm_sel[:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([d["feature"] for d in top], [d["importance_mean"] for d in top],
            xerr=[d["importance_std"] for d in top], color=TEAL)
    ax.set(xlabel="Mean drop in test ROC-AUC when permuted", title="Permutation importance — selected model")
    _save(fig, fig_dir / "permutation_importance.png")

    plt.figure()
    shap.plots.beeswarm(shap_ex, max_display=15, show=False)
    plt.title("SHAP (P(malignant)) — test set")
    _save(plt.gcf(), fig_dir / "shap_beeswarm.png")

    plt.figure()
    shap.plots.waterfall(shap_ex[0], max_display=12, show=False)
    plt.title(f"SHAP waterfall — test sample #{shap_X.index[0]}")
    _save(plt.gcf(), fig_dir / "shap_waterfall_example.png")
