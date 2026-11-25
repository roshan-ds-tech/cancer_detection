"""Train, compare, select and evaluate all candidate models.

Usage (from backend/):
    python scripts/train_models.py            # full run (nested CV 5x2)
    python scripts/train_models.py --quick    # faster smoke run (nested CV 5x1)
"""
import argparse

import _bootstrap  # noqa: F401

from app.core.config import RANDOM_SEED
from app.ml.training import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    res = run_training(seed=args.seed, quick=args.quick, make_figures=not args.no_figures)
    md, mt = res["metadata"], res["metrics"]
    sel = next(m for m in mt["models"] if m["key"] == mt["selected_key"])
    print("\nSelected:", md["selected_model"]["name"], md["selected_model"]["best_params"])
    print("Rationale:", md["selection_rationale"])
    t = sel["test"]
    print(f"Test @0.5: acc={t['accuracy']:.4f} recall={t['recall']:.4f} spec={t['specificity']:.4f} "
          f"auc={t['roc_auc']:.4f} brier={t['brier']:.4f} cm={t['confusion_matrix']}")
    to = sel["test_at_operating_threshold"]
    print(f"Test @{to['threshold']}: acc={to['accuracy']:.4f} recall={to['recall']:.4f} "
          f"spec={to['specificity']:.4f} cm={to['confusion_matrix']}")


if __name__ == "__main__":
    main()
