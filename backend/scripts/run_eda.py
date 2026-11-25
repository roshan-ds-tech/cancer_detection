"""Run the exploratory data analysis and write artifacts/eda.json + EDA figures.

Usage (from backend/):  python scripts/run_eda.py
"""
import _bootstrap  # noqa: F401

from app.core.config import ARTIFACTS_DIR, FIGURES_DIR
from app.ml.artifacts import write_json
from app.ml.eda import compute_eda, eda_figures


def main():
    eda = compute_eda()
    write_json(ARTIFACTS_DIR / "eda.json", eda)
    figs = eda_figures(eda, FIGURES_DIR)
    i = eda["integrity"]
    print(f"rows={i['n_rows']} features={i['n_features']} missing={i['missing_values_total']} "
          f"duplicates={i['duplicate_rows']} classes={i['class_counts']}")
    print(f"high-correlation pairs (|r|>=0.9): {len(eda['high_correlation_pairs'])}")
    print("figures:", ", ".join(figs))


if __name__ == "__main__":
    main()
