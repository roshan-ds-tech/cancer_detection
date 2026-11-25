"""Train and compare all models from the terminal:  python train.py"""
import training

summary = training.train_models()
print(f"Trained on {summary['train_rows']} rows, tested on {summary['test_rows']} rows "
      f"({summary['duration_seconds']} s)\n")
print(f"{'Model':<24}{'CV accuracy':>16}{'Test acc':>10}{'Recall':>8}{'ROC-AUC':>9}  Best parameters")
for r in summary["results"]:
    cv = f"{r['cv_accuracy']:.3f} ± {r['cv_std']:.3f}"
    print(f"{r['model']:<24}{cv:>16}{r['accuracy']:>10.3f}{r['recall']:>8.3f}{r['roc_auc']:>9.3f}  {r['best_params']}")
print(f"\nBest model (highest CV accuracy): {summary['best_model']}")
