"""Train and compare all models from the command line:  python train.py"""
import model as ml

summary = ml.train_models()
print(f"Trained on {summary['train_rows']} rows, tested on {summary['test_rows']} rows\n")
print(f"{'Model':<25}{'CV accuracy':>12}{'Test accuracy':>15}{'Recall':>9}")
for r in summary["results"]:
    print(f"{r['model']:<25}{r['cv_accuracy']:>12.3f}{r['accuracy']:>15.3f}{r['recall']:>9.3f}")
print(f"\nBest model (highest CV accuracy): {summary['best_model']}")
