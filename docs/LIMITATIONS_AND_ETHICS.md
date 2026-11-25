# Limitations and Ethics

> Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.

The canonical, maintained version of this document is **[ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md)**.
Keeping one copy avoids the two drifting apart. Summary:

- **Data:** single centre, pre-1995, referral-population prevalence (37% malignant), no demographics, so fairness cannot be assessed. The features depend on a specific image-analysis process.
- **Evaluation:** internal validation only (nested CV + 114-sample hold-out). No external or prospective validation. Wide CIs.
- **Errors:** the selected model missed 2 of 42 malignant test samples (one at P = 0.05). Lowering the threshold trades misses for false alarms.
- **Probabilities:** calibrated to this dataset's prevalence only.
- **Explanations:** describe model behaviour, not biology. They are unstable under the dataset's severe multicollinearity.
- **Regulatory:** not a medical device. No clearance. No clinical claims (see the FDA CDS guidance).
- **Privacy:** public de-identified data only. Uploads are processed in memory and never stored. Do not submit real patient data.
