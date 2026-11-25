# Ethics and Limitations

> **Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.**

## 1. Intended use and non-use

| Intended | Not intended |
|---|---|
| Teaching ML methodology, evaluation and explainability | Diagnosing, screening, triaging or advising any person |
| Demonstrating leakage-aware model selection | Use as, or marketing as, a medical device or decision-support tool |
| Exploring threshold and uncertainty trade-offs | Processing real patient records |

The app shows the disclaimer in a persistent banner on every page, inside every prediction result, in the batch CSV
export and in the API responses (`disclaimer` field). Outputs are worded as *"predicted dataset class"* and *"model probability for
the 'malignant' dataset class"*. They are never worded as a person's risk.

## 2. Data limitations

- **Sampling bias.** All samples come from a single institution (University of Wisconsin Hospitals) and were collected before 1995. The 37% malignant rate reflects patients referred for FNA, not a screening population.
- **No demographics.** There is no age, ethnicity, sex or breast-density information, so fairness across subgroups **cannot be evaluated**. Performance could differ across groups without anyone being able to detect it from this dataset.
- **Measurement process.** The features depend on a specific digitisation set-up and on operator-initialised contours. Other microscopes, stains, operators or software would shift the distributions.
- **Reference standard.** The dataset does not document how labels were confirmed. Label noise cannot be quantified.
- **Feature availability.** Ordinary users and most clinics cannot obtain these 30 measurements. The manual form says so.

## 3. Model and evaluation limitations

- **No external validation.** All results are internal: nested CV plus one 114-sample hold-out.
- **Small test set.** There are 42 malignant test samples, so one error changes recall by 2.4 points. The confidence intervals are wide, and degenerate where no errors occurred.
- **False-negative risk.** The selected model missed 2 of 42 malignant test samples. One of them (sample 73) received P(malignant) = 0.05. In clinical terms, a missed malignancy can delay treatment. No threshold removes this risk without many false positives.
- **False-positive risk.** Lowering the threshold raises recall but creates false alarms. In practice, false alarms lead to anxiety, more procedures and cost.
- **Calibration.** Probabilities are calibrated to this dataset's prevalence and are not valid at a different prevalence.
- **Distribution shift.** The OOD check is a heuristic based on training-range and Mahalanobis rules. It is not validated and can miss subtle shift.
- **Explainability.** SHAP and permutation importance describe the model's statistical behaviour. Under strong multicollinearity (VIF up to about 4,000), attributions are unstable and can split credit arbitrarily. They are **not causal**.
- **Model choice.** LR and the SVM are statistically tied. A different seed or split could reverse the ranking.

## 4. Regulatory context

Under the FDA's Clinical Decision Support Software guidance (2022), software that processes medical images or signals,
or that provides patient-specific diagnostic outputs, may fall outside the non-device CDS criteria. Such software can then be regulated as a
medical device. OncoSense has no regulatory clearance or approval in any jurisdiction and makes no clinical claim.
Source: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software

## 5. Privacy and data handling

- Only the public, de-identified, CC BY 4.0 dataset is used. No patient registration, records or histories exist.
- Uploaded CSVs are processed **in memory**. They are never written to disk, logged or retained. Request bodies are not logged.
- No inference logging is enabled by default. The SQLite database stores only training-experiment metadata.
- Do not upload real patient data. The app is not designed or authorised for it.

## 6. Safeguards implemented

| Risk | Safeguard |
|---|---|
| Output taken as diagnosis | Disclaimers everywhere; neutral colours (blue/purple, no alarm red); "dataset class" wording; limitations shown with each result |
| Garbage-in predictions | Strict schema (30 required, finite, ≥ 0, no extra fields); no imputation |
| Out-of-range inputs | Warnings, and **abstention** for extreme inputs |
| Label leakage in the demo | Ground truth is hidden until the user clicks "Reveal" |
| Fabricated results | All UI numbers come from API artifacts produced by training; failed experiments are displayed as failed |
| Unsafe model loading | SHA-256 manifest check and path containment before `joblib.load`; uploads are never unpickled |

## 7. Responsible next steps (out of scope here)

These would be required before any clinical consideration: external multi-site validation, prospective
evaluation, subgroup analysis, a documented reference standard, human-factors testing and a regulatory pathway.
