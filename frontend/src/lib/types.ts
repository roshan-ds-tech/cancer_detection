// Types mirror the FastAPI response schemas (backend/app/api/routes.py, app/schemas).

export type ClassLabel = 'malignant' | 'benign'

export interface Health {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  model_version: string | null
  load_error: string | null
  disclaimer: string
}

export interface FeatureSpec {
  name: string
  sklearn_name: string
  measurement: string
  statistic: 'mean' | 'se' | 'worst'
  description: string
  unit: string
  train_min: number
  train_max: number
  train_median: number
  input_min: number
}

export interface FeaturesResponse {
  features: FeatureSpec[]
  missing_value_policy: string
  measurement_note: string
}

export interface FeatureStat {
  feature: string
  mean: number
  std: number
  min: number
  '25%': number
  '50%': number
  '75%': number
  max: number
  skew: number
  iqr_outliers: number
  benign_mean: number
  malignant_mean: number
  benign_median: number
  malignant_median: number
  cohens_d: number
  univariate_auc: number
}

export interface DatasetSummary {
  dataset: {
    name: string
    uci_url: string
    doi: string
    license: string
    fingerprint: string
    n_samples: number
    n_train: number
    n_test: number
    train_class_counts: Record<ClassLabel, number>
    test_class_counts: Record<ClassLabel, number>
  }
  integrity: {
    n_rows: number
    n_features: number
    missing_values_total: number
    non_finite_values: number
    duplicate_rows: number
    negative_values: number
    zero_values_by_feature: Record<string, number>
    class_counts: Record<ClassLabel, number>
    class_proportion_malignant: number
  } | null
  feature_stats: FeatureStat[] | null
  scope_note: string | null
  scaling: { note: string; orders_of_magnitude_between_feature_means: number } | null
}

export interface HistBin { bin_start: number; bin_end: number; benign: number; malignant: number }
export interface BoxStat {
  min: number; whisker_low: number; q1: number; median: number; q3: number; whisker_high: number; max: number; n: number
}

export interface EdaResponse {
  histograms: Record<string, HistBin[]>
  boxplots: Record<string, Record<ClassLabel, BoxStat>>
  correlation: { features: string[]; matrix: number[][] }
  high_correlation_pairs: { a: string; b: string; r: number }[]
  vif: { feature: string; vif: number }[]
  pca: {
    note: string
    explained_variance_ratio: number[]
    points: { pc1: number; pc2: number; label: ClassLabel; sample_id: number }[]
  }
  scope_note: string
}

export interface RecordsResponse {
  split: string
  total: number
  offset: number
  limit: number
  records: { sample_id: number; label: ClassLabel; features: Record<string, number> }[]
}

export interface MetricSummary { mean: number; std: number; se: number; ci95: [number, number]; folds: number[] }
export type CvMetrics = Record<'accuracy' | 'precision' | 'recall' | 'specificity' | 'f1' | 'roc_auc' | 'brier', MetricSummary>

export interface Confusion { tn: number; fp: number; fn: number; tp: number }
export interface TestMetrics {
  accuracy: number
  precision: number
  recall: number
  specificity: number
  f1: number
  roc_auc: number
  pr_auc: number
  brier: number
  threshold: number
  confusion_matrix: Confusion
}

export interface ModelSummary {
  key: string
  name: string
  status: 'completed' | 'failed'
  error: string | null
  notes: string
  tags: string[]
  best_params: Record<string, string | number | null> | null
  param_grid: Record<string, string[]>
  training_seconds: number | null
  cv_seconds: number | null
  model_size_bytes: number | null
  inference: { single_row_ms_median: number; batch_per_row_ms: number; batch_rows: number } | null
  complexity_rank: number
  experiment_id: string
  cv_scheme: string | null
  selected: boolean
}

export interface ModelsResponse {
  selected_key: string
  model_version: string
  selection_rationale: string
  selection_rule: string
  calibration: { oof_brier_uncalibrated: number; oof_brier_sigmoid_calibrated: number; applied: string; rule: string }
  threshold: { operating_threshold: number; rule: string; target_recall: number; achieved_oof_recall: number; achieved_oof_specificity: number }
  training_config: Record<string, string | number | boolean>
  library_versions: Record<string, string>
  models: ModelSummary[]
}

export interface ReportRow { precision: number; recall: number; 'f1-score': number; support: number }

export interface ModelMetrics {
  key: string
  name: string
  status: 'completed' | 'failed'
  error: string | null
  cv: CvMetrics | null
  test: TestMetrics | null
  test_ci95: Record<string, [number, number]> | null
  classification_report: Record<string, ReportRow | number> | null
  test_at_operating_threshold: TestMetrics | null
  test_at_operating_threshold_ci95: Record<string, [number, number]> | null
}

export interface MetricsResponse {
  notes: { cv: string; test: string }
  selected_key: string
  operating_threshold: number
  models: ModelMetrics[]
  selected_test_errors: { sample_id: number; true_label: ClassLabel; predicted_label: ClassLabel; probability_malignant: number; error_type: string }[]
  learning_curve: { train_size: number; train_auc_mean: number; train_auc_std: number; val_auc_mean: number; val_auc_std: number }[]
}

export interface SweepPoint {
  threshold: number; recall: number; specificity: number; precision: number; f1: number; accuracy: number
  tn: number; fp: number; fn: number; tp: number
}

export interface RocResponse {
  split: string
  selected_key: string
  oof_threshold_sweep: SweepPoint[]
  oof_calibration_curve: { mean_predicted: number; fraction_positive: number }[]
  models: {
    key: string
    name: string
    status: string
    curves: {
      roc: { fpr: number; tpr: number }[]
      pr: { recall: number; precision: number }[]
      calibration: { mean_predicted: number; fraction_positive: number }[]
    } | null
    threshold_sweep: SweepPoint[] | null
  }[]
}

export interface ImportanceResponse {
  model_version: string
  selected_key: string
  permutation_importance: Record<string, { feature: string; importance_mean: number; importance_std: number }[]>
  native_importance: Record<string, { kind: string; description: string; values: { feature: string; value: number }[] } | null>
  shap_global: { feature: string; mean_abs_shap: number }[]
  shap_method: string
  correlation: { features: string[]; matrix: number[][] }
  caveat: string
}

export interface Contribution { feature: string; value: number; contribution: number; direction: string }

export interface DistributionCheck {
  status: 'ok' | 'warning' | 'abstain'
  mahalanobis_distance: number
  mahalanobis_p99_train: number
  mahalanobis_max_train: number
  flagged_features: { feature: string; value: number; train_min: number; train_max: number; robust_z: number; outside_training_range: boolean; severity: 'unusual' | 'extreme' }[]
  messages: string[]
  note: string
}

export interface PredictResponse {
  disclaimer: string
  model_version: string
  model_name: string
  input_validation: 'valid'
  abstained: boolean
  abstain_reason: string[]
  prediction: { predicted_class: ClassLabel; probability_malignant: number; operating_threshold: number; probability_note: string } | null
  distribution_check: DistributionCheck
  explanation: {
    method: string
    output_space: string
    base_value: number
    sum_of_contributions: number
    top_contributions: Contribution[]
    all_contributions: Contribution[]
  } | null
  limitations: string[]
}

export interface SamplePredictResponse extends PredictResponse {
  sample_id: number
  features: Record<string, number>
  ground_truth: ClassLabel
}

export interface BatchRow {
  row: number
  sample_id: string | null
  status: 'predicted' | 'warning' | 'abstained' | 'invalid'
  predicted_class: ClassLabel | null
  probability_malignant: number | null
  errors: string[]
  warnings: string[]
}

export interface BatchResponse {
  disclaimer: string
  model_version: string
  operating_threshold: number
  summary: Record<string, number>
  rows: BatchRow[]
}

export interface ExperimentsResponse {
  runs: { run_id: string; created_at: string; dataset_version: string; seed: number; selected_model: string; model_version: string; selection_rule: string; library_versions: Record<string, string>; duration_seconds: number }[]
  experiments: { experiment_id: string; run_id: string; algorithm: string; status: string; error: string | null; hyperparameters: Record<string, unknown> | null; seed: number; cv_metrics: Record<string, number> | null; test_metrics: Record<string, number> | null; training_seconds: number | null; dataset_version: string; artifact_path: string | null; created_at: string }[]
}

export interface MethodologyResponse {
  disclaimer: string
  model_version: string
  sections: { id: string; title: string; body: string[] }[]
  citations: { id: string; text: string; url: string | null; doi?: string }[]
}

export interface ApiErrorBody { error: { code: string; message: string; details: unknown } }
