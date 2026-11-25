export interface FeatureInfo {
  key: string
  label: string
  help: string
  column: string
  min: number
  max: number
  mean: number
  std: number
  benign_mean: number
  malignant_mean: number
  benign_example: number
  malignant_example: number
  train_mean: number
}

export interface DatasetSummary {
  name: string
  source: string
  total: number
  benign: number
  malignant: number
  missing_values: number
  duplicate_rows: number
  train_rows: number
  test_rows: number
  features: FeatureInfo[]
}

export interface Eda {
  histograms: Record<string, { bin: number; benign: number; malignant: number }[]>
  correlation: { features: string[]; matrix: number[][] }
  scatter: { radius: number; texture: number; diagnosis: string }[]
}

export interface RecordsPage {
  total: number
  limit: number
  offset: number
  rows: (Record<string, number> & { id: number; diagnosis: string })[]
}

export interface ModelResult {
  model: string
  best_params: Record<string, string | number | null>
  cv_accuracy: number
  cv_std: number
  accuracy: number
  precision: number
  recall: number
  specificity: number
  f1: number
  roc_auc: number
  confusion_matrix: number[][]
  roc_curve: { fpr: number; tpr: number }[]
  train_seconds: number
}

export interface ImportanceRow { feature: string; value: number; std?: number }

export interface Results {
  trained_at: string
  train_rows: number
  test_rows: number
  best_model: string
  duration_seconds: number
  results: ModelResult[]
  importance: { permutation: ImportanceRow[]; random_forest: ImportanceRow[]; logistic_regression: ImportanceRow[] }
}

export interface HistoryRun {
  run: number
  trained_at: string
  best_model: string
  cv_accuracy: number
  test_accuracy: number
  test_recall: number
  duration_seconds: number
}

export type Level = 'Low' | 'Medium' | 'High'

export interface Contribution { feature: string; value: number; contribution: number }

export interface Prediction {
  prediction: 'Benign' | 'Malignant'
  malignant_probability: number
  level: Level
  unusual_values: string[]
  explanation: { base_value: number; items: Contribution[] } | null
  model: string
}

export interface SamplePrediction extends Prediction {
  id: number
  values: Record<string, number>
  actual: 'Benign' | 'Malignant'
  correct: boolean
}

export interface BatchRow {
  row: number
  id: string
  status: 'ok' | 'invalid'
  errors: string[]
  prediction?: 'Benign' | 'Malignant'
  malignant_probability?: number
  level?: Level
  unusual_values?: string[]
}

export interface BatchResult {
  model: string
  summary: { total: number; valid: number; invalid: number; benign: number; malignant: number }
  rows: BatchRow[]
}
