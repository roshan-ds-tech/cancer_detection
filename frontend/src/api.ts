// Small helper for talking to the FastAPI backend.

export type Feature = {
  key: string
  label: string
  help: string
  min: number
  max: number
  benign_avg: number
  malignant_avg: number
  benign_example: number
  malignant_example: number
}

export type Dataset = {
  total: number
  benign: number
  malignant: number
  features: Feature[]
  scatter: { radius: number; texture: number; diagnosis: string }[]
  sample_rows: Record<string, number | string>[]
}

export type ModelResult = {
  model: string
  cv_accuracy: number
  accuracy: number
  precision: number
  recall: number
  f1: number
  confusion_matrix: number[][]
}

export type Results = {
  trained_at: string
  train_rows: number
  test_rows: number
  best_model: string
  results: ModelResult[]
  feature_importance: { feature: string; importance: number }[]
}

export type Prediction = {
  prediction: 'Benign' | 'Malignant'
  malignant_probability: number
  level: 'Low' | 'Medium' | 'High'
  unusual_values: string[]
  model: string
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Server error (${res.status})`)
  return res.json()
}

export const getDataset = () => request<Dataset>('/api/dataset')
export const getResults = () => request<Results>('/api/results')
export const runModels = () => request<Results>('/api/train', {})
export const predict = (values: Record<string, number>) => request<Prediction>('/api/predict', values)

export const percent = (x: number) => `${(x * 100).toFixed(1)}%`
