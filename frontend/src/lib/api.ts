import type {
  BatchResult, DatasetSummary, Eda, HistoryRun, Prediction, RecordsPage, Results, SamplePrediction,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError(0, 'Cannot reach the API. Is the backend running on port 8000?')
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') message = body.detail
      else if (Array.isArray(body.detail)) message = 'Some values are missing or invalid.'
    } catch { /* not JSON */ }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}

const post = (body?: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
})

export const api = {
  health: () => request<{ status: string; best_model: string; trained_at: string }>('/api/health'),
  summary: () => request<DatasetSummary>('/api/dataset/summary'),
  eda: () => request<Eda>('/api/dataset/eda'),
  records: (limit: number, offset: number) => request<RecordsPage>(`/api/dataset/records?limit=${limit}&offset=${offset}`),
  results: () => request<Results>('/api/models/results'),
  train: () => request<Results>('/api/models/train', post()),
  history: () => request<HistoryRun[]>('/api/models/history'),
  samples: () => request<{ id: number; values: Record<string, number> }[]>('/api/samples'),
  predict: (values: Record<string, number>) => request<Prediction>('/api/predict', post(values)),
  predictSample: (id: number) => request<SamplePrediction>(`/api/predict/sample/${id}`, post()),
  predictBatch: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<BatchResult>('/api/predict/batch', { method: 'POST', body: form })
  },
  templateUrl: '/api/predict/template.csv',
}
