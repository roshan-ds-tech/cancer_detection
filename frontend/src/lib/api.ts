import type {
  ApiErrorBody,
  BatchResponse,
  DatasetSummary,
  EdaResponse,
  ExperimentsResponse,
  FeaturesResponse,
  Health,
  ImportanceResponse,
  MethodologyResponse,
  MetricsResponse,
  ModelsResponse,
  PredictResponse,
  RecordsResponse,
  RocResponse,
  SamplePredictResponse,
} from './types'

// Same-origin by default (Vite dev proxy). Override with VITE_API_BASE for other setups.
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  status: number
  code: string
  details: unknown
  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'network_error', 'Cannot reach the OncoSense API. Is the backend running on port 8000?')
  }
  if (!res.ok) {
    let body: ApiErrorBody | null = null
    try {
      body = (await res.json()) as ApiErrorBody
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(
      res.status,
      body?.error?.code ?? 'http_error',
      body?.error?.message ?? `Request failed (${res.status})`,
      body?.error?.details,
    )
  }
  return (await res.json()) as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  datasetSummary: () => request<DatasetSummary>('/api/dataset/summary'),
  features: () => request<FeaturesResponse>('/api/dataset/features'),
  eda: () => request<EdaResponse>('/api/dataset/eda'),
  records: (limit: number, offset: number) =>
    request<RecordsResponse>(`/api/dataset/records?limit=${limit}&offset=${offset}`),
  models: () => request<ModelsResponse>('/api/models'),
  metrics: () => request<MetricsResponse>('/api/models/metrics'),
  roc: () => request<RocResponse>('/api/models/roc-data'),
  importance: () => request<ImportanceResponse>('/api/models/feature-importance'),
  experiments: () => request<ExperimentsResponse>('/api/experiments'),
  methodology: () => request<MethodologyResponse>('/api/research/methodology'),
  demoSamples: () =>
    request<{ split: string; note: string; samples: { sample_id: number; features: Record<string, number> }[] }>(
      '/api/demo/samples',
    ),
  predict: (features: Record<string, number>) =>
    request<PredictResponse>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, explain: true }),
    }),
  predictSample: (id: number) =>
    request<SamplePredictResponse>(`/api/predict/sample/${id}`, { method: 'POST' }),
  predictBatch: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<BatchResponse>('/api/predict/batch', { method: 'POST', body: fd })
  },
  templateUrl: `${API_BASE}/api/predict/template.csv`,
  figureUrl: (name: string) => `${API_BASE}/static/figures/${name}`,
}
