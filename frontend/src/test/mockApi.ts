// Fixtures are real responses captured from the running API (see TESTING_REPORT.md).
import { vi } from 'vitest'
import dataset_eda from './fixtures/dataset_eda.json'
import dataset_features from './fixtures/dataset_features.json'
import dataset_records from './fixtures/dataset_records.json'
import dataset_summary from './fixtures/dataset_summary.json'
import demo_samples from './fixtures/demo_samples.json'
import experiments from './fixtures/experiments.json'
import health from './fixtures/health.json'
import models from './fixtures/models.json'
import models_feature_importance from './fixtures/models_feature_importance.json'
import models_metrics from './fixtures/models_metrics.json'
import models_roc_data from './fixtures/models_roc_data.json'
import predict_sample from './fixtures/predict_sample.json'
import research_methodology from './fixtures/research_methodology.json'

export const fixtures = {
  '/api/health': health,
  '/api/dataset/summary': dataset_summary,
  '/api/dataset/features': dataset_features,
  '/api/dataset/eda': dataset_eda,
  '/api/dataset/records': dataset_records,
  '/api/models': models,
  '/api/models/metrics': models_metrics,
  '/api/models/roc-data': models_roc_data,
  '/api/models/feature-importance': models_feature_importance,
  '/api/experiments': experiments,
  '/api/research/methodology': research_methodology,
  '/api/demo/samples': demo_samples,
  '/api/predict/sample': predict_sample,
  '/api/predict': predict_sample,
} as Record<string, unknown>

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))

export function mockFetch(overrides: Record<string, () => Promise<Response>> = {}) {
  const fn = vi.fn((input: RequestInfo | URL) => {
    const path = String(input).split('?')[0]
    for (const [k, v] of Object.entries(overrides)) if (path.startsWith(k)) return v()
    const key = Object.keys(fixtures)
      .filter((k) => path === k || path.startsWith(k + '/'))
      .sort((a, b) => b.length - a.length)[0]
    if (!key) return json({ error: { code: 'not_found', message: 'not mocked', details: null } }, 404)
    return json(fixtures[key])
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

export { json }
