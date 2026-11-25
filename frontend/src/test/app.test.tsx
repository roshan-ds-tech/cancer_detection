import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { DISCLAIMER } from '../components/Disclaimer'
import { fixtures, json, mockFetch } from './mockApi'

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )

const ROUTES: [string, RegExp][] = [
  ['/', /Transparent, explainable classification/],
  ['/dataset', /Dataset Explorer/],
  ['/models', /Model Performance & Comparison Lab/],
  ['/predict', /Prediction Laboratory/],
  ['/explain', /Explainable AI/],
  ['/experiments', /Experiment Tracker/],
  ['/research', /Research & Methodology/],
]

describe('page rendering', () => {
  it.each(ROUTES)('renders %s with the research disclaimer', async (path, heading) => {
    mockFetch()
    renderAt(path)
    // headers render during loading and again with data, so re-query instead of holding a node
    await waitFor(() => expect(screen.getByRole('heading', { level: 1, name: heading })).toBeInTheDocument(), { timeout: 4000 })
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument(), { timeout: 4000 })
    expect(screen.getAllByTestId('disclaimer')[0]).toHaveTextContent(DISCLAIMER)
  })

  it('shows loading then live values on the landing page (no hardcoded metrics)', async () => {
    mockFetch()
    renderAt('/')
    expect(screen.getByRole('status')).toHaveTextContent(/Loading/)
    const models = fixtures['/api/models'] as { models: { name: string; selected: boolean }[] }
    const sel = models.models.find((m) => m.selected)!.name
    expect(await screen.findByText(sel)).toBeInTheDocument()
  })

  it('model performance page renders data from the metrics API', async () => {
    mockFetch()
    renderAt('/models')
    expect(await screen.findByText(/Algorithm comparison/, {}, { timeout: 4000 })).toBeInTheDocument()
    const table = screen.getByText(/Algorithm comparison/).closest('section')!
    expect(within(table).getByText('Gaussian Naive Bayes')).toBeInTheDocument()
    expect(screen.getAllByText(/Held-out test set/).length).toBeGreaterThan(0)
  })

  it('shows a failed experiment instead of hiding it', async () => {
    const m = structuredClone(fixtures['/api/models']) as { models: { status: string; error: string | null }[] }
    m.models[3].status = 'failed'
    m.models[3].error = 'ValueError: boom'
    // overrides match by prefix, so keep the sub-endpoints on their fixtures
    const fetchMock = mockFetch({
      '/api/models/metrics': () => json(fixtures['/api/models/metrics']),
      '/api/models/roc-data': () => json(fixtures['/api/models/roc-data']),
      '/api/models': () => json(m),
    })
    renderAt('/models')
    expect(await screen.findByText(/Experiment failed: ValueError: boom/, {}, { timeout: 4000 })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalled()
  })
})

describe('error states', () => {
  it('shows a meaningful error and retry when the API is unreachable', async () => {
    mockFetch({ '/api/research': () => Promise.reject(new TypeError('Failed to fetch')) })
    renderAt('/research')
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/Cannot reach the OncoSense API/)
    expect(within(alert).getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })

  it('surfaces 503 model-unavailable errors from the backend', async () => {
    mockFetch({
      '/api/experiments': () =>
        json({ error: { code: 'model_unavailable', message: 'Model artifacts are not loaded.', details: null } }, 503),
    })
    renderAt('/experiments')
    expect(await screen.findByRole('alert')).toHaveTextContent(/Model artifacts are not loaded/)
  })
})

describe('prediction laboratory', () => {
  it('keeps the ground-truth label hidden until revealed', async () => {
    mockFetch()
    const user = userEvent.setup()
    renderAt('/predict')
    const select = await screen.findByLabelText('Sample ID', {}, { timeout: 4000 })
    const sid = (fixtures['/api/predict/sample'] as { sample_id: number }).sample_id
    await user.selectOptions(select, String(sid))
    await user.click(screen.getByRole('button', { name: 'Run model' }))
    const result = await screen.findByTestId('prediction-result')
    expect(within(result).getByTestId('disclaimer')).toHaveTextContent(DISCLAIMER)
    expect(screen.queryByText(/Known dataset label/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Reveal known dataset label/ }))
    expect(screen.getByText(/Known dataset label/)).toBeInTheDocument()
    expect(screen.getByText(/Why the model produced this output/)).toBeInTheDocument()
  })

  it('validates the manual form before calling the API', async () => {
    const fetchMock = mockFetch()
    const user = userEvent.setup()
    renderAt('/predict')
    await user.click(await screen.findByRole('tab', { name: /Manual input/ }))
    const form = await screen.findByRole('form', { name: 'Manual feature input' })
    await user.click(within(form).getByRole('button', { name: 'Run model' }))
    expect(await screen.findByText(/30 field\(s\) need attention/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/predict')).toBe(false)

    await user.click(screen.getByRole('button', { name: /Fill with training medians/ }))
    const radius = screen.getByLabelText('mean radius')
    await user.clear(radius)
    await user.type(radius, '-3')
    await user.click(within(form).getByRole('button', { name: 'Run model' }))
    expect(await screen.findByText('Must be ≥ 0')).toBeInTheDocument()
    expect(radius).toHaveAttribute('aria-invalid', 'true')

    await user.clear(radius)
    await user.type(radius, '14')
    await user.click(within(form).getByRole('button', { name: 'Run model' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/predict')).toBe(true))
    expect(await screen.findByTestId('prediction-result')).toBeInTheDocument()
  })

  it('rejects non-CSV files client-side and displays batch results', async () => {
    mockFetch({
      '/api/predict/batch': () =>
        json({
          disclaimer: DISCLAIMER, model_version: 'v-test', operating_threshold: 0.4,
          summary: { predicted: 1, warning: 0, abstained: 0, invalid: 1, total: 2 },
          rows: [
            { row: 1, sample_id: 'a', status: 'predicted', predicted_class: 'benign', probability_malignant: 0.01, errors: [], warnings: [] },
            { row: 2, sample_id: 'b', status: 'invalid', predicted_class: null, probability_malignant: null, errors: ['mean_area: not a number'], warnings: [] },
          ],
        }),
    })
    const user = userEvent.setup({ applyAccept: false })
    renderAt('/predict')
    await user.click(await screen.findByRole('tab', { name: /CSV batch/ }))
    const input = screen.getByTestId('csv-input')
    await user.upload(input, new File(['x'], 'data.txt', { type: 'text/plain' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/Only .csv files/)
    await user.upload(input, new File(['a,b\n1,2'], 'data.csv', { type: 'text/csv' }))
    await user.click(screen.getByRole('button', { name: /Upload & predict/ }))
    expect(await screen.findByText('Batch results')).toBeInTheDocument()
    expect(screen.getByText(/mean_area: not a number/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Download CSV/ })).toBeInTheDocument()
  })

  it('shows server-side schema errors for a bad CSV', async () => {
    mockFetch({
      '/api/predict/batch': () =>
        json({ error: { code: 'request_error', message: 'CSV columns do not match the model schema', details: { message: 'x', missing_columns: ['mean_radius'], unexpected_columns: ['age'], duplicate_columns: [] } } }, 422),
    })
    const user = userEvent.setup()
    renderAt('/predict')
    await user.click(await screen.findByRole('tab', { name: /CSV batch/ }))
    await user.upload(screen.getByTestId('csv-input'), new File(['a'], 'bad.csv', { type: 'text/csv' }))
    await user.click(screen.getByRole('button', { name: /Upload & predict/ }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/do not match the model schema/)
    expect(alert).toHaveTextContent(/unexpected columns: age/)
  })
})

describe('responsive navigation', () => {
  it('toggles the mobile navigation menu', async () => {
    mockFetch()
    const user = userEvent.setup()
    renderAt('/')
    const btn = screen.getByRole('button', { name: 'Open navigation' })
    expect(btn).toHaveAttribute('aria-expanded', 'false')
    await user.click(btn)
    expect(screen.getByRole('button', { name: 'Close navigation' })).toHaveAttribute('aria-expanded', 'true')
    expect(document.getElementById('mobile-nav')).toBeInTheDocument()
  })
})
