import { Download, Eye, FileUp, Keyboard, ListChecks, Shuffle, Upload } from 'lucide-react'
import { useState } from 'react'
import { Disclaimer } from '../components/Disclaimer'
import { PredictionResult } from '../components/PredictionResult'
import { Badge, Card, Empty, ErrorState, Loading, PageHeader, Tabs } from '../components/ui'
import { api, ApiError } from '../lib/api'
import { LEVEL_STYLES, prob, sig } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { BatchResult, FeatureInfo, Prediction, SamplePrediction } from '../lib/types'

type Tab = 'manual' | 'samples' | 'csv'

export default function Predict() {
  const [tab, setTab] = useState<Tab>('manual')
  const summary = useApi(api.summary)
  return (
    <div>
      <PageHeader
        title="Prediction Lab"
        subtitle="Enter the 10 cell measurements yourself, try samples from the test data, or upload a CSV file with many samples."
      />
      <Tabs<Tab>
        label="Prediction mode"
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'manual', label: 'Manual input', icon: <Keyboard className="h-4 w-4" aria-hidden /> },
          { id: 'samples', label: 'Test samples', icon: <ListChecks className="h-4 w-4" aria-hidden /> },
          { id: 'csv', label: 'CSV batch', icon: <FileUp className="h-4 w-4" aria-hidden /> },
        ]}
      />
      {summary.loading ? <Loading /> : summary.error ? <ErrorState error={summary.error} onRetry={summary.reload} /> : (
        <>
          {tab === 'manual' && <ManualInput features={summary.data!.features} />}
          {tab === 'samples' && <SampleExplorer features={summary.data!.features} />}
          {tab === 'csv' && <BatchUpload />}
        </>
      )}
    </div>
  )
}

function ManualInput({ features }: { features: FeatureInfo[] }) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [res, setRes] = useState<Prediction | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)

  const fill = (kind: 'benign_example' | 'malignant_example') => {
    setValues(Object.fromEntries(features.map((f) => [f.key, String(f[kind])])))
    setErrors({})
    setRes(null)
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    const found: Record<string, string> = {}
    const numbers: Record<string, number> = {}
    for (const f of features) {
      const raw = (values[f.key] ?? '').trim()
      const n = Number(raw)
      if (raw === '') found[f.key] = 'Required'
      else if (!Number.isFinite(n)) found[f.key] = 'Must be a number'
      else if (n < 0) found[f.key] = 'Cannot be negative'
      else numbers[f.key] = n
    }
    setErrors(found)
    if (Object.keys(found).length) return
    setBusy(true)
    setErr(null)
    try {
      setRes(await api.predict(numbers))
    } catch (e) {
      setErr(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  const errorCount = Object.keys(errors).length
  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <Card
        className="xl:col-span-3"
        title="Cell measurements"
        subtitle="Under each box you can see what the value means and the range found in the dataset. Not sure what to enter? Use an example."
      >
        <form onSubmit={submit} noValidate>
          <div className="mb-4 flex flex-wrap gap-2">
            <button type="button" onClick={() => fill('benign_example')} className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-800 hover:bg-blue-100">Fill typical benign values</button>
            <button type="button" onClick={() => fill('malignant_example')} className="rounded-md border border-violet-200 bg-violet-50 px-3 py-1.5 text-xs font-medium text-violet-800 hover:bg-violet-100">Fill typical malignant values</button>
            <button type="button" onClick={() => { setValues({}); setErrors({}); setRes(null) }} className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium hover:bg-slate-50">Clear</button>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {features.map((f) => {
              const v = Number(values[f.key])
              const outside = values[f.key] && Number.isFinite(v) && (v < f.min || v > f.max)
              return (
                <div key={f.key}>
                  <label htmlFor={f.key} className="text-sm font-medium text-navy-900">{f.label}</label>
                  <input
                    id={f.key}
                    inputMode="decimal"
                    value={values[f.key] ?? ''}
                    onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
                    placeholder={`e.g. ${sig(f.benign_example)}`}
                    aria-invalid={!!errors[f.key]}
                    className={`mt-1 w-full rounded-md border px-3 py-2 text-sm tabular-nums focus:outline-none focus:ring-2 focus:ring-teal-500 ${errors[f.key] ? 'border-amber-500 bg-amber-50' : 'border-slate-300'}`}
                  />
                  <p className={`mt-1 text-[11px] ${errors[f.key] ? 'text-amber-700' : outside ? 'text-amber-700' : 'text-slate-500'}`}>
                    {errors[f.key] ?? (outside ? `Outside the dataset range ${sig(f.min)} – ${sig(f.max)}` : `${f.help}. Range: ${sig(f.min)} – ${sig(f.max)}`)}
                  </p>
                </div>
              )
            })}
          </div>
          {errorCount > 0 && <p role="alert" className="mt-4 text-sm text-amber-800">{errorCount} field(s) need attention.</p>}
          <button type="submit" disabled={busy} className="mt-5 w-full rounded-lg bg-navy-900 py-2.5 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-60">
            {busy ? 'Predicting…' : 'Predict'}
          </button>
        </form>
      </Card>
      <div className="xl:col-span-2">
        {err && <ErrorState error={err} />}
        {busy && <Loading label="Running the model…" />}
        {!busy && res && <PredictionResult res={res} />}
        {!busy && !res && !err && (
          <Card><Empty>Fill in the 10 measurements and press <b>Predict</b> to see the prediction level here.</Empty></Card>
        )}
      </div>
    </div>
  )
}

function SampleExplorer({ features }: { features: FeatureInfo[] }) {
  const samples = useApi(api.samples)
  const [id, setId] = useState<number | null>(null)
  const [res, setRes] = useState<SamplePrediction | null>(null)
  const [revealed, setRevealed] = useState(false)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)

  async function run(sampleId: number) {
    setBusy(true)
    setErr(null)
    setRevealed(false)
    try {
      setRes(await api.predictSample(sampleId))
    } catch (e) {
      setErr(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  if (samples.loading) return <Loading />
  if (samples.error) return <ErrorState error={samples.error} onRetry={samples.reload} />
  const list = samples.data!
  const current = list.find((s) => s.id === id)

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <div className="space-y-4 xl:col-span-2">
        <Card title="Choose a test sample" subtitle={`${list.length} samples the models never saw during training. The real answer stays hidden until you reveal it.`}>
          <div className="flex gap-2">
            <select aria-label="Sample" value={id ?? ''} onChange={(e) => { setId(Number(e.target.value)); setRes(null) }} className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm">
              <option value="" disabled>Select a sample…</option>
              {list.map((s) => <option key={s.id} value={s.id}>Sample #{s.id}</option>)}
            </select>
            <button onClick={() => { const s = list[Math.floor(Math.random() * list.length)]; setId(s.id); setRes(null) }} className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-3 text-sm hover:bg-slate-50">
              <Shuffle className="h-4 w-4" aria-hidden /> Random
            </button>
          </div>
          <button disabled={id == null || busy} onClick={() => id != null && run(id)} className="mt-3 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-50">
            {busy ? 'Predicting…' : 'Predict'}
          </button>
        </Card>
        {current && (
          <Card title={`Measurements · sample #${current.id}`}>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
              {features.map((f) => (
                <div key={f.key} className="flex justify-between border-b border-slate-100 py-1">
                  <dt className="text-slate-600">{f.label}</dt>
                  <dd className="tabular-nums">{sig(current.values[f.key])}</dd>
                </div>
              ))}
            </dl>
          </Card>
        )}
      </div>
      <div className="xl:col-span-3">
        {err && <ErrorState error={err} />}
        {busy && <Loading label="Running the model…" />}
        {!busy && res && (
          <PredictionResult
            res={res}
            extra={revealed ? (
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span>Real diagnosis:</span>
                <Badge tone={res.actual === 'Malignant' ? 'purple' : 'blue'}>{res.actual}</Badge>
                <Badge tone={res.correct ? 'teal' : 'amber'}>{res.correct ? 'The model was correct' : 'The model was wrong'}</Badge>
              </div>
            ) : (
              <button onClick={() => setRevealed(true)} className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">
                <Eye className="h-4 w-4" aria-hidden /> Reveal real diagnosis
              </button>
            )}
          />
        )}
        {!busy && !res && !err && <Card><Empty>Choose a sample and press <b>Predict</b>.</Empty></Card>}
      </div>
    </div>
  )
}

function BatchUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [res, setRes] = useState<BatchResult | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)

  async function upload() {
    if (!file) return
    setErr(null)
    setRes(null)
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErr(new ApiError(415, 'Please choose a .csv file.'))
      return
    }
    setBusy(true)
    try {
      setRes(await api.predictBatch(file))
    } catch (e) {
      setErr(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  function download() {
    if (!res) return
    const lines = [
      '# Educational project only. Not a medical diagnosis.',
      'row,id,status,prediction,malignant_probability,level,errors',
      ...res.rows.map((r) => [r.row, r.id, r.status, r.prediction ?? '', r.malignant_probability ?? '', r.level ?? '', `"${r.errors.join('; ')}"`].join(',')),
    ]
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }))
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <Card title="Predict many samples from a CSV file">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>The first line must contain the 10 column names: <code className="text-xs">radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension</code>, plus an optional <code className="text-xs">id</code>.</li>
          <li>Download the template to see the format. Rows with missing or invalid values are reported, not guessed.</li>
          <li>Maximum 1 MB and 1,000 rows. The file is not stored.</li>
        </ul>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <a href={api.templateUrl} className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-2 text-sm hover:bg-slate-50">
            <Download className="h-4 w-4" aria-hidden /> Template CSV
          </a>
          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-dashed border-teal-400 bg-teal-50 px-3 py-2 text-sm text-teal-800">
            <FileUp className="h-4 w-4" aria-hidden /> {file ? file.name : 'Choose CSV file'}
            <input data-testid="csv-input" type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setRes(null); setErr(null) }} />
          </label>
          <button disabled={!file || busy} onClick={upload} className="inline-flex items-center gap-1.5 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-50">
            <Upload className="h-4 w-4" aria-hidden /> {busy ? 'Predicting…' : 'Upload & predict'}
          </button>
        </div>
      </Card>

      {err && <ErrorState error={err} />}
      {res && (
        <>
          <Disclaimer variant="inline" />
          <Card
            title="Batch results"
            subtitle={`Model used: ${res.model}`}
            action={<button onClick={download} className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50"><Download className="h-4 w-4" aria-hidden /> Download CSV</button>}
          >
            <div className="mb-4 flex flex-wrap gap-2 text-xs">
              <Badge>Total {res.summary.total}</Badge>
              <Badge tone="teal">Valid {res.summary.valid}</Badge>
              <Badge tone="amber">Invalid {res.summary.invalid}</Badge>
              <Badge tone="blue">Benign {res.summary.benign}</Badge>
              <Badge tone="purple">Malignant {res.summary.malignant}</Badge>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                  <tr><th className="py-2 pr-4">Row</th><th className="py-2 pr-4">ID</th><th className="py-2 pr-4">Level</th><th className="py-2 pr-4">Prediction</th><th className="py-2 pr-4 text-right">Chance of malignant</th><th className="py-2">Notes</th></tr>
                </thead>
                <tbody>
                  {res.rows.map((r) => (
                    <tr key={r.row} className="border-b border-slate-100 last:border-0">
                      <td className="py-2 pr-4 text-slate-400">{r.row}</td>
                      <td className="py-2 pr-4 font-mono text-xs">{r.id}</td>
                      <td className="py-2 pr-4">{r.level ? <span className={`font-semibold ${LEVEL_STYLES[r.level].text}`}>{r.level}</span> : <Badge tone="amber">invalid</Badge>}</td>
                      <td className="py-2 pr-4">{r.prediction ?? '—'}</td>
                      <td className="py-2 pr-4 text-right tabular-nums">{r.malignant_probability != null ? prob(r.malignant_probability) : '—'}</td>
                      <td className="py-2 text-xs text-slate-600">
                        {r.errors.length ? r.errors.join('; ') : r.unusual_values?.length ? `Outside dataset range: ${r.unusual_values.join(', ')}` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
