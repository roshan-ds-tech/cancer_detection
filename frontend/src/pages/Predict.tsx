import { Download, Eye, FileUp, FlaskConical, Keyboard, ListChecks, Loader2, Shuffle } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { Disclaimer } from '../components/Disclaimer'
import { PredictionResult } from '../components/PredictionResult'
import { Badge, Card, Empty, ErrorState, Loading, PageHeader, Tabs } from '../components/ui'
import { api, ApiError } from '../lib/api'
import { pct, prettyFeature, sig } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { BatchResponse, FeatureSpec, PredictResponse, SamplePredictResponse } from '../lib/types'
import { validateFeatureInputs } from '../lib/validation'

const MAX_UPLOAD_BYTES = 1_000_000

function SubmitButton({ busy, children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { busy: boolean }) {
  return (
    <button
      {...rest}
      disabled={busy || rest.disabled}
      className="inline-flex items-center gap-2 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-50"
    >
      {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  )
}

/* ---------------- A. held-out sample explorer ---------------- */
function SampleExplorer() {
  const samples = useApi(api.demoSamples)
  const [id, setId] = useState<number | null>(null)
  const [res, setRes] = useState<SamplePredictResponse | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const [revealed, setRevealed] = useState(false)

  if (samples.loading) return <Loading />
  if (samples.error) return <ErrorState error={samples.error} onRetry={samples.reload} />
  const list = samples.data!.samples
  const current = list.find((s) => s.sample_id === id) ?? null

  const choose = (sid: number) => {
    setId(sid)
    setRes(null)
    setErr(null)
    setRevealed(false)
  }
  const run = async () => {
    if (id == null) return
    setBusy(true)
    setErr(null)
    try {
      setRes(await api.predictSample(id))
    } catch (e) {
      setErr(e as ApiError)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <div className="space-y-4 xl:col-span-2">
        <Card title="Choose a held-out sample" subtitle={`${list.length} records from the untouched test split. ${samples.data!.note}`}>
          <div className="flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="sample-select">Sample ID</label>
            <select id="sample-select" value={id ?? ''} onChange={(e) => choose(Number(e.target.value))} className="flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-sm">
              <option value="" disabled>Select sample ID…</option>
              {list.map((s) => <option key={s.sample_id} value={s.sample_id}>Sample #{s.sample_id}</option>)}
            </select>
            <button onClick={() => choose(list[Math.floor(Math.random() * list.length)].sample_id)} className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1.5 text-sm hover:bg-slate-50">
              <Shuffle className="h-4 w-4" aria-hidden /> Random
            </button>
          </div>
          <div className="mt-4">
            <SubmitButton busy={busy} disabled={id == null} onClick={run}>Run model</SubmitButton>
          </div>
        </Card>
        {current ? (
          <Card title={`Feature values · sample #${current.sample_id}`}>
            <div className="grid max-h-[420px] grid-cols-1 gap-x-4 overflow-y-auto text-xs sm:grid-cols-2">
              {Object.entries(current.features).map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-slate-100 py-1">
                  <span className="text-slate-600">{prettyFeature(k)}</span>
                  <span className="tabular-nums">{sig(v)}</span>
                </div>
              ))}
            </div>
          </Card>
        ) : (
          <Card><Empty>Select a sample to see its 30 measurements.</Empty></Card>
        )}
      </div>
      <div className="xl:col-span-3">
        {err && <ErrorState error={err} onRetry={run} />}
        {busy && <Loading label="Running model and computing explanation…" />}
        {res && !busy && (
          <PredictionResult
            res={res}
            groundTruth={
              revealed ? (
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  Known dataset label: <Badge tone={res.ground_truth === 'malignant' ? 'purple' : 'blue'}>{res.ground_truth}</Badge>
                  {res.prediction && (
                    res.prediction.predicted_class === res.ground_truth
                      ? <Badge tone="teal">model agreed with label</Badge>
                      : <Badge tone="amber">model disagreed with label ({res.ground_truth === 'malignant' ? 'false negative' : 'false positive'})</Badge>
                  )}
                </div>
              ) : (
                <button onClick={() => setRevealed(true)} className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">
                  <Eye className="h-4 w-4" aria-hidden /> Reveal known dataset label
                </button>
              )
            }
          />
        )}
        {!res && !busy && !err && <Card><Empty>Pick a sample and run the model. The known label stays hidden until you reveal it.</Empty></Card>}
      </div>
    </div>
  )
}

/* ---------------- B. manual feature input ---------------- */
function ManualInput() {
  const feats = useApi(api.features)
  const samples = useApi(api.demoSamples)
  const [values, setValues] = useState<Record<string, string>>({})
  const [touched, setTouched] = useState(false)
  const [res, setRes] = useState<PredictResponse | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const specs = useMemo(() => feats.data?.features ?? [], [feats.data])
  const v = useMemo(() => validateFeatureInputs(values, specs), [values, specs])

  if (feats.loading) return <Loading />
  if (feats.error) return <ErrorState error={feats.error} onRetry={feats.reload} />

  const fill = (src: Record<string, number>) => {
    setValues(Object.fromEntries(Object.entries(src).map(([k, x]) => [k, String(x)])))
    setRes(null)
    setErr(null)
  }
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setTouched(true)
    if (Object.keys(v.errors).length) return
    setBusy(true)
    setErr(null)
    try {
      setRes(await api.predict(v.parsed))
    } catch (ex) {
      setErr(ex as ApiError)
    } finally {
      setBusy(false)
    }
  }
  const groups: { stat: FeatureSpec['statistic']; title: string }[] = [
    { stat: 'mean', title: 'Mean values' },
    { stat: 'se', title: 'Standard errors' },
    { stat: 'worst', title: '“Worst” values (mean of three largest)' },
  ]
  const nErr = Object.keys(v.errors).length

  return (
    <div className="space-y-6">
      <Card title="Manual feature entry (advanced)">
        <div className="mb-4 space-y-2 text-xs text-slate-600">
          <p>{feats.data!.measurement_note} Do not enter real patient data.</p>
          <p><span className="font-medium">Missing-value policy:</span> {feats.data!.missing_value_policy}</p>
        </div>
        <div className="mb-4 flex flex-wrap gap-2">
          <button type="button" onClick={() => fill(Object.fromEntries(specs.map((s) => [s.name, s.train_median])))} className="rounded-md border border-slate-300 px-2.5 py-1 text-xs hover:bg-slate-50">Fill with training medians</button>
          {samples.data && (
            <button type="button" onClick={() => fill(samples.data!.samples[Math.floor(Math.random() * samples.data!.samples.length)].features)} className="rounded-md border border-slate-300 px-2.5 py-1 text-xs hover:bg-slate-50">Fill with a random held-out sample</button>
          )}
          <button type="button" onClick={() => { setValues({}); setTouched(false); setRes(null) }} className="rounded-md border border-slate-300 px-2.5 py-1 text-xs hover:bg-slate-50">Clear</button>
        </div>
        <form onSubmit={submit} noValidate aria-label="Manual feature input">
          {groups.map((g) => (
            <fieldset key={g.stat} className="mb-5">
              <legend className="mb-2 text-sm font-semibold text-navy-900">{g.title}</legend>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                {specs.filter((s) => s.statistic === g.stat).map((s) => {
                  const e = touched ? v.errors[s.name] : undefined
                  const w = v.warnings[s.name]
                  return (
                    <div key={s.name}>
                      <label htmlFor={s.name} className="block text-xs font-medium text-slate-700" title={s.description}>
                        {prettyFeature(s.name)}
                      </label>
                      <input
                        id={s.name}
                        name={s.name}
                        inputMode="decimal"
                        value={values[s.name] ?? ''}
                        onChange={(ev) => setValues({ ...values, [s.name]: ev.target.value })}
                        aria-invalid={!!e}
                        aria-describedby={`${s.name}-hint`}
                        className={`mt-1 w-full rounded-md border px-2 py-1.5 text-sm tabular-nums ${e ? 'border-amber-500 bg-amber-50' : 'border-slate-300'}`}
                      />
                      <p id={`${s.name}-hint`} className={`mt-0.5 text-[10px] ${e ? 'text-amber-800' : w ? 'text-amber-700' : 'text-slate-500'}`}>
                        {e ?? w ?? `${s.unit}; train ${sig(s.train_min, 3)}–${sig(s.train_max, 3)}`}
                      </p>
                    </div>
                  )
                })}
              </div>
            </fieldset>
          ))}
          <div className="flex flex-wrap items-center gap-3">
            <SubmitButton busy={busy} type="submit">Run model</SubmitButton>
            {touched && nErr > 0 && <span role="alert" className="text-sm text-amber-800">{nErr} field(s) need attention.</span>}
          </div>
        </form>
      </Card>
      {err && (
        <ErrorState error={err} />
      )}
      {res && <PredictionResult res={res} />}
    </div>
  )
}

/* ---------------- C. CSV batch inference ---------------- */
function toCsv(r: BatchResponse) {
  const esc = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s)
  const lines = [
    `# ${r.disclaimer}`,
    `# model_version=${r.model_version} operating_threshold=${r.operating_threshold}`,
    'row,sample_id,status,predicted_dataset_class,probability_malignant,errors,warnings',
    ...r.rows.map((x) => [x.row, x.sample_id ?? '', x.status, x.predicted_class ?? '', x.probability_malignant?.toFixed(6) ?? '', x.errors.join('; '), x.warnings.join('; ')].map((c) => esc(String(c))).join(',')),
  ]
  return lines.join('\n')
}

function BatchUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [clientErr, setClientErr] = useState<string | null>(null)
  const [res, setRes] = useState<BatchResponse | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const pick = (f: File | null) => {
    setRes(null)
    setErr(null)
    setClientErr(null)
    if (!f) { setFile(null); return }
    if (!f.name.toLowerCase().endsWith('.csv')) { setClientErr('Only .csv files are accepted.'); setFile(null); return }
    if (f.size > MAX_UPLOAD_BYTES) { setClientErr(`File is ${(f.size / 1e6).toFixed(2)} MB; limit is 1 MB.`); setFile(null); return }
    setFile(f)
  }
  const upload = async () => {
    if (!file) return
    setBusy(true)
    try {
      setRes(await api.predictBatch(file))
    } catch (e) {
      setErr(e as ApiError)
    } finally {
      setBusy(false)
    }
  }
  const download = () => {
    if (!res) return
    const url = URL.createObjectURL(new Blob([toCsv(res)], { type: 'text/csv' }))
    const a = document.createElement('a')
    a.href = url
    a.download = 'oncosense_predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
  }
  const details = err?.details as Record<string, unknown> | undefined

  return (
    <div className="space-y-6">
      <Card title="CSV batch inference">
        <ul className="mb-4 list-disc space-y-1 pl-5 text-xs text-slate-600">
          <li>Header must contain exactly the 30 feature columns (snake_case) plus an optional <code>sample_id</code>. Unknown columns are rejected.</li>
          <li>Missing, non-numeric, negative or non-finite cells mark the row invalid — nothing is imputed.</li>
          <li>Limits: 1 MB, 2,000 rows. Files are processed in memory and not stored.</li>
        </ul>
        <div className="flex flex-wrap items-center gap-3">
          <a href={api.templateUrl} className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50" download>
            <Download className="h-4 w-4" aria-hidden /> Template CSV
          </a>
          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-dashed border-teal-400 bg-teal-50 px-3 py-1.5 text-sm text-teal-900 hover:bg-teal-100">
            <FileUp className="h-4 w-4" aria-hidden /> {file ? file.name : 'Choose CSV file'}
            <input ref={input} data-testid="csv-input" type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          </label>
          <SubmitButton busy={busy} disabled={!file} onClick={upload}>Upload &amp; predict</SubmitButton>
        </div>
        {clientErr && <p role="alert" className="mt-3 text-sm text-amber-800">{clientErr}</p>}
      </Card>

      {err && (
        <div role="alert" className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          <div className="font-medium">{err.message}</div>
          {details && typeof details === 'object' && (
            <ul className="mt-2 list-disc pl-5 text-xs">
              {Object.entries(details).filter(([k, val]) => k !== 'message' && Array.isArray(val) ? (val as unknown[]).length > 0 : k === 'hint').map(([k, val]) => (
                <li key={k}><span className="font-medium">{k.replace(/_/g, ' ')}:</span> {Array.isArray(val) ? val.join(', ') : String(val)}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {res && (
        <div className="space-y-4">
          <Disclaimer variant="inline" />
          <Card
            title="Batch results"
            subtitle={`${res.model_version} · threshold ${res.operating_threshold}`}
            action={<button onClick={download} className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50"><Download className="h-4 w-4" aria-hidden /> Download CSV</button>}
          >
            <div className="mb-4 flex flex-wrap gap-2 text-xs">
              <Badge>total {res.summary.total}</Badge>
              <Badge tone="teal">predicted {res.summary.predicted}</Badge>
              <Badge tone="amber">with warning {res.summary.warning}</Badge>
              <Badge>abstained {res.summary.abstained}</Badge>
              <Badge tone="amber">invalid {res.summary.invalid}</Badge>
            </div>
            <div className="max-h-[480px] overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 border-b bg-white text-slate-500">
                  <tr><th className="py-1.5 pr-2">Row</th><th className="pr-2">Sample ID</th><th className="pr-2">Status</th><th className="pr-2">Dataset class</th><th className="pr-2">P(malignant)</th><th>Messages</th></tr>
                </thead>
                <tbody>
                  {res.rows.map((r) => (
                    <tr key={r.row} className="border-b border-slate-100 align-top">
                      <td className="py-1.5 pr-2 tabular-nums">{r.row}</td>
                      <td className="pr-2 font-mono">{r.sample_id ?? '—'}</td>
                      <td className="pr-2"><Badge tone={r.status === 'predicted' ? 'teal' : r.status === 'warning' || r.status === 'invalid' ? 'amber' : 'slate'}>{r.status}</Badge></td>
                      <td className="pr-2 capitalize">{r.predicted_class ?? '—'}</td>
                      <td className="pr-2 tabular-nums">{r.probability_malignant == null ? '—' : pct(r.probability_malignant, 2)}</td>
                      <td className="text-[11px] text-slate-600">{[...r.errors, ...r.warnings].join(' · ') || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

export default function Predict() {
  const [tab, setTab] = useState<'sample' | 'manual' | 'csv'>('sample')
  return (
    <div>
      <PageHeader
        title="Prediction Laboratory"
        subtitle="Run the selected model on held-out records, hand-entered measurements or a CSV file. Outputs are dataset-class predictions for teaching and research — never a diagnosis."
      >
        <FlaskConical className="hidden h-8 w-8 text-teal-600 md:block" aria-hidden />
      </PageHeader>
      <Tabs
        label="Prediction input mode"
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'sample', label: 'Held-out sample explorer', icon: <ListChecks className="h-4 w-4" aria-hidden /> },
          { id: 'manual', label: 'Manual input', icon: <Keyboard className="h-4 w-4" aria-hidden /> },
          { id: 'csv', label: 'CSV batch', icon: <FileUp className="h-4 w-4" aria-hidden /> },
        ]}
      />
      {tab === 'sample' && <SampleExplorer />}
      {tab === 'manual' && <ManualInput />}
      {tab === 'csv' && <BatchUpload />}
    </div>
  )
}
