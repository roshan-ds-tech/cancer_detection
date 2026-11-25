import { Award, Play, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Badge, Card, ErrorState, Loading, PageHeader, SourceTag } from '../components/ui'
import { api, ApiError } from '../lib/api'
import { COLORS, MODEL_COLORS, num, pct } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { ModelResult, Results } from '../lib/types'

export default function Models() {
  const initial = useApi(api.results)
  const history = useApi(api.history)
  const [fresh, setFresh] = useState<Results | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<ApiError | null>(null)

  async function runModels() {
    setRunning(true)
    setRunError(null)
    try {
      setFresh(await api.train())
      history.reload()
    } catch (e) {
      setRunError(e as ApiError)
    } finally {
      setRunning(false)
    }
  }

  const data = fresh ?? initial.data
  const runButton = (
    <button onClick={runModels} disabled={running} className="inline-flex items-center gap-2 rounded-lg bg-navy-900 px-4 py-2 text-sm font-semibold text-white hover:bg-navy-800 disabled:opacity-60">
      {running ? <RefreshCw className="h-4 w-4 animate-spin" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
      {running ? 'Training models…' : 'Run models'}
    </button>
  )

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Performance"
        subtitle="Six machine learning models are tuned with 5-fold cross-validation on the training data and then tested on the 114 test samples. Press Run models to train them all again."
      >
        {runButton}
      </PageHeader>

      {runError && <ErrorState error={runError} />}
      {initial.loading && !data && <Loading label="Loading results…" />}
      {initial.error && !data && <ErrorState error={initial.error} onRetry={initial.reload} />}

      {data && (
        <>
          <SelectedModel data={data} />
          <ComparisonTable data={data} />
          <div className="grid gap-6 lg:grid-cols-2">
            <AccuracyChart data={data} />
            <RocChart data={data} />
          </div>
          <Card title="Confusion matrices" subtitle="How each model classified the test samples" action={<SourceTag kind="test" />}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.results.map((r) => <ConfusionMatrix key={r.model} result={r} best={r.model === data.best_model} />)}
            </div>
          </Card>
        </>
      )}

      <Card title="Training history" subtitle="Every time the models are trained, the run is saved here">
        {history.error ? <ErrorState error={history.error} onRetry={history.reload} /> : !history.data ? <Loading /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2 pr-4">Run</th>
                  <th className="py-2 pr-4">Date</th>
                  <th className="py-2 pr-4">Best model</th>
                  <th className="py-2 pr-4 text-right">CV accuracy</th>
                  <th className="py-2 pr-4 text-right">Test accuracy</th>
                  <th className="py-2 pr-4 text-right">Test recall</th>
                  <th className="py-2 text-right">Time</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {history.data.map((h) => (
                  <tr key={h.run} className="border-b border-slate-100 last:border-0">
                    <td className="py-2 pr-4">#{h.run}</td>
                    <td className="py-2 pr-4">{h.trained_at}</td>
                    <td className="py-2 pr-4">{h.best_model}</td>
                    <td className="py-2 pr-4 text-right">{pct(h.cv_accuracy)}</td>
                    <td className="py-2 pr-4 text-right">{pct(h.test_accuracy)}</td>
                    <td className="py-2 pr-4 text-right">{pct(h.test_recall)}</td>
                    <td className="py-2 text-right">{h.duration_seconds} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function SelectedModel({ data }: { data: Results }) {
  const best = data.results.find((r) => r.model === data.best_model)!
  return (
    <section className="rounded-xl border border-teal-200 bg-gradient-to-br from-teal-50 to-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-teal-700">
            <Award className="h-4 w-4" aria-hidden /> Best model
          </div>
          <h2 className="mt-1 text-2xl font-semibold text-navy-900">{best.model}</h2>
          <p className="mt-1 max-w-2xl text-sm text-slate-600">
            Chosen because it had the highest cross-validation accuracy on the training data
            ({pct(best.cv_accuracy)} ± {pct(best.cv_std)}). The test data was not used to choose it.
            Trained {data.trained_at} on {data.train_rows} samples in {data.duration_seconds} s.
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {Object.entries(best.best_params).map(([k, v]) => <Badge key={k}>{k} = {String(v)}</Badge>)}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 text-center sm:grid-cols-4">
          {[['Accuracy', best.accuracy], ['Recall', best.recall], ['Precision', best.precision], ['ROC-AUC', best.roc_auc]].map(([l, v]) => (
            <div key={l as string} className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
              <div className="text-lg font-semibold tabular-nums text-navy-900">{l === 'ROC-AUC' ? num(v as number) : pct(v as number)}</div>
              <div className="text-[11px] text-slate-500">{l} (test)</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function ComparisonTable({ data }: { data: Results }) {
  return (
    <Card title="Model comparison" subtitle="CV = 5-fold cross-validation on training data. All other metrics are on the test data." action={<SourceTag kind="test" />}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4 text-right">CV accuracy</th>
              <th className="py-2 pr-4 text-right">Accuracy</th>
              <th className="py-2 pr-4 text-right">Precision</th>
              <th className="py-2 pr-4 text-right">Recall</th>
              <th className="py-2 pr-4 text-right">Specificity</th>
              <th className="py-2 pr-4 text-right">F1</th>
              <th className="py-2 pr-4 text-right">ROC-AUC</th>
              <th className="py-2 pr-4">Best settings</th>
              <th className="py-2 text-right">Train time</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {data.results.map((r) => (
              <tr key={r.model} className={`border-b border-slate-100 last:border-0 ${r.model === data.best_model ? 'bg-teal-50/60 font-medium' : ''}`}>
                <td className="py-2 pr-4">
                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLORS[r.model] }} />
                  {r.model}
                </td>
                <td className="py-2 pr-4 text-right">{pct(r.cv_accuracy)} <span className="text-xs text-slate-400">± {pct(r.cv_std)}</span></td>
                <td className="py-2 pr-4 text-right">{pct(r.accuracy)}</td>
                <td className="py-2 pr-4 text-right">{pct(r.precision)}</td>
                <td className="py-2 pr-4 text-right">{pct(r.recall)}</td>
                <td className="py-2 pr-4 text-right">{pct(r.specificity)}</td>
                <td className="py-2 pr-4 text-right">{pct(r.f1)}</td>
                <td className="py-2 pr-4 text-right">{num(r.roc_auc)}</td>
                <td className="py-2 pr-4 text-xs text-slate-500">
                  {Object.keys(r.best_params).length ? Object.entries(r.best_params).map(([k, v]) => `${k}=${v}`).join(', ') : 'defaults'}
                </td>
                <td className="py-2 text-right text-xs">{r.train_seconds} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function AccuracyChart({ data }: { data: Results }) {
  const rows = data.results.map((r) => ({
    name: r.model.replace('Support Vector Machine', 'SVM').replace('K-Nearest Neighbors', 'KNN').replace('Logistic Regression', 'Log. Regression'),
    'CV accuracy': +(r.cv_accuracy * 100).toFixed(1),
    'Test accuracy': +(r.accuracy * 100).toFixed(1),
  }))
  return (
    <Card title="Accuracy comparison" subtitle="Cross-validation accuracy (training data) vs accuracy on the test data">
      <div className="h-72">
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" tick={{ fontSize: 11 }} />
            <YAxis domain={[85, 100]} allowDataOverflow unit="%" tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend verticalAlign="top" />
            <Bar dataKey="CV accuracy" fill={COLORS.teal} radius={[3, 3, 0, 0]} isAnimationActive={false} />
            <Bar dataKey="Test accuracy" fill={COLORS.navy} radius={[3, 3, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function RocChart({ data }: { data: Results }) {
  const [shown, setShown] = useState<string[]>(data.results.map((r) => r.model))
  const toggle = (m: string) => setShown(shown.includes(m) ? shown.filter((x) => x !== m) : [...shown, m])
  return (
    <Card title="ROC curves" subtitle="Closer to the top-left corner is better. The dashed line is random guessing." action={<SourceTag kind="test" />}>
      <div className="mb-2 flex flex-wrap gap-1.5">
        {data.results.map((r) => (
          <button key={r.model} onClick={() => toggle(r.model)} className={`rounded-full px-2 py-0.5 text-xs ring-1 ring-inset ${shown.includes(r.model) ? 'ring-slate-300' : 'opacity-40 ring-slate-200'}`}>
            <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: MODEL_COLORS[r.model] }} />
            {r.model} ({num(r.roc_auc)})
          </button>
        ))}
      </div>
      <div className="h-64">
        <ResponsiveContainer>
          <LineChart margin={{ bottom: 16, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="fpr" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fontSize: 11 }} label={{ value: 'False positive rate', position: 'bottom', fontSize: 11 }} />
            <YAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fontSize: 11 }} label={{ value: 'True positive rate', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip formatter={(v) => Number(v).toFixed(3)} labelFormatter={(v) => `FPR ${Number(v).toFixed(3)}`} />
            <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#94a3b8" strokeDasharray="4 4" dot={false} isAnimationActive={false} legendType="none" name="Random" />
            {data.results.filter((r) => shown.includes(r.model)).map((r) => (
              <Line key={r.model} data={r.roc_curve} dataKey="tpr" name={r.model} stroke={MODEL_COLORS[r.model]} strokeWidth={r.model === data.best_model ? 2.5 : 1.5} dot={false} type="stepAfter" isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function ConfusionMatrix({ result, best }: { result: ModelResult; best: boolean }) {
  const [[tn, fp], [fn, tp]] = result.confusion_matrix
  const cell = (v: number, label: string, ok: boolean) => (
    <div className={`rounded-md p-2 text-center ${ok ? 'bg-teal-50 text-teal-900' : 'bg-amber-50 text-amber-900'}`}>
      <div className="text-xl font-semibold tabular-nums">{v}</div>
      <div className="text-[10px]">{label}</div>
    </div>
  )
  return (
    <div className={`rounded-lg border p-3 ${best ? 'border-teal-300' : 'border-slate-200'}`}>
      <div className="mb-2 flex items-center justify-between text-sm font-medium text-navy-900">
        {result.model}
        {best && <Badge tone="teal">best</Badge>}
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-1.5 text-[11px] text-slate-500">
        <span />
        <span className="text-center">Pred. benign</span>
        <span className="text-center">Pred. malignant</span>
        <span>Benign</span>
        {cell(tn, 'correct', true)}
        {cell(fp, 'false alarm', false)}
        <span>Malignant</span>
        {cell(fn, 'missed', false)}
        {cell(tp, 'correct', true)}
      </div>
    </div>
  )
}
