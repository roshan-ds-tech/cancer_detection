import { useState } from 'react'
import { Badge, Card, Empty, ErrorState, Loading, PageHeader } from '../components/ui'
import { api } from '../lib/api'
import { num } from '../lib/format'
import { useApi } from '../lib/hooks'

const METRICS = ['roc_auc', 'recall', 'specificity', 'f1', 'brier'] as const

export default function Experiments() {
  const { data, error, loading, reload } = useApi(api.experiments)
  const [a, setA] = useState<string | null>(null)
  const [b, setB] = useState<string | null>(null)

  if (loading) return <><PageHeader title="Experiment Tracker" /><Loading /></>
  if (error) return <><PageHeader title="Experiment Tracker" /><ErrorState error={error} onRetry={reload} /></>
  const runs = data!.runs
  if (!runs.length) return <><PageHeader title="Experiment Tracker" /><Empty>No runs logged yet. Run <code>python scripts/train_models.py</code>.</Empty></>
  const runA = a ?? runs[0].run_id
  const runB = b ?? runs[Math.min(1, runs.length - 1)].run_id
  const exps = (rid: string) => data!.experiments.filter((e) => e.run_id === rid)
  const algos = Array.from(new Set(data!.experiments.map((e) => e.algorithm)))

  return (
    <div className="space-y-6">
      <PageHeader title="Experiment Tracker" subtitle="Every training run is logged to a local SQLite database (artifacts/experiments.sqlite) with seed, dataset fingerprint, hyperparameters, CV and test metrics. No external MLOps service." />

      <Card title={`Training runs (${runs.length})`}>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b text-slate-500">
              <tr><th className="py-2 pr-3">Run ID</th><th className="pr-3">Created (UTC)</th><th className="pr-3">Selected model</th><th className="pr-3">Seed</th><th className="pr-3">Dataset</th><th className="pr-3">Duration</th><th className="pr-3">scikit-learn</th></tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={r.run_id} className="border-b border-slate-100">
                  <td className="py-1.5 pr-3 font-mono">{r.run_id} {i === 0 && <Badge tone="teal">latest</Badge>}</td>
                  <td className="pr-3">{r.created_at.replace('T', ' ').replace('+00:00', '')}</td>
                  <td className="pr-3">{r.selected_model}</td>
                  <td className="pr-3 tabular-nums">{r.seed}</td>
                  <td className="pr-3 font-mono">{r.dataset_version}</td>
                  <td className="pr-3 tabular-nums">{num(r.duration_seconds, 1)} s</td>
                  <td className="pr-3">{r.library_versions['scikit-learn']}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Compare two runs" subtitle="Mean CV metrics (training split). Δ = run B − run A.">
        <div className="mb-4 flex flex-wrap gap-4 text-sm">
          {[['Run A', runA, setA], ['Run B', runB, setB]].map(([label, val, set]) => (
            <label key={label as string} className="flex items-center gap-2">{label as string}
              <select value={val as string} onChange={(e) => (set as (v: string) => void)(e.target.value)} className="rounded-md border border-slate-300 px-2 py-1 font-mono text-xs">
                {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
              </select>
            </label>
          ))}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b text-slate-500">
              <tr>
                <th className="py-2 pr-3">Algorithm</th>
                {METRICS.map((m) => <th key={m} className="pr-3">{m} A / B (Δ)</th>)}
                <th className="pr-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {algos.map((alg) => {
                const ea = exps(runA).find((e) => e.algorithm === alg)
                const eb = exps(runB).find((e) => e.algorithm === alg)
                return (
                  <tr key={alg} className="border-b border-slate-100">
                    <td className="py-1.5 pr-3 font-medium">{alg}</td>
                    {METRICS.map((m) => {
                      const va = ea?.cv_metrics?.[m]
                      const vb = eb?.cv_metrics?.[m]
                      const d = va != null && vb != null ? vb - va : null
                      return (
                        <td key={m} className="whitespace-nowrap pr-3 tabular-nums">
                          {num(va, 4)} / {num(vb, 4)} {d != null && Math.abs(d) > 1e-9 && <span className="text-slate-500">({d > 0 ? '+' : ''}{d.toFixed(4)})</span>}
                        </td>
                      )
                    })}
                    <td className="pr-3">
                      {[ea, eb].map((e, i) => e ? <Badge key={i} tone={e.status === 'completed' ? 'teal' : 'amber'}>{e.status}</Badge> : <Badge key={i}>n/a</Badge>)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="All logged experiments">
        <div className="max-h-[480px] overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 border-b bg-white text-slate-500">
              <tr><th className="py-2 pr-3">Experiment</th><th className="pr-3">Status</th><th className="pr-3">Hyperparameters</th><th className="pr-3">CV AUC</th><th className="pr-3">Test AUC</th><th className="pr-3">Test recall</th><th className="pr-3">Fit (s)</th><th className="pr-3">Artifact</th></tr>
            </thead>
            <tbody>
              {data!.experiments.map((e) => (
                <tr key={e.experiment_id} className="border-b border-slate-100 align-top">
                  <td className="py-1.5 pr-3"><div className="font-medium">{e.algorithm}</div><div className="font-mono text-[10px] text-slate-500">{e.experiment_id}</div></td>
                  <td className="pr-3"><Badge tone={e.status === 'completed' ? 'teal' : 'amber'}>{e.status}</Badge>{e.error && <div className="mt-1 text-[10px] text-amber-800">{e.error}</div>}</td>
                  <td className="pr-3 font-mono text-[10px]">{e.hyperparameters ? Object.entries(e.hyperparameters).map(([k, v]) => `${k.replace(/^clf__(estimator__)?/, '')}=${String(v)}`).join(', ') : '—'}</td>
                  <td className="pr-3 tabular-nums">{num(e.cv_metrics?.roc_auc, 4)}</td>
                  <td className="pr-3 tabular-nums">{num(e.test_metrics?.roc_auc, 4)}</td>
                  <td className="pr-3 tabular-nums">{num(e.test_metrics?.recall, 3)}</td>
                  <td className="pr-3 tabular-nums">{num(e.training_seconds, 3)}</td>
                  <td className="pr-3 font-mono text-[10px]">{e.artifact_path ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
