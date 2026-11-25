import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ContributionChart } from '../components/ContributionChart'
import { Card, Empty, ErrorState, Loading, PageHeader, SourceTag } from '../components/ui'
import { api, ApiError } from '../lib/api'
import { COLORS, num, pct, prettyFeature } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { SamplePredictResponse } from '../lib/types'

function LocalExplanation() {
  const samples = useApi(api.demoSamples)
  const [res, setRes] = useState<SamplePredictResponse | null>(null)
  const [err, setErr] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const run = async (id: number) => {
    setBusy(true)
    setErr(null)
    try { setRes(await api.predictSample(id)) } catch (e) { setErr(e as ApiError) } finally { setBusy(false) }
  }
  return (
    <Card title="Sample-level explanation" subtitle="SHAP values for one held-out record, computed live by the API" action={<SourceTag kind="test" />}>
      {samples.error && <ErrorState error={samples.error} onRetry={samples.reload} />}
      {samples.data && (
        <label className="mb-4 flex items-center gap-2 text-sm">Sample
          <select defaultValue="" onChange={(e) => run(Number(e.target.value))} className="rounded-md border border-slate-300 px-2 py-1 text-sm">
            <option value="" disabled>Select…</option>
            {samples.data.samples.map((s) => <option key={s.sample_id} value={s.sample_id}>#{s.sample_id}</option>)}
          </select>
        </label>
      )}
      {busy && <Loading label="Computing SHAP values…" />}
      {err && <ErrorState error={err} />}
      {res && !busy && res.explanation && (
        <div>
          <p className="mb-3 text-xs text-slate-600">
            Model output P(malignant) = {pct(res.prediction?.probability_malignant, 2)} = baseline {pct(res.explanation.base_value, 1)} + Σ contributions ({(res.explanation.sum_of_contributions * 100).toFixed(1)} pts). All 30 features shown.
          </p>
          <ContributionChart items={res.explanation.all_contributions} />
        </div>
      )}
      {res && !busy && !res.explanation && <Empty>The model abstained for this sample, so no explanation is produced.</Empty>}
      {!res && !busy && !err && <Empty>Select a sample to see which feature values moved the model output.</Empty>}
    </Card>
  )
}

export default function Explain() {
  const imp = useApi(api.importance)
  const feats = useApi(api.features)
  const models = useApi(api.models)
  const [permKey, setPermKey] = useState<string | null>(null)
  const [nativeKey, setNativeKey] = useState<string>('logistic_regression')

  if (imp.loading || feats.loading || models.loading) return <><PageHeader title="Explainable AI" /><Loading /></>
  const err = imp.error ?? feats.error ?? models.error
  if (err) return <><PageHeader title="Explainable AI" /><ErrorState error={err} onRetry={() => { imp.reload(); feats.reload(); models.reload() }} /></>
  const I = imp.data!
  const key = permKey ?? I.selected_key
  const name = (k: string) => models.data!.models.find((m) => m.key === k)?.name ?? k
  const perm = (I.permutation_importance[key] ?? []).slice(0, 15).map((d) => ({ ...d, label: prettyFeature(d.feature) }))
  const shap = I.shap_global.slice(0, 15).map((d) => ({ ...d, label: prettyFeature(d.feature) }))
  const nativeOptions = Object.entries(I.native_importance).filter(([, v]) => v)
  const native = I.native_importance[nativeKey]
  const nat = (native?.values ?? []).slice(0, 15).map((d) => ({ ...d, label: prettyFeature(d.feature) }))
  const top = I.shap_global.slice(0, 6).map((s) => feats.data!.features.find((f) => f.name === s.feature)!)

  return (
    <div className="space-y-6">
      <PageHeader title="Explainable AI" subtitle={<>{I.caveat} Selected model: <span className="font-medium">{name(I.selected_key)}</span>.</>} />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Global SHAP importance (selected model)" subtitle={I.shap_method} action={<SourceTag kind="test" />}>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shap} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [Number(v).toFixed(4), 'mean |SHAP| (prob.)']} />
                <Bar dataKey="mean_abs_shap" fill={COLORS.purple} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card
          title="Permutation importance"
          subtitle="Mean drop in test ROC-AUC when a feature is shuffled (± SD over repeats)"
          action={
            <select aria-label="Model for permutation importance" value={key} onChange={(e) => setPermKey(e.target.value)} className="rounded-md border border-slate-300 px-2 py-1 text-xs">
              {Object.keys(I.permutation_importance).map((k) => <option key={k} value={k}>{name(k)}</option>)}
            </select>
          }
        >
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={perm} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                <Bar dataKey="importance_mean" fill={COLORS.teal} isAnimationActive={false}>
                  <ErrorBar dataKey="importance_std" direction="x" stroke="#334155" width={3} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-500">With highly correlated features, shuffling one leaves its correlated partners intact, so permutation importance often understates shared information (many values near zero).</p>
        </Card>
      </div>

      <LocalExplanation />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Model-native importances"
          subtitle={native?.description ?? 'Not available for this model'}
          action={
            <select aria-label="Model for native importance" value={nativeKey} onChange={(e) => setNativeKey(e.target.value)} className="rounded-md border border-slate-300 px-2 py-1 text-xs">
              {nativeOptions.map(([k]) => <option key={k} value={k}>{name(k)}</option>)}
            </select>
          }
        >
          {nat.length ? (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={nat} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid horizontal={false} stroke="#e2e8f0" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                  <Bar dataKey="value" isAnimationActive={false}>
                    {nat.map((d) => <Cell key={d.feature} fill={d.value >= 0 ? COLORS.purple : COLORS.blue} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty>No native importance for this model.</Empty>}
          {I.native_importance[I.selected_key] == null && (
            <p className="mt-2 text-xs text-slate-500">The selected model ({name(I.selected_key)}) has no native importance (e.g. RBF-kernel SVM), which is why model-agnostic SHAP and permutation methods are used for it.</p>
          )}
        </Card>
        <Card title="What the most influential features measure" subtitle="Ranked by global mean |SHAP| for the selected model">
          <dl className="space-y-3 text-sm">
            {top.map((f) => (
              <div key={f.name}>
                <dt className="font-medium text-navy-900">{prettyFeature(f.name)}</dt>
                <dd className="text-xs text-slate-600">{f.description} Unit: {f.unit}. Training range {num(f.train_min, 4)}–{num(f.train_max, 4)}.</dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 text-xs text-slate-500">
            Larger, more irregular nuclei (size, concavity, concave points) are associated with the malignant class in this dataset, consistent with the original authors' findings. This is a statistical association learned by the model, not a demonstration of causation.
          </p>
        </Card>
      </div>

      <Card title="Static SHAP figures" subtitle="Beeswarm over test samples and an example waterfall, generated during training">
        <div className="grid gap-4 md:grid-cols-2">
          {['shap_beeswarm.png', 'shap_waterfall_example.png'].map((f) => (
            <img key={f} src={api.figureUrl(f)} alt={f.replace(/_/g, ' ').replace('.png', '')} loading="lazy" className="w-full rounded border border-slate-200" />
          ))}
        </div>
      </Card>
    </div>
  )
}
