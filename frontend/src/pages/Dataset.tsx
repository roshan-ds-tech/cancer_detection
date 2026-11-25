import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Heatmap } from '../components/Heatmap'
import { Badge, Card, Empty, ErrorState, Loading, PageHeader, SourceTag, Stat } from '../components/ui'
import { api } from '../lib/api'
import { COLORS, num, pct, prettyFeature, sig } from '../lib/format'
import { type AsyncState, useApi } from '../lib/hooks'
import type { BoxStat, EdaResponse, FeatureStat, RecordsResponse } from '../lib/types'

function BoxRow({ label, b, lo, hi, color }: { label: string; b: BoxStat; lo: number; hi: number; color: string }) {
  const x = (v: number) => `${((v - lo) / (hi - lo || 1)) * 100}%`
  return (
    <div className="flex items-center gap-3">
      <div className="w-20 text-xs capitalize text-slate-600">{label}</div>
      <div className="relative h-6 flex-1" role="img" aria-label={`${label}: median ${sig(b.median)}, IQR ${sig(b.q1)} to ${sig(b.q3)}`}>
        <div className="absolute top-1/2 h-px bg-slate-400" style={{ left: x(b.whisker_low), width: `calc(${x(b.whisker_high)} - ${x(b.whisker_low)})` }} />
        <div className="absolute top-1 h-4 rounded-sm opacity-80" style={{ left: x(b.q1), width: `calc(${x(b.q3)} - ${x(b.q1)})`, background: color }} />
        <div className="absolute top-0.5 h-5 w-0.5 bg-navy-900" style={{ left: x(b.median) }} />
      </div>
    </div>
  )
}

function FeatureDistribution({ eda, stats }: { eda: EdaResponse; stats: FeatureStat[] }) {
  const [feature, setFeature] = useState('mean_concave_points')
  const hist = eda.histograms[feature].map((b) => ({ ...b, bin: sig(b.bin_start, 3) }))
  const box = eda.boxplots[feature]
  const st = stats.find((s) => s.feature === feature)!
  const lo = Math.min(box.benign.whisker_low, box.malignant.whisker_low)
  const hi = Math.max(box.benign.whisker_high, box.malignant.whisker_high)
  return (
    <Card
      title="Feature distributions by class"
      subtitle="Histograms (20 bins) and box plots, training split only"
      action={
        <label className="text-xs text-slate-600">
          <span className="sr-only">Feature</span>
          <select value={feature} onChange={(e) => setFeature(e.target.value)} className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm">
            {stats.map((s) => <option key={s.feature} value={s.feature}>{prettyFeature(s.feature)}</option>)}
          </select>
        </label>
      }
    >
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="h-64 lg:col-span-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={hist} barCategoryGap={1}>
              <CartesianGrid vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={3} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip labelFormatter={(l) => `bin starting ${l}`} />
              <Legend />
              <Bar dataKey="benign" fill={COLORS.benign} isAnimationActive={false} />
              <Bar dataKey="malignant" fill={COLORS.malignant} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-3">
          <BoxRow label="benign" b={box.benign} lo={lo} hi={hi} color={COLORS.benign} />
          <BoxRow label="malignant" b={box.malignant} lo={lo} hi={hi} color={COLORS.malignant} />
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 pt-2 text-xs">
            <dt className="text-slate-500">Benign median</dt><dd className="tabular-nums">{sig(st.benign_median)}</dd>
            <dt className="text-slate-500">Malignant median</dt><dd className="tabular-nums">{sig(st.malignant_median)}</dd>
            <dt className="text-slate-500">Cohen's d</dt><dd className="tabular-nums">{num(st.cohens_d, 2)}</dd>
            <dt className="text-slate-500">Univariate AUC</dt><dd className="tabular-nums">{num(st.univariate_auc, 3)}</dd>
            <dt className="text-slate-500">Skewness</dt><dd className="tabular-nums">{num(st.skew, 2)}</dd>
            <dt className="text-slate-500">IQR outliers</dt><dd className="tabular-nums">{st.iqr_outliers}</dd>
          </dl>
        </div>
      </div>
    </Card>
  )
}

function RecordsTable() {
  const [offset, setOffset] = useState(0)
  const limit = 10
  const state = useApi(() => api.records(limit, offset), [offset])
  const { data } = state
  return (
    <Card title="Sample records" subtitle="Training split, with known labels" action={<SourceTag kind="train" />}>
      <RecordsPage {...state} />
      <div className="mt-3 flex items-center justify-between text-xs text-slate-600">
        <span>{data ? `Rows ${offset + 1}–${Math.min(offset + limit, data.total)} of ${data.total}` : ''}</span>
        <div className="flex gap-2">
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))} className="rounded-md border px-2 py-1 disabled:opacity-40">Previous</button>
          <button disabled={!data || offset + limit >= data.total} onClick={() => setOffset(offset + limit)} className="rounded-md border px-2 py-1 disabled:opacity-40">Next</button>
        </div>
      </div>
    </Card>
  )
}

function RecordsPage({ data, error, loading, reload }: AsyncState<RecordsResponse>) {
  if (loading) return <Loading />
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!data?.records.length) return <Empty>No records.</Empty>
  const cols = ['mean_radius', 'mean_texture', 'mean_area', 'mean_concavity', 'mean_concave_points', 'worst_radius', 'worst_concave_points']
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="border-b text-slate-500">
          <tr>
            <th className="py-2 pr-3 font-medium">ID</th>
            <th className="py-2 pr-3 font-medium">Label</th>
            {cols.map((c) => <th key={c} className="py-2 pr-3 font-medium">{prettyFeature(c)}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.records.map((r) => (
            <tr key={r.sample_id} className="border-b border-slate-100">
              <td className="py-1.5 pr-3 font-mono">{r.sample_id}</td>
              <td className="py-1.5 pr-3"><Badge tone={r.label === 'malignant' ? 'purple' : 'blue'}>{r.label}</Badge></td>
              {cols.map((c) => <td key={c} className="py-1.5 pr-3 tabular-nums">{sig(r.features[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-slate-500">7 of 30 features shown. Sample ID = row index in the scikit-learn copy (original UCI IDs are not used).</p>
    </div>
  )
}

export default function Dataset() {
  const summary = useApi(api.datasetSummary)
  const eda = useApi(api.eda)
  const [sortKey, setSortKey] = useState<'univariate_auc' | 'cohens_d' | 'feature'>('univariate_auc')

  if (summary.loading || eda.loading) return <><PageHeader title="Dataset Explorer" /><Loading /></>
  const err = summary.error ?? eda.error
  if (err) return <><PageHeader title="Dataset Explorer" /><ErrorState error={err} onRetry={() => { summary.reload(); eda.reload() }} /></>
  const s = summary.data!
  const e = eda.data!
  const integ = s.integrity!
  const stats = [...(s.feature_stats ?? [])].sort((a, b) =>
    sortKey === 'feature' ? a.feature.localeCompare(b.feature) : (b[sortKey] as number) - (a[sortKey] as number),
  )
  const classData = [
    { name: 'benign', count: integ.class_counts.benign },
    { name: 'malignant', count: integ.class_counts.malignant },
  ]
  const evr = e.pca.explained_variance_ratio

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset Explorer"
        subtitle={<>{s.dataset.name} — {s.dataset.license}, doi:{s.dataset.doi}. {s.scope_note}</>}
      />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Rows × features" value={`${integ.n_rows} × ${integ.n_features}`} />
        <Stat label="Missing values" value={integ.missing_values_total} hint={`Non-finite: ${integ.non_finite_values}`} />
        <Stat label="Duplicate rows" value={integ.duplicate_rows} hint={`Negative values: ${integ.negative_values}`} />
        <Stat label="Malignant share" value={pct(integ.class_proportion_malignant)} hint={`Train ${s.dataset.n_train} / test ${s.dataset.n_test} (stratified)`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Class distribution" subtitle="All 569 samples">
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData}>
                <CartesianGrid vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" isAnimationActive={false} fill={COLORS.teal} label={{ position: 'top', fontSize: 12 }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-500">Moderate imbalance (≈37% malignant): accuracy alone is misleading, so recall, specificity and AUC are reported.</p>
        </Card>
        <Card title="PCA projection" subtitle={`Exploratory only · PC1 ${pct(evr[0])}, PC2 ${pct(evr[1])} of variance`} className="lg:col-span-2">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="#e2e8f0" />
                <XAxis type="number" dataKey="pc1" name="PC1" tick={{ fontSize: 11 }} />
                <YAxis type="number" dataKey="pc2" name="PC2" tick={{ fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v) => Number(v).toFixed(2)} />
                <Legend />
                <Scatter name="benign" data={e.pca.points.filter((p) => p.label === 'benign')} fill={COLORS.benign} fillOpacity={0.55} isAnimationActive={false} />
                <Scatter name="malignant" data={e.pca.points.filter((p) => p.label === 'malignant')} fill={COLORS.malignant} fillOpacity={0.55} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-500">{e.pca.note}</p>
        </Card>
      </div>

      <FeatureDistribution eda={e} stats={s.feature_stats ?? []} />

      <div className="grid gap-6 xl:grid-cols-5">
        <Card title="Feature correlations" subtitle={e.scope_note} className="xl:col-span-3">
          <Heatmap features={e.correlation.features} matrix={e.correlation.matrix} />
        </Card>
        <Card title="Multicollinearity" subtitle={`${e.high_correlation_pairs.length} feature pairs with |r| ≥ 0.9`} className="xl:col-span-2">
          <ul className="max-h-56 space-y-1 overflow-y-auto text-xs">
            {e.high_correlation_pairs.map((p) => (
              <li key={p.a + p.b} className="flex justify-between gap-2 border-b border-slate-100 py-1">
                <span>{prettyFeature(p.a)} ↔ {prettyFeature(p.b)}</span><span className="tabular-nums font-medium">{p.r.toFixed(3)}</span>
              </li>
            ))}
          </ul>
          <h3 className="mt-4 text-xs font-semibold text-slate-700">Highest variance inflation factors</h3>
          <ul className="mt-1 space-y-1 text-xs">
            {e.vif.slice(0, 6).map((v) => (
              <li key={v.feature} className="flex justify-between"><span>{prettyFeature(v.feature)}</span><span className="tabular-nums">{v.vif.toFixed(0)}</span></li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            Radius, perimeter and area are near-deterministic functions of each other. This motivates regularised models and cautions against reading individual coefficients or attributions as independent effects.
          </p>
          {s.scaling && <p className="mt-2 text-xs text-slate-500">{s.scaling.note}</p>}
        </Card>
      </div>

      <Card
        title="Feature statistics"
        subtitle="Training split. Univariate AUC = how well one feature alone ranks malignant above benign."
        action={
          <label className="text-xs text-slate-600">Sort by{' '}
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value as typeof sortKey)} className="rounded-md border border-slate-300 bg-white px-2 py-1">
              <option value="univariate_auc">Univariate AUC</option>
              <option value="cohens_d">Cohen's d</option>
              <option value="feature">Name</option>
            </select>
          </label>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b text-slate-500">
              <tr>{['Feature', 'Mean', 'SD', 'Min', 'Median', 'Max', 'Benign mean', 'Malignant mean', "Cohen's d", 'Univ. AUC', 'Outliers'].map((h) => <th key={h} className="py-2 pr-3 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody>
              {stats.map((r) => (
                <tr key={r.feature} className="border-b border-slate-100">
                  <td className="py-1.5 pr-3 font-medium text-slate-700">{prettyFeature(r.feature)}</td>
                  <td className="pr-3 tabular-nums">{sig(r.mean)}</td>
                  <td className="pr-3 tabular-nums">{sig(r.std)}</td>
                  <td className="pr-3 tabular-nums">{sig(r.min)}</td>
                  <td className="pr-3 tabular-nums">{sig(r['50%'])}</td>
                  <td className="pr-3 tabular-nums">{sig(r.max)}</td>
                  <td className="pr-3 tabular-nums">{sig(r.benign_mean)}</td>
                  <td className="pr-3 tabular-nums">{sig(r.malignant_mean)}</td>
                  <td className="pr-3 tabular-nums">{num(r.cohens_d, 2)}</td>
                  <td className="pr-3 tabular-nums">{num(r.univariate_auc, 3)}</td>
                  <td className="pr-3 tabular-nums">{r.iqr_outliers}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <RecordsTable />

      <Card title="Static EDA figures" subtitle="Generated by scripts/run_eda.py from the same data">
        <div className="grid gap-4 md:grid-cols-2">
          {['eda_histograms_mean_features.png', 'eda_boxplots_mean_features.png'].map((f) => (
            <img key={f} src={api.figureUrl(f)} alt={f.replace(/_/g, ' ').replace('.png', '')} loading="lazy" className="w-full rounded border border-slate-200" />
          ))}
        </div>
      </Card>
    </div>
  )
}
