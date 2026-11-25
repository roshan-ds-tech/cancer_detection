import { useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Heatmap } from '../components/Heatmap'
import { Card, ErrorState, Loading, PageHeader, SourceTag, Stat } from '../components/ui'
import { api } from '../lib/api'
import { COLORS, sig } from '../lib/format'
import { useApi } from '../lib/hooks'

const PAGE_SIZE = 15

export default function Dataset() {
  const summary = useApi(api.summary)
  const eda = useApi(api.eda)
  const [featureKey, setFeatureKey] = useState('radius')

  if (summary.loading || eda.loading) return <Loading />
  const err = summary.error ?? eda.error
  if (err) return <ErrorState error={err} onRetry={() => { summary.reload(); eda.reload() }} />
  const s = summary.data!
  const e = eda.data!
  const feature = s.features.find((f) => f.key === featureKey)!

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset Explorer"
        subtitle={<>The <b>{s.name}</b> dataset ({s.source}). Each row describes the cell nuclei in a microscope image of a breast tissue sample. This project uses its 10 “mean” measurements.</>}
      />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Total samples" value={s.total} hint={`${s.train_rows} training · ${s.test_rows} test`} />
        <Stat label="Benign" value={<span className="text-blue-700">{s.benign}</span>} hint={`${((s.benign / s.total) * 100).toFixed(1)}% of samples`} />
        <Stat label="Malignant" value={<span className="text-violet-700">{s.malignant}</span>} hint={`${((s.malignant / s.total) * 100).toFixed(1)}% of samples`} />
        <Stat label="Data quality" value={<span className="text-lg">{s.missing_values} missing</span>} hint={`${s.duplicate_rows} duplicate rows`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Class distribution" subtitle="How many samples are in each class" action={<SourceTag kind="all" />}>
          <div className="h-64">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={[{ name: 'Benign', value: s.benign }, { name: 'Malignant', value: s.malignant }]} dataKey="value" nameKey="name" innerRadius={50} outerRadius={90} label isAnimationActive={false}>
                  <Cell fill={COLORS.benign} />
                  <Cell fill={COLORS.malignant} />
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card
          className="lg:col-span-2"
          title="Distribution by class"
          subtitle={feature.help}
          action={
            <select aria-label="Measurement" value={featureKey} onChange={(ev) => setFeatureKey(ev.target.value)} className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm">
              {s.features.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
            </select>
          }
        >
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={e.histograms[featureKey]} barCategoryGap={1}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="bin" tickFormatter={(v) => sig(v, 3)} tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} label={{ value: 'Samples', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip labelFormatter={(v) => `${feature.label} ≈ ${sig(Number(v), 4)}`} />
                <Legend />
                <Bar dataKey="benign" name="Benign" fill={COLORS.benign} isAnimationActive={false} />
                <Bar dataKey="malignant" name="Malignant" fill={COLORS.malignant} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Average {feature.label.toLowerCase()}: benign {sig(feature.benign_mean)}, malignant {sig(feature.malignant_mean)}.
          </p>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Feature correlations" subtitle="Values near 1 mean two measurements rise together (e.g. radius, perimeter and area).">
          <Heatmap features={e.correlation.features} matrix={e.correlation.matrix} />
        </Card>

        <Card title="Cell radius vs texture" subtitle="Each dot is one sample. Malignant samples tend to have larger cells.">
          <div className="h-80">
            <ResponsiveContainer>
              <ScatterChart margin={{ bottom: 16, left: 0, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="radius" name="Cell radius" domain={['auto', 'auto']} tick={{ fontSize: 11 }} label={{ value: 'Cell radius', position: 'bottom', fontSize: 11 }} />
                <YAxis type="number" dataKey="texture" name="Texture" domain={['auto', 'auto']} tick={{ fontSize: 11 }} label={{ value: 'Texture', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend verticalAlign="top" />
                <Scatter name="Benign" data={e.scatter.filter((p) => p.diagnosis === 'Benign')} fill={COLORS.benign} fillOpacity={0.55} isAnimationActive={false} />
                <Scatter name="Malignant" data={e.scatter.filter((p) => p.diagnosis === 'Malignant')} fill={COLORS.malignant} fillOpacity={0.55} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Feature statistics" subtitle="Summary of each measurement over all 569 samples" action={<SourceTag kind="all" />}>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="py-2 pr-4">Measurement</th>
                <th className="py-2 pr-4">Meaning</th>
                <th className="py-2 pr-4 text-right">Min</th>
                <th className="py-2 pr-4 text-right">Max</th>
                <th className="py-2 pr-4 text-right">Mean</th>
                <th className="py-2 pr-4 text-right">Benign avg</th>
                <th className="py-2 text-right">Malignant avg</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {s.features.map((f) => (
                <tr key={f.key} className="border-b border-slate-100 last:border-0">
                  <td className="py-2 pr-4 font-medium text-navy-900">{f.label}</td>
                  <td className="py-2 pr-4 text-xs text-slate-500">{f.help}</td>
                  <td className="py-2 pr-4 text-right">{sig(f.min)}</td>
                  <td className="py-2 pr-4 text-right">{sig(f.max)}</td>
                  <td className="py-2 pr-4 text-right">{sig(f.mean)}</td>
                  <td className="py-2 pr-4 text-right text-blue-700">{sig(f.benign_mean)}</td>
                  <td className="py-2 text-right text-violet-700">{sig(f.malignant_mean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Records labels={s.features.map((f) => ({ key: f.key, label: f.label }))} />
    </div>
  )
}

function Records({ labels }: { labels: { key: string; label: string }[] }) {
  const [offset, setOffset] = useState(0)
  const page = useApi(() => api.records(PAGE_SIZE, offset), [offset])
  return (
    <Card title="Sample records" subtitle="Browse the raw data">
      {page.error ? (
        <ErrorState error={page.error} onRetry={page.reload} />
      ) : !page.data ? (
        <Loading />
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2 pr-3">#</th>
                  {labels.map((l) => <th key={l.key} className="whitespace-nowrap py-2 pr-3 text-right">{l.label}</th>)}
                  <th className="py-2">Diagnosis</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {page.data.rows.map((r) => (
                  <tr key={r.id} className="border-b border-slate-100 last:border-0">
                    <td className="py-1.5 pr-3 text-slate-400">{r.id}</td>
                    {labels.map((l) => <td key={l.key} className="py-1.5 pr-3 text-right">{sig(r[l.key])}</td>)}
                    <td className={`py-1.5 font-medium ${r.diagnosis === 'Malignant' ? 'text-violet-700' : 'text-blue-700'}`}>{r.diagnosis}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex items-center justify-between text-sm text-slate-600">
            <span>Rows {offset + 1}–{Math.min(offset + PAGE_SIZE, page.data.total)} of {page.data.total}</span>
            <div className="flex gap-2">
              <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} className="rounded-md border border-slate-300 px-3 py-1 hover:bg-slate-50 disabled:opacity-40">Previous</button>
              <button disabled={offset + PAGE_SIZE >= page.data.total} onClick={() => setOffset(offset + PAGE_SIZE)} className="rounded-md border border-slate-300 px-3 py-1 hover:bg-slate-50 disabled:opacity-40">Next</button>
            </div>
          </div>
        </>
      )}
    </Card>
  )
}
