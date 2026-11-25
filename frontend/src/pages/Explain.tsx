import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, ErrorState, Loading, PageHeader, SourceTag } from '../components/ui'
import { api } from '../lib/api'
import { COLORS, sig } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { ImportanceRow } from '../lib/types'

function ImportanceChart({ rows, color, signed, unit = '' }: { rows: ImportanceRow[]; color?: string; signed?: boolean; unit?: string }) {
  return (
    <div className="h-80">
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }}>
          <CartesianGrid horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => sig(Number(v), 2) + unit} />
          <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
          {signed && <ReferenceLine x={0} stroke="#64748b" />}
          <Tooltip formatter={(v) => sig(Number(v), 3)} />
          <Bar dataKey="value" isAnimationActive={false} fill={color}>
            {signed && rows.map((r) => <Cell key={r.feature} fill={r.value > 0 ? COLORS.malignant : COLORS.benign} />)}
            {rows[0]?.std !== undefined && <ErrorBar dataKey="std" width={4} stroke="#475569" direction="x" />}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Explain() {
  const results = useApi(api.results)
  const summary = useApi(api.summary)
  if (results.loading || summary.loading) return <Loading />
  const err = results.error ?? summary.error
  if (err) return <ErrorState error={err} onRetry={() => { results.reload(); summary.reload() }} />
  const r = results.data!
  const imp = r.importance

  return (
    <div className="space-y-6">
      <PageHeader
        title="Explainable AI"
        subtitle={<>Which measurements do the models rely on? Three different methods are shown. The Prediction Lab also explains every single prediction. Best model: <b>{r.best_model}</b>.</>}
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title={`Permutation importance (${r.best_model})`}
          subtitle="How much the test accuracy drops when one measurement is shuffled randomly. A bigger drop means the model needs that measurement more."
          action={<SourceTag kind="test" />}
        >
          <ImportanceChart rows={imp.permutation} color={COLORS.teal} />
          <p className="mt-2 text-xs text-slate-500">
            Error bars show the variation over 20 repeats. Measurements that are strongly correlated (radius, perimeter, area)
            can stand in for each other, so each may look less important on its own.
          </p>
        </Card>

        <Card title="Random Forest feature importance" subtitle="How often and how usefully each measurement is used to split the trees." action={<SourceTag kind="train" />}>
          <ImportanceChart rows={imp.random_forest} color={COLORS.navy} />
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Logistic Regression coefficients"
          subtitle="Positive (purple) values push towards malignant, negative (blue) towards benign. Values are on scaled data so they can be compared."
          action={<SourceTag kind="train" />}
        >
          <ImportanceChart rows={imp.logistic_regression} signed />
        </Card>

        <Card title="What the measurements mean">
          <dl className="space-y-2 text-sm">
            {summary.data!.features.map((f) => (
              <div key={f.key} className="grid grid-cols-[8rem_1fr] gap-2 border-b border-slate-100 pb-2 last:border-0">
                <dt className="font-medium text-navy-900">{f.label}</dt>
                <dd className="text-slate-600">
                  {f.help}.{' '}
                  <span className="text-xs text-slate-500">
                    Average: benign <span className="text-blue-700">{sig(f.benign_mean)}</span>, malignant <span className="text-violet-700">{sig(f.malignant_mean)}</span>.
                  </span>
                </dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>

      <Card title="How to read these charts">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
          <li>The three methods measure different things, so their rankings are not identical. Compare them to find measurements that are important in all of them (for example, concave points).</li>
          <li>Cell radius, perimeter and area are almost the same information (see the correlation heatmap), so the credit can be split between them differently in each method.</li>
          <li>Importance shows how the <em>model</em> behaves. It does not prove that a measurement causes cancer.</li>
          <li>For a single prediction, open the Prediction Lab: the “What influenced this prediction” chart shows which values pushed that result up or down.</li>
        </ul>
      </Card>
    </div>
  )
}
