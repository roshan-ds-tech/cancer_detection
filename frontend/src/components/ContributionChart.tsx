import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { COLORS, prettyFeature, sig } from '../lib/format'
import type { Contribution } from '../lib/types'

/** Horizontal bar chart of SHAP contributions (probability units). */
export function ContributionChart({ items }: { items: Contribution[] }) {
  const data = items.map((c) => ({ ...c, label: prettyFeature(c.feature) }))
  return (
    <div style={{ height: Math.max(220, data.length * 28) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 11 }} />
          <ReferenceLine x={0} stroke="#64748b" />
          <Tooltip
            formatter={(v) => [Number(v).toFixed(4), 'Δ P(malignant)']}
            labelFormatter={(l, p) => `${l} = ${sig(p?.[0]?.payload?.value)}`}
          />
          <Bar dataKey="contribution" isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.feature} fill={d.contribution > 0 ? COLORS.malignant : COLORS.benign} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
