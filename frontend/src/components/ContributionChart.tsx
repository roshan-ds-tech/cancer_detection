import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { COLORS, sig } from '../lib/format'
import type { Contribution } from '../lib/types'

/** Horizontal bars: how much each measurement moved the malignant probability. */
export function ContributionChart({ items }: { items: Contribution[] }) {
  const data = items.map((c) => ({ ...c, points: +(c.contribution * 100).toFixed(1) }))
  return (
    <div style={{ height: Math.max(240, data.length * 30) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 11 }} unit=" pts" />
          <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
          <ReferenceLine x={0} stroke="#64748b" />
          <Tooltip
            formatter={(v) => [`${Number(v) > 0 ? '+' : ''}${v} percentage points`, 'Effect on malignant chance']}
            labelFormatter={(l, p) => `${l} = ${sig(p?.[0]?.payload?.value)}`}
          />
          <Bar dataKey="points" isAnimationActive={false}>
            {data.map((d) => <Cell key={d.feature} fill={d.points > 0 ? COLORS.malignant : COLORS.benign} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
