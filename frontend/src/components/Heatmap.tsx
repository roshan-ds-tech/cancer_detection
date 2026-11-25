import { useState } from 'react'
import { prettyFeature } from '../lib/format'

// Diverging blue (negative) – white – purple (positive)
function color(r: number) {
  const t = Math.max(-1, Math.min(1, r))
  const [lo, hi] = [[59, 130, 246], [124, 58, 237]]
  const c = t < 0 ? lo : hi
  const a = Math.abs(t)
  const mix = (x: number) => Math.round(255 + (x - 255) * a)
  return `rgb(${mix(c[0])},${mix(c[1])},${mix(c[2])})`
}

/** Correlation matrix as a CSS grid; hover/focus shows the exact value. */
export function Heatmap({ features, matrix }: { features: string[]; matrix: number[][] }) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)
  const n = features.length
  return (
    <div>
      <div className="mb-2 h-5 text-xs text-slate-600" aria-live="polite">
        {hover
          ? `${prettyFeature(features[hover.i])} × ${prettyFeature(features[hover.j])}: r = ${matrix[hover.i][hover.j].toFixed(3)}`
          : 'Hover a cell to read the Pearson correlation (training split).'}
      </div>
      <div className="overflow-x-auto">
        <div
          role="img"
          aria-label="Feature correlation heatmap"
          className="grid gap-px"
          style={{ gridTemplateColumns: `9rem repeat(${n}, minmax(12px, 1fr))`, minWidth: 560 }}
          onMouseLeave={() => setHover(null)}
        >
          {matrix.map((row, i) => (
            <div key={i} className="contents">
              <div className="truncate pr-2 text-right text-[10px] leading-[14px] text-slate-600" title={features[i]}>
                {prettyFeature(features[i])}
              </div>
              {row.map((r, j) => (
                <div
                  key={j}
                  onMouseEnter={() => setHover({ i, j })}
                  className={`aspect-square ${hover && (hover.i === i || hover.j === j) ? 'ring-1 ring-navy-900/40' : ''}`}
                  style={{ background: color(r) }}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2 text-[11px] text-slate-500">
        <span>−1</span>
        <div className="h-2 w-40 rounded" style={{ background: `linear-gradient(90deg, ${color(-1)}, #fff, ${color(1)})` }} />
        <span>+1</span>
      </div>
    </div>
  )
}
