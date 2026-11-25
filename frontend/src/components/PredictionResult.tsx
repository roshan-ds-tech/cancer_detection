import { AlertTriangle, CheckCircle2, Info } from 'lucide-react'
import type { ReactNode } from 'react'
import { LEVEL_STYLES, pct, prob } from '../lib/format'
import type { Prediction } from '../lib/types'
import { ContributionChart } from './ContributionChart'
import { Disclaimer } from './Disclaimer'
import { Card } from './ui'

export function PredictionResult({ res, extra }: { res: Prediction; extra?: ReactNode }) {
  const style = LEVEL_STYLES[res.level]
  const p = res.malignant_probability
  return (
    <div className="space-y-4" data-testid="prediction-result">
      <Disclaimer variant="inline" />
      <Card title="Prediction result" subtitle={`Model used: ${res.model}`}>
        <div className={`flex flex-wrap items-center justify-between gap-3 rounded-lg p-4 ring-1 ring-inset ${style.bg} ${style.ring}`}>
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">Prediction level</div>
            <div className={`text-3xl font-bold ${style.text}`}>{res.level}</div>
          </div>
          <div className="flex gap-1.5 text-[11px]">
            {(['Low', 'Medium', 'High'] as const).map((l) => (
              <span key={l} className={`rounded-md px-2 py-1 ${l === res.level ? `${LEVEL_STYLES[l].bar} font-semibold text-white` : 'bg-white text-slate-500 ring-1 ring-slate-200'}`}>
                {l} {l === 'Low' ? '< 30%' : l === 'Medium' ? '30–70%' : '≥ 70%'}
              </span>
            ))}
          </div>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">Chance of malignant</div>
            <div className="mt-1 text-2xl font-semibold tabular-nums text-navy-900">{prob(p)}</div>
            <div className="relative mt-2 h-2.5 w-full rounded-full bg-slate-100" aria-hidden>
              <div className={`h-2.5 rounded-full ${style.bar}`} style={{ width: `${Math.max(1, p * 100)}%` }} />
              <span className="absolute top-0 h-2.5 w-px bg-slate-400" style={{ left: '30%' }} />
              <span className="absolute top-0 h-2.5 w-px bg-slate-400" style={{ left: '70%' }} />
            </div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">Predicted class</div>
            <div className={`mt-1 text-2xl font-semibold ${res.prediction === 'Malignant' ? 'text-violet-700' : 'text-blue-700'}`}>
              {res.prediction}
            </div>
            <div className="text-xs text-slate-500">
              {res.prediction === 'Malignant' ? 'Cancerous' : 'Not cancerous'} · chosen when the chance is {res.prediction === 'Malignant' ? '≥' : '<'} 50%
            </div>
          </div>
        </div>

        <div className="mt-4">
          {res.unusual_values.length === 0 ? (
            <p className="flex items-center gap-2 text-sm text-teal-800">
              <CheckCircle2 className="h-4 w-4" aria-hidden /> All values are inside the range seen in the dataset.
            </p>
          ) : (
            <p className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <span>
                These values are outside the range seen in the dataset, so the result may not be reliable:{' '}
                <b>{res.unusual_values.join(', ')}</b>.
              </span>
            </p>
          )}
        </div>
        {extra && <div className="mt-4 border-t border-slate-100 pt-4">{extra}</div>}
      </Card>

      {res.explanation && (
        <Card
          title="What influenced this prediction"
          subtitle="Shapley values: each measurement's fair share of the difference between this sample and an average sample."
        >
          <p className="mb-3 text-xs text-slate-600">
            Average sample {pct(res.explanation.base_value)}{' '}
            {res.explanation.items.reduce((a, c) => a + c.contribution, 0) >= 0 ? '+' : '−'}{' '}
            {(Math.abs(res.explanation.items.reduce((a, c) => a + c.contribution, 0)) * 100).toFixed(1)} points from the
            measurements = {prob(p)} for this sample.
          </p>
          <ContributionChart items={res.explanation.items} />
          <p className="mt-3 flex items-start gap-1.5 text-xs text-slate-500">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
            Purple bars push the prediction towards malignant, blue bars towards benign. This shows how the model behaves,
            not the biological cause.
          </p>
        </Card>
      )}
    </div>
  )
}
