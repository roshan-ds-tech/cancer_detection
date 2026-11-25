import { AlertTriangle, CheckCircle2, Info, PauseCircle } from 'lucide-react'
import { pct, prettyFeature, sig } from '../lib/format'
import type { PredictResponse } from '../lib/types'
import { ContributionChart } from './ContributionChart'
import { Disclaimer } from './Disclaimer'
import { Badge, Card } from './ui'

function DistributionNotice({ res }: { res: PredictResponse }) {
  const d = res.distribution_check
  if (d.status === 'ok') {
    return (
      <p className="flex items-center gap-2 text-sm text-teal-800">
        <CheckCircle2 className="h-4 w-4" aria-hidden /> Input is within the typical range of the training data.
      </p>
    )
  }
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
      <div className="flex items-center gap-2 font-medium">
        <AlertTriangle className="h-4 w-4" aria-hidden />
        {d.status === 'abstain' ? 'Input far outside the training distribution' : 'Unusual input — interpret with extra caution'}
      </div>
      <ul className="mt-1 list-disc pl-6 text-xs">
        {d.messages.map((m) => <li key={m}>{m}</li>)}
      </ul>
      {d.flagged_features.length > 0 && (
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer">Flagged features ({d.flagged_features.length})</summary>
          <ul className="mt-1 space-y-0.5">
            {d.flagged_features.map((f) => (
              <li key={f.feature}>
                {prettyFeature(f.feature)} = {sig(f.value)} (training range {sig(f.train_min)}–{sig(f.train_max)}, robust z {f.robust_z.toFixed(1)}) — {f.severity}
              </li>
            ))}
          </ul>
        </details>
      )}
      <p className="mt-2 text-[11px] text-amber-800">{d.note}</p>
    </div>
  )
}

export function PredictionResult({ res, groundTruth }: { res: PredictResponse; groundTruth?: React.ReactNode }) {
  const p = res.prediction
  return (
    <div className="space-y-4" data-testid="prediction-result">
      <Disclaimer variant="inline" />
      <Card title="Model output" subtitle={`${res.model_name} · ${res.model_version}`}>
        {res.abstained || !p ? (
          <div className="flex items-start gap-3">
            <PauseCircle className="mt-0.5 h-6 w-6 text-slate-500" aria-hidden />
            <div>
              <div className="font-semibold text-navy-900">The model abstained</div>
              <p className="text-sm text-slate-600">
                No prediction is shown because the input is too far from the data the model was trained on.
              </p>
              <ul className="mt-1 list-disc pl-5 text-xs text-slate-600">
                {res.abstain_reason.map((r) => <li key={r}>{r}</li>)}
              </ul>
            </div>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-3">
            <div>
              <div className="text-xs uppercase tracking-wide text-slate-500">Predicted dataset class</div>
              <div className={`mt-1 text-2xl font-semibold capitalize ${p.predicted_class === 'malignant' ? 'text-violet-700' : 'text-blue-700'}`}>
                {p.predicted_class}
              </div>
              <div className="mt-1 text-xs text-slate-500">at operating threshold {p.operating_threshold}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-slate-500">Model probability for “malignant” class</div>
              <div className="mt-1 text-2xl font-semibold tabular-nums text-navy-900">{pct(p.probability_malignant, 2)}</div>
              <div className="mt-2 h-2 w-full rounded-full bg-slate-100" aria-hidden>
                <div className="relative h-2 rounded-full bg-gradient-to-r from-blue-400 to-violet-500" style={{ width: `${Math.max(1, p.probability_malignant * 100)}%` }} />
              </div>
              <div className="relative h-3">
                <span className="absolute top-0 h-3 w-px bg-navy-900" style={{ left: `${p.operating_threshold * 100}%` }} title="Operating threshold" />
              </div>
            </div>
            <div className="text-xs text-slate-600">
              <div className="flex items-center gap-1 font-medium text-slate-700"><Info className="h-3.5 w-3.5" aria-hidden /> How to read this</div>
              <p className="mt-1">{p.probability_note}</p>
              <p className="mt-1">This is a dataset-specific model estimate, not a person's probability of having cancer.</p>
            </div>
          </div>
        )}
        <div className="mt-4 flex flex-wrap gap-2 text-xs">
          <Badge tone="teal">Input validation: {res.input_validation}</Badge>
          <Badge tone={res.distribution_check.status === 'ok' ? 'teal' : 'amber'}>Distribution check: {res.distribution_check.status}</Badge>
          <Badge>Mahalanobis {res.distribution_check.mahalanobis_distance.toFixed(1)} (train p99 {res.distribution_check.mahalanobis_p99_train.toFixed(1)})</Badge>
        </div>
        <div className="mt-3"><DistributionNotice res={res} /></div>
        {groundTruth && <div className="mt-4 border-t border-slate-100 pt-4">{groundTruth}</div>}
      </Card>

      {res.explanation && (
        <Card
          title="Why the model produced this output"
          subtitle={`${res.explanation.method}. Output space: ${res.explanation.output_space}.`}
        >
          <p className="mb-3 text-xs text-slate-600">
            Baseline (average model output over background) {pct(res.explanation.base_value, 1)} + sum of contributions{' '}
            {res.explanation.sum_of_contributions >= 0 ? '+' : ''}{(res.explanation.sum_of_contributions * 100).toFixed(1)} pts
            {p && <> = {pct(p.probability_malignant, 1)}</>}. Purple bars push towards the malignant class, blue towards benign.
          </p>
          <ContributionChart items={res.explanation.top_contributions} />
          <p className="mt-3 text-xs text-slate-500">
            Attributions describe this statistical model's behaviour, not biological causation. Correlated features (radius, perimeter, area) can share credit unpredictably.
          </p>
        </Card>
      )}

      <Card title="Limitations">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
          {res.limitations.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </Card>
    </div>
  )
}
