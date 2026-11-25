import { ExternalLink } from 'lucide-react'
import { Card, ErrorState, Loading, PageHeader } from '../components/ui'
import { api } from '../lib/api'
import { useApi } from '../lib/hooks'

const LIMITATIONS = [
  ['Sampling bias', 'Single institution (University of Wisconsin Hospitals), consecutive patients in the late 1980s–early 1990s; prevalence (37% malignant) reflects a referral FNA population, not a screening population.'],
  ['Generalisation', 'No external or temporal validation. Different imaging equipment, staining, operators or contour-drawing tools would shift feature distributions.'],
  ['Feature availability', 'The 30 inputs need the original interactive nuclear-segmentation workflow; they are not routinely produced in clinical practice today.'],
  ['False negatives', 'A missed malignant sample is the most harmful error type. Even a few percent miss rate is unacceptable for unsupervised clinical use.'],
  ['False positives', 'Lower thresholds raise sensitivity but increase false alarms, which in practice mean anxiety and unnecessary procedures.'],
  ['Calibration', 'Probabilities are calibrated to this dataset\'s class mix. Under a different prevalence they would be systematically wrong.'],
  ['Small test set', '114 held-out samples: a single misclassification moves recall by ~2.4 points. Confidence intervals are wide.'],
  ['Explainability', 'SHAP and permutation importance describe model behaviour; with near-collinear features, attributions are unstable and not causal.'],
  ['Regulatory status', 'Not cleared or approved by any regulator. Not a medical device. Not for diagnosis, screening or treatment decisions.'],
]

export default function Research() {
  const { data, error, loading, reload } = useApi(api.methodology)
  if (loading) return <><PageHeader title="Research & Methodology" /><Loading /></>
  if (error) return <><PageHeader title="Research & Methodology" /><ErrorState error={error} onRetry={reload} /></>
  const d = data!
  const cite = (s: string) => s.replace(/\[([a-z0-9_]+)\]/g, (_, id) => {
    const i = d.citations.findIndex((c) => c.id === id)
    return i >= 0 ? `[${i + 1}]` : `[${id}]`
  })
  return (
    <div className="space-y-6">
      <PageHeader title="Research & Methodology" subtitle={<>Run-specific values on this page are read from the training metadata of <span className="font-mono">{d.model_version}</span>.</>} />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          {d.sections.map((s) => (
            <Card key={s.id} title={s.title}>
              <div className="space-y-2 text-sm leading-relaxed text-slate-700">
                {s.body.map((p, i) => <p key={i}>{cite(p)}</p>)}
              </div>
            </Card>
          ))}
          <Card title="Limitations and ethical considerations">
            <dl className="space-y-3 text-sm">
              {LIMITATIONS.map(([t, b]) => (
                <div key={t}><dt className="font-medium text-navy-900">{t}</dt><dd className="text-slate-600">{b}</dd></div>
              ))}
            </dl>
          </Card>
        </div>
        <div className="space-y-6">
          <Card title="References">
            <ol className="list-decimal space-y-2 pl-5 text-xs text-slate-700">
              {d.citations.map((c) => (
                <li key={c.id}>
                  {c.text}{' '}
                  {c.doi && <span className="text-slate-500">doi:{c.doi} </span>}
                  {c.url && (
                    <a href={c.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-teal-700 underline hover:text-teal-900">
                      link<ExternalLink className="h-3 w-3" aria-hidden />
                    </a>
                  )}
                </li>
              ))}
            </ol>
          </Card>
          <Card title="Reproduce this run">
            <pre className="overflow-x-auto rounded bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100">{`cd backend
python scripts/run_eda.py
python scripts/train_models.py
pytest`}</pre>
            <p className="mt-2 text-xs text-slate-500">Fixed seed (42), pinned library versions in requirements.txt; each run is logged in the experiment tracker.</p>
          </Card>
        </div>
      </div>
    </div>
  )
}
