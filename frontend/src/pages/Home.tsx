import { Activity, ArrowRight, BookOpen, Brain, Database, FlaskConical, History, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, ErrorState, Loading, SourceTag, Stat } from '../components/ui'
import { api } from '../lib/api'
import { num, pct } from '../lib/format'
import { useApi } from '../lib/hooks'

const FEATURES = [
  { to: '/dataset', icon: Database, title: 'Dataset Explorer', text: 'Class balance, feature statistics, correlations, distributions and exploratory PCA.' },
  { to: '/models', icon: Activity, title: 'Model Comparison Lab', text: 'Seven algorithms compared with nested cross-validation, ROC, confusion matrices and threshold trade-offs.' },
  { to: '/predict', icon: FlaskConical, title: 'Prediction Lab', text: 'Held-out sample explorer, manual 30-feature input and validated CSV batch inference.' },
  { to: '/explain', icon: Brain, title: 'Explainable AI', text: 'Permutation importance, SHAP attributions and model-native importances.' },
  { to: '/experiments', icon: History, title: 'Experiment Tracker', text: 'Every training run logged to SQLite with hyperparameters, seeds and metrics.' },
  { to: '/research', icon: BookOpen, title: 'Research & Methods', text: 'Provenance, preprocessing, evaluation design, limitations and citations.' },
]

function LiveSnapshot() {
  const summary = useApi(api.datasetSummary)
  const models = useApi(api.models)
  const metrics = useApi(api.metrics)
  if (summary.loading || models.loading || metrics.loading) return <Loading label="Loading live results from the API…" />
  const err = summary.error ?? models.error ?? metrics.error
  if (err) return <ErrorState error={err} onRetry={() => { summary.reload(); models.reload(); metrics.reload() }} />
  const sel = metrics.data!.models.find((m) => m.key === metrics.data!.selected_key)!
  const selName = models.data!.models.find((m) => m.selected)?.name
  const ds = summary.data!.dataset
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Stat label="Samples" value={ds.n_samples} hint={`${summary.data!.integrity?.class_counts.malignant} malignant · ${summary.data!.integrity?.class_counts.benign} benign`} />
      <Stat label="Selected model" value={<span className="text-lg">{selName}</span>} hint={`${models.data!.models.length} candidates compared`} />
      <Stat label="CV ROC-AUC" value={num(sel.cv?.roc_auc.mean, 3)} hint={<SourceTag kind="cv" />} />
      <Stat label="Test recall (malignant)" value={pct(sel.test_at_operating_threshold?.recall)} hint={<SourceTag kind="test" />} />
    </div>
  )
}

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="overflow-hidden rounded-2xl bg-gradient-to-br from-navy-900 via-navy-800 to-indigo-900 p-6 text-white md:p-10">
        <p className="text-xs font-semibold uppercase tracking-widest text-teal-300">Educational ML research prototype</p>
        <h1 className="mt-2 max-w-3xl text-3xl font-semibold tracking-tight md:text-4xl">
          Transparent, explainable classification of breast-mass FNA measurements
        </h1>
        <p className="mt-4 max-w-2xl text-sm text-slate-200 md:text-base">
          OncoSense AI demonstrates a rigorous, leakage-aware machine-learning workflow on the public Breast Cancer Wisconsin
          (Diagnostic) dataset: 30 numeric descriptors of cell nuclei computed from digitised fine-needle aspirate images.
          Every number in this app comes from a real training and evaluation run.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/models" className="inline-flex items-center gap-1.5 rounded-lg bg-teal-500 px-4 py-2 text-sm font-semibold text-navy-950 hover:bg-teal-400">
            Explore model results <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <Link to="/predict" className="inline-flex items-center gap-1.5 rounded-lg border border-white/30 px-4 py-2 text-sm font-semibold text-white hover:bg-white/10">
            Open Prediction Lab
          </Link>
        </div>
      </section>

      <LiveSnapshot />

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map(({ to, icon: Icon, title, text }) => (
          <Link key={to} to={to} className="group rounded-xl border border-slate-200 bg-white p-5 shadow-sm hover:border-teal-300">
            <Icon className="h-5 w-5 text-teal-600" aria-hidden />
            <h2 className="mt-3 font-semibold text-navy-900 group-hover:text-teal-700">{title}</h2>
            <p className="mt-1 text-sm text-slate-600">{text}</p>
          </Link>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Methodology at a glance">
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-700">
            <li>Stratified 80/20 split; the 20% test set is locked away until a model is chosen.</li>
            <li>Seven candidate algorithms, each a scikit-learn Pipeline (scaling fitted inside training folds only).</li>
            <li>Nested cross-validation: inner grid search tunes, outer folds estimate performance.</li>
            <li>Selection by CV ROC-AUC with a one-standard-error rule and Brier score tie-break.</li>
            <li>Operating threshold chosen from out-of-fold training predictions to favour sensitivity.</li>
            <li>One final evaluation on the held-out test set, with bootstrap confidence intervals.</li>
          </ol>
        </Card>
        <Card title={<span className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-indigo-600" aria-hidden /> Research-only boundaries</span>}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
            <li>Not a medical device, screening service or diagnostic tool. Not validated for clinical use.</li>
            <li>Outputs are predictions of a <em>dataset class</em>, not a person's probability of having cancer.</li>
            <li>569 samples from one institution in the early 1990s; no external validation.</li>
            <li>The 30 inputs require a specific research image-analysis process; they are not available to ordinary users.</li>
            <li>Do not enter real patient data. No uploads are stored.</li>
          </ul>
        </Card>
      </div>
    </div>
  )
}
