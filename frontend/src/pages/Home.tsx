import { Activity, ArrowRight, BookOpen, Brain, Database, FlaskConical, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, ErrorState, Loading, SourceTag, Stat } from '../components/ui'
import { api } from '../lib/api'
import { pct } from '../lib/format'
import { useApi } from '../lib/hooks'

const PAGES = [
  { to: '/dataset', icon: Database, title: 'Dataset Explorer', text: 'Class balance, feature statistics, histograms, correlations and the raw records.' },
  { to: '/models', icon: Activity, title: 'Model Performance', text: 'Run and compare 6 tuned models with accuracy, ROC curves, confusion matrices and run history.' },
  { to: '/predict', icon: FlaskConical, title: 'Prediction Lab', text: 'Enter 10 measurements by hand, try the test samples, or upload a CSV file.' },
  { to: '/explain', icon: Brain, title: 'Explainable AI', text: 'See which measurements matter most to the models and why.' },
  { to: '/about', icon: BookOpen, title: 'About & Methods', text: 'How the project works, what the metrics mean, limitations and references.' },
]

function LiveSnapshot() {
  const summary = useApi(api.summary)
  const results = useApi(api.results)
  if (summary.loading || results.loading) return <Loading label="Loading live results from the API…" />
  const err = summary.error ?? results.error
  if (err) return <ErrorState error={err} onRetry={() => { summary.reload(); results.reload() }} />
  const r = results.data!
  const best = r.results.find((m) => m.model === r.best_model)!
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Stat label="Samples" value={summary.data!.total} hint={`${summary.data!.malignant} malignant · ${summary.data!.benign} benign`} />
      <Stat label="Best model" value={<span className="text-lg">{r.best_model}</span>} hint={`${r.results.length} models compared`} />
      <Stat label="CV accuracy" value={pct(best.cv_accuracy)} hint={<SourceTag kind="cv" />} />
      <Stat label="Test accuracy" value={pct(best.accuracy)} hint={<SourceTag kind="test" />} />
    </div>
  )
}

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="overflow-hidden rounded-2xl bg-gradient-to-br from-navy-900 via-navy-800 to-indigo-900 p-6 text-white md:p-10">
        <p className="text-xs font-semibold uppercase tracking-widest text-teal-300">Machine learning project</p>
        <h1 className="mt-2 max-w-3xl text-3xl font-semibold tracking-tight md:text-4xl">
          Breast cancer classification with explainable machine learning
        </h1>
        <p className="mt-4 max-w-2xl text-sm text-slate-200 md:text-base">
          OncoSense AI trains and compares 6 machine learning models on the Breast Cancer Wisconsin (Diagnostic) dataset.
          It uses 10 measurements of cell nuclei from microscope images to predict whether a sample is benign or
          malignant, and explains what influenced each prediction.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/predict" className="inline-flex items-center gap-1.5 rounded-lg bg-teal-500 px-4 py-2 text-sm font-semibold text-navy-950 hover:bg-teal-400">
            Make a prediction <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <Link to="/models" className="inline-flex items-center gap-1.5 rounded-lg border border-white/30 px-4 py-2 text-sm font-semibold text-white hover:bg-white/10">
            Compare the models
          </Link>
        </div>
      </section>

      <LiveSnapshot />

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {PAGES.map(({ to, icon: Icon, title, text }) => (
          <Link key={to} to={to} className="group rounded-xl border border-slate-200 bg-white p-5 shadow-sm hover:border-teal-300">
            <Icon className="h-5 w-5 text-teal-600" aria-hidden />
            <h2 className="mt-3 font-semibold text-navy-900 group-hover:text-teal-700">{title}</h2>
            <p className="mt-1 text-sm text-slate-600">{text}</p>
          </Link>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="How it works">
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-700">
            <li>The 569 samples are split into 80% training data and 20% test data (stratified).</li>
            <li>6 models are trained: Logistic Regression, Decision Tree, Random Forest, KNN, SVM and Naive Bayes.</li>
            <li>Each model's settings are tuned with grid search and 5-fold cross-validation on the training data.</li>
            <li>The model with the best cross-validation accuracy is chosen. The test data is not used for this choice.</li>
            <li>All models are evaluated once on the test data.</li>
            <li>The chosen model turns its output into a Low / Medium / High prediction level.</li>
          </ol>
        </Card>
        <Card title={<span className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-indigo-600" aria-hidden /> Important to know</span>}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
            <li>This is an educational project, not a medical device. It must not be used to diagnose anyone.</li>
            <li>The prediction is about the dataset class, not a person's real chance of having cancer.</li>
            <li>The data is 569 samples from one hospital in the early 1990s.</li>
            <li>The measurements come from a special image analysis process, not from a normal medical report.</li>
            <li>Do not enter real patient data. Uploaded files are not stored.</li>
          </ul>
        </Card>
      </div>
    </div>
  )
}
