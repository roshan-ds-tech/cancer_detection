import { useEffect, useState } from 'react'
import { getDataset, predict, type Feature, type Prediction } from '../api'

const LEVEL_STYLE = {
  Low: { color: 'bg-green-500', text: 'text-green-700', box: 'bg-green-50 border-green-200' },
  Medium: { color: 'bg-amber-500', text: 'text-amber-700', box: 'bg-amber-50 border-amber-200' },
  High: { color: 'bg-red-500', text: 'text-red-700', box: 'bg-red-50 border-red-200' },
}

export default function Predict() {
  const [features, setFeatures] = useState<Feature[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getDataset().then((d) => setFeatures(d.features)).catch(() => setError('Cannot reach the backend. Is it running on port 8000?'))
  }, [])

  function fill(kind: 'benign_example' | 'malignant_example') {
    setValues(Object.fromEntries(features.map((f) => [f.key, String(f[kind])])))
    setResult(null)
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    const numbers: Record<string, number> = {}
    for (const f of features) {
      const n = Number(values[f.key])
      if (values[f.key] === undefined || values[f.key].trim() === '' || Number.isNaN(n) || n < 0) {
        setError(`Please enter a valid number (0 or more) for "${f.label}".`)
        return
      }
      numbers[f.key] = n
    }
    setLoading(true)
    try {
      setResult(await predict(numbers))
    } catch {
      setError('Prediction failed. Please check the backend.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <form onSubmit={onSubmit} className="rounded-xl bg-white p-6 shadow-sm lg:col-span-3">
        <h2 className="text-lg font-semibold">Enter the cell measurements</h2>
        <p className="mt-1 text-sm text-slate-500">
          These values come from a microscope image of a breast tissue sample. Not sure what to enter? Use an example button.
        </p>

        <div className="mt-4 flex flex-wrap gap-2">
          <button type="button" onClick={() => fill('benign_example')} className="rounded-md border border-green-300 px-3 py-1.5 text-sm text-green-700 hover:bg-green-50">
            Fill benign example
          </button>
          <button type="button" onClick={() => fill('malignant_example')} className="rounded-md border border-red-300 px-3 py-1.5 text-sm text-red-700 hover:bg-red-50">
            Fill malignant example
          </button>
          <button type="button" onClick={() => { setValues({}); setResult(null) }} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">
            Clear
          </button>
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          {features.map((f) => (
            <label key={f.key} className="block">
              <span className="text-sm font-medium">{f.label}</span>
              <input
                type="number"
                step="any"
                min="0"
                value={values[f.key] ?? ''}
                onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
                placeholder={`e.g. ${f.benign_example}`}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 focus:border-indigo-500 focus:outline-none"
              />
              <span className="mt-1 block text-xs text-slate-500">
                {f.help}. Range in dataset: {f.min} – {f.max}
              </span>
            </label>
          ))}
        </div>

        {error && <p className="mt-4 rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p>}

        <button type="submit" disabled={loading || features.length === 0} className="mt-6 w-full rounded-md bg-indigo-700 py-3 font-semibold text-white hover:bg-indigo-800 disabled:opacity-50">
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      <div className="lg:col-span-2">
        <ResultCard result={result} />
      </div>
    </div>
  )
}

function ResultCard({ result }: { result: Prediction | null }) {
  if (!result) {
    return (
      <div className="rounded-xl border-2 border-dashed border-slate-300 p-8 text-center text-slate-500">
        Fill in the values and press <b>Predict</b> to see the result here.
      </div>
    )
  }
  const style = LEVEL_STYLE[result.level]
  return (
    <div className={`rounded-xl border p-6 ${style.box}`} data-testid="result">
      <p className="text-sm text-slate-600">Prediction level</p>
      <p className={`text-4xl font-bold ${style.text}`}>{result.level}</p>

      <p className="mt-4 text-sm text-slate-600">Chance that the sample is malignant (cancerous)</p>
      <p className="text-2xl font-semibold">
        {result.malignant_probability >= 99.9 ? 'more than 99.9' : result.malignant_probability <= 0.1 ? 'less than 0.1' : result.malignant_probability}%
      </p>
      <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-white">
        <div className={`h-3 ${style.color}`} style={{ width: `${result.malignant_probability}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>Low (&lt;30%)</span><span>Medium</span><span>High (≥70%)</span>
      </div>

      <p className="mt-5">
        Predicted class: <b>{result.prediction}</b>{' '}
        <span className="text-slate-500">({result.prediction === 'Benign' ? 'not cancerous' : 'cancerous'})</span>
      </p>
      <p className="text-sm text-slate-500">Model used: {result.model}</p>

      {result.unusual_values.length > 0 && (
        <p className="mt-4 rounded-md bg-amber-100 p-3 text-sm text-amber-800">
          Warning: these values are outside the range seen in the dataset, so the result may not be reliable:{' '}
          {result.unusual_values.join(', ')}.
        </p>
      )}

      <p className="mt-5 text-xs text-slate-500">
        This is a student machine learning project, not a medical diagnosis. Always consult a doctor.
      </p>
    </div>
  )
}
