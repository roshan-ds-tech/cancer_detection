import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getResults, percent, runModels, type Results } from '../api'

export default function Models() {
  const [data, setData] = useState<Results | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getResults().then(setData).catch(() => setError('Cannot reach the backend. Is it running on port 8000?'))
  }, [])

  async function onRun() {
    setRunning(true)
    setError('')
    try {
      setData(await runModels())
    } catch {
      setError('Training failed. Please check the backend.')
    } finally {
      setRunning(false)
    }
  }

  const best = data?.results.find((r) => r.model === data.best_model)
  const chartData = data?.results.map((r) => ({ name: r.model, accuracy: +(r.accuracy * 100).toFixed(1) })) ?? []

  return (
    <div className="space-y-6">
      <section className="flex flex-wrap items-center justify-between gap-4 rounded-xl bg-white p-6 shadow-sm">
        <div>
          <h2 className="text-lg font-semibold">Run the models on the dataset</h2>
          <p className="mt-1 text-sm text-slate-500">
            Trains 6 machine learning models on 80% of the data and tests them on the other 20%.
            {data && ` Last run: ${data.trained_at} (${data.train_rows} training rows, ${data.test_rows} test rows).`}
          </p>
        </div>
        <button onClick={onRun} disabled={running} className="rounded-md bg-indigo-700 px-6 py-3 font-semibold text-white hover:bg-indigo-800 disabled:opacity-50">
          {running ? 'Running models…' : 'Run models'}
        </button>
      </section>

      {error && <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      {data && best && (
        <>
          <section className="rounded-xl bg-indigo-50 p-6">
            <p className="text-sm text-indigo-700">Best model</p>
            <p className="text-2xl font-bold text-indigo-900">{best.model}</p>
            <p className="mt-1 text-sm text-slate-600">
              Chosen because it had the highest cross-validation accuracy on the training data ({percent(best.cv_accuracy)}).
              On the test data it got <b>{percent(best.accuracy)}</b> accuracy. This model is used on the Predict page.
            </p>
          </section>

          <section className="rounded-xl bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Accuracy of each model on the test data</h2>
            <p className="text-sm text-slate-500">Higher is better. The best model is shown in dark blue.</p>
            <div className="mt-4 h-72">
              <ResponsiveContainer>
                <BarChart data={chartData} margin={{ bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" tick={{ fontSize: 12 }} />
                  <YAxis domain={[80, 100]} unit="%" />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="accuracy" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                    {chartData.map((d) => <Cell key={d.name} fill={d.name === data.best_model ? '#3730a3' : '#a5b4fc'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="rounded-xl bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Comparison table</h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-500">
                  <tr>
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">CV accuracy</th>
                    <th className="py-2 pr-4">Test accuracy</th>
                    <th className="py-2 pr-4">Precision</th>
                    <th className="py-2 pr-4">Recall</th>
                    <th className="py-2 pr-4">F1 score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r) => (
                    <tr key={r.model} className={`border-b border-slate-100 last:border-0 ${r.model === data.best_model ? 'bg-indigo-50 font-semibold' : ''}`}>
                      <td className="py-2 pr-4">{r.model}</td>
                      <td className="py-2 pr-4">{percent(r.cv_accuracy)}</td>
                      <td className="py-2 pr-4">{percent(r.accuracy)}</td>
                      <td className="py-2 pr-4">{percent(r.precision)}</td>
                      <td className="py-2 pr-4">{percent(r.recall)}</td>
                      <td className="py-2 pr-4">{percent(r.f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="mt-4 space-y-1 text-sm text-slate-600">
              <li><b>Accuracy</b>: how many test samples were predicted correctly.</li>
              <li><b>Precision</b>: of the samples predicted malignant, how many really were malignant.</li>
              <li><b>Recall</b>: of the real malignant samples, how many the model found.</li>
              <li><b>F1 score</b>: a balance of precision and recall.</li>
              <li><b>CV accuracy</b>: average accuracy over 5 rounds of cross-validation on the training data.</li>
            </ul>
          </section>

          <div className="grid gap-6 lg:grid-cols-2">
            <section className="rounded-xl bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold">Confusion matrix ({best.model})</h2>
              <p className="text-sm text-slate-500">How the {data.test_rows} test samples were classified.</p>
              <ConfusionMatrix matrix={best.confusion_matrix} />
            </section>

            <section className="rounded-xl bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold">Most important measurements</h2>
              <p className="text-sm text-slate-500">Which measurements the Random Forest model relied on most.</p>
              <div className="mt-4 h-72">
                <ResponsiveContainer>
                  <BarChart data={data.feature_importance} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                    <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  )
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const [[tn, fp], [fn, tp]] = matrix
  const cell = (value: number, label: string, good: boolean) => (
    <div className={`rounded-lg p-4 text-center ${good ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
      <p className="text-3xl font-bold">{value}</p>
      <p className="text-xs">{label}</p>
    </div>
  )
  return (
    <div className="mt-4">
      <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-2 text-sm">
        <span />
        <span className="text-center text-slate-500">Predicted benign</span>
        <span className="text-center text-slate-500">Predicted malignant</span>
        <span className="text-slate-500">Actually benign</span>
        {cell(tn, 'correct', true)}
        {cell(fp, 'false alarm', false)}
        <span className="text-slate-500">Actually malignant</span>
        {cell(fn, 'missed', false)}
        {cell(tp, 'correct', true)}
      </div>
      <p className="mt-3 text-sm text-slate-600">
        Green boxes are correct predictions. Red boxes are mistakes: {fn} malignant sample(s) were missed and {fp} benign sample(s) were wrongly flagged.
      </p>
    </div>
  )
}
