import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { getDataset, type Dataset as DatasetData } from '../api'

const BENIGN = '#22c55e'
const MALIGNANT = '#ef4444'

export default function Dataset() {
  const [data, setData] = useState<DatasetData | null>(null)
  const [featureKey, setFeatureKey] = useState('radius')
  const [error, setError] = useState('')

  useEffect(() => {
    getDataset().then(setData).catch(() => setError('Cannot reach the backend. Is it running on port 8000?'))
  }, [])

  if (error) return <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p>
  if (!data) return <p className="text-slate-500">Loading…</p>

  const feature = data.features.find((f) => f.key === featureKey)!
  const pieData = [
    { name: 'Benign', value: data.benign },
    { name: 'Malignant', value: data.malignant },
  ]
  const avgData = [
    { name: 'Benign', value: feature.benign_avg },
    { name: 'Malignant', value: feature.malignant_avg },
  ]

  return (
    <div className="space-y-6">
      <section className="rounded-xl bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold">About the dataset</h2>
        <p className="mt-1 text-sm text-slate-600">
          The <b>Breast Cancer Wisconsin (Diagnostic)</b> dataset contains measurements of cell nuclei taken from
          microscope images of breast tissue samples. Each sample is labelled <b>benign</b> (not cancerous) or{' '}
          <b>malignant</b> (cancerous). This project uses 10 of its measurements.
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <Stat label="Total samples" value={data.total} color="text-slate-800" />
          <Stat label="Benign" value={data.benign} color="text-green-600" />
          <Stat label="Malignant" value={data.malignant} color="text-red-600" />
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-xl bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold">Benign vs malignant</h2>
          <p className="text-sm text-slate-500">How many samples are in each class.</p>
          <div className="h-72">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={100} label isAnimationActive={false}>
                  <Cell fill={BENIGN} />
                  <Cell fill={MALIGNANT} />
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-xl bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold">Average value by class</h2>
            <select value={featureKey} onChange={(e) => setFeatureKey(e.target.value)} className="rounded-md border border-slate-300 px-2 py-1 text-sm">
              {data.features.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
            </select>
          </div>
          <p className="text-sm text-slate-500">{feature.help}.</p>
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={avgData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" name={feature.label} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  <Cell fill={BENIGN} />
                  <Cell fill={MALIGNANT} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="rounded-xl bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold">Cell radius vs texture</h2>
        <p className="text-sm text-slate-500">
          Each dot is one sample. Malignant samples (red) usually have bigger cells, which is why the models can tell them apart.
        </p>
        <div className="mt-4 h-80">
          <ResponsiveContainer>
            <ScatterChart margin={{ bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="radius" name="Cell radius" label={{ value: 'Cell radius', position: 'bottom' }} domain={['auto', 'auto']} />
              <YAxis type="number" dataKey="texture" name="Texture" label={{ value: 'Texture', angle: -90, position: 'insideLeft' }} domain={['auto', 'auto']} />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Scatter name="Benign" data={data.scatter.filter((p) => p.diagnosis === 'Benign')} fill={BENIGN} fillOpacity={0.6} isAnimationActive={false} />
              <Scatter name="Malignant" data={data.scatter.filter((p) => p.diagnosis === 'Malignant')} fill={MALIGNANT} fillOpacity={0.6} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-xl bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold">First 10 rows of the data</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-slate-500">
              <tr>
                {data.features.map((f) => <th key={f.key} className="whitespace-nowrap py-2 pr-4">{f.label}</th>)}
                <th className="py-2 pr-4">Diagnosis</th>
              </tr>
            </thead>
            <tbody>
              {data.sample_rows.map((row, i) => (
                <tr key={i} className="border-b border-slate-100 last:border-0">
                  {data.features.map((f) => <td key={f.key} className="py-2 pr-4">{row[f.key]}</td>)}
                  <td className={`py-2 pr-4 font-medium ${row.diagnosis === 'Benign' ? 'text-green-600' : 'text-red-600'}`}>{row.diagnosis}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4 text-center">
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
      <p className="text-sm text-slate-500">{label}</p>
    </div>
  )
}
