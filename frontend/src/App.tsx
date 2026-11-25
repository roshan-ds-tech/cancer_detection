import { NavLink, Route, Routes } from 'react-router-dom'
import Dataset from './pages/Dataset'
import Models from './pages/Models'
import Predict from './pages/Predict'

const links = [
  { to: '/', label: 'Predict' },
  { to: '/models', label: 'Models' },
  { to: '/dataset', label: 'Dataset' },
]

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="bg-indigo-700 text-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <h1 className="text-xl font-bold">Breast Cancer Prediction</h1>
          <nav className="flex gap-1">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end
                className={({ isActive }) =>
                  `rounded-md px-4 py-2 text-sm font-medium ${isActive ? 'bg-white text-indigo-700' : 'hover:bg-indigo-600'}`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/models" element={<Models />} />
          <Route path="/dataset" element={<Dataset />} />
        </Routes>
      </main>

      <footer className="pb-8 text-center text-xs text-slate-500">
        College machine learning project using the Breast Cancer Wisconsin dataset. For learning only, not a medical diagnosis.
      </footer>
    </div>
  )
}
