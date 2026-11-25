import { lazy, Suspense } from 'react'
import { Link, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Loading } from './components/ui'
import Home from './pages/Home'

const Dataset = lazy(() => import('./pages/Dataset'))
const Models = lazy(() => import('./pages/Models'))
const Predict = lazy(() => import('./pages/Predict'))
const Explain = lazy(() => import('./pages/Explain'))
const About = lazy(() => import('./pages/About'))

function NotFound() {
  return (
    <div className="py-20 text-center">
      <h1 className="text-2xl font-semibold text-navy-900">Page not found</h1>
      <Link to="/" className="mt-4 inline-block text-teal-700 underline">Back to overview</Link>
    </div>
  )
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="dataset" element={<Dataset />} />
          <Route path="models" element={<Models />} />
          <Route path="predict" element={<Predict />} />
          <Route path="explain" element={<Explain />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  )
}
