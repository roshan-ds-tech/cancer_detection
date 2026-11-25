import {
  Activity,
  BookOpen,
  Brain,
  Database,
  FlaskConical,
  Home,
  Menu,
  X,
} from 'lucide-react'
import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { api } from '../lib/api'
import { useApi } from '../lib/hooks'
import { Disclaimer } from './Disclaimer'

const NAV = [
  { to: '/', label: 'Overview', icon: Home },
  { to: '/dataset', label: 'Dataset Explorer', icon: Database },
  { to: '/models', label: 'Model Performance', icon: Activity },
  { to: '/predict', label: 'Prediction Lab', icon: FlaskConical },
  { to: '/explain', label: 'Explainable AI', icon: Brain },
  { to: '/about', label: 'About & Methods', icon: BookOpen },
]

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <img src="/favicon.svg" alt="" className="h-8 w-8" />
      <div>
        <div className="text-sm font-semibold leading-tight text-white">OncoSense AI</div>
        <div className="text-[11px] leading-tight text-slate-300">Breast cancer ML project</div>
      </div>
    </div>
  )
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <ul className="space-y-1">
      {NAV.map(({ to, label, icon: Icon }) => (
        <li key={to}>
          <NavLink
            to={to}
            end={to === '/'}
            onClick={onNavigate}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium ${
                isActive ? 'bg-white/10 text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white'
              }`
            }
          >
            <Icon className="h-4 w-4" aria-hidden />
            {label}
          </NavLink>
        </li>
      ))}
    </ul>
  )
}

function ModelStatus() {
  const { pathname } = useLocation()
  const { data, error } = useApi(api.health, [pathname])
  const ok = data?.status === 'ok'
  return (
    <div className="rounded-lg bg-white/5 p-3 text-[11px] text-slate-300">
      <div className="flex items-center gap-1.5">
        <span className={`h-2 w-2 rounded-full ${ok ? 'bg-teal-400' : error ? 'bg-amber-400' : 'bg-slate-500'}`} aria-hidden />
        {ok ? 'API connected' : error ? 'API unreachable' : 'Checking API…'}
      </div>
      {data && <div className="mt-1 text-[10px] text-slate-400">Model: {data.best_model}<br />Trained: {data.trained_at}</div>}
    </div>
  )
}

export function Layout() {
  const [open, setOpen] = useState(false)
  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2">
        Skip to content
      </a>
      <aside className="fixed inset-y-0 left-0 hidden w-64 flex-col gap-6 bg-navy-900 p-4 lg:flex" aria-label="Primary">
        <Logo />
        <nav aria-label="Main navigation" className="flex-1">
          <NavItems />
        </nav>
        <ModelStatus />
      </aside>

      <div className="flex items-center justify-between bg-navy-900 px-4 py-3 lg:hidden">
        <Logo />
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls="mobile-nav"
          aria-label={open ? 'Close navigation' : 'Open navigation'}
          className="rounded-md p-2 text-white hover:bg-white/10"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Main navigation" className="bg-navy-900 px-4 pb-4 lg:hidden">
          <NavItems onNavigate={() => setOpen(false)} />
        </nav>
      )}

      <div className="lg:pl-64">
        <Disclaimer />
        <main id="main" className="mx-auto max-w-7xl px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </main>
        <footer className="mx-auto max-w-7xl px-4 pb-8 text-xs text-slate-500 md:px-8">
          Data: Breast Cancer Wisconsin (Diagnostic), Wolberg, Street &amp; Mangasarian — UCI ML Repository, CC BY 4.0
          (doi:10.24432/C5DW2B). Not a medical device.
        </footer>
      </div>
    </div>
  )
}
