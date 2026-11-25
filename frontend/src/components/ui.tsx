import { AlertCircle, Inbox, Loader2, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ApiError } from '../lib/api'

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-navy-900 md:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-slate-600">{subtitle}</p>}
      </div>
      {children}
    </header>
  )
}

export function Card({ title, subtitle, children, className = '', action }: {
  title?: ReactNode
  subtitle?: ReactNode
  children: ReactNode
  className?: string
  action?: ReactNode
}) {
  return (
    <section className={`min-w-0 rounded-xl border border-slate-200 bg-white p-4 shadow-sm md:p-5 ${className}`}>
      {(title || action) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold text-navy-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  )
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-2 py-10 text-sm text-slate-500 justify-center">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {label}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: ApiError | Error; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
      <div className="flex items-center gap-2 font-medium">
        <AlertCircle className="h-4 w-4" aria-hidden /> Could not load data
      </div>
      <p>{error.message}</p>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex items-center gap-1 rounded-md border border-amber-300 bg-white px-2.5 py-1 text-xs font-medium hover:bg-amber-100">
          <RefreshCw className="h-3 w-3" aria-hidden /> Retry
        </button>
      )}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 py-10 text-center text-sm text-slate-500">
      <Inbox className="h-6 w-6 text-slate-400" aria-hidden />
      {children}
    </div>
  )
}

export function Badge({ children, tone = 'slate' }: { children: ReactNode; tone?: 'slate' | 'teal' | 'blue' | 'purple' | 'amber' | 'navy' }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700 ring-slate-200',
    teal: 'bg-teal-50 text-teal-800 ring-teal-200',
    blue: 'bg-blue-50 text-blue-800 ring-blue-200',
    purple: 'bg-violet-50 text-violet-800 ring-violet-200',
    amber: 'bg-amber-50 text-amber-800 ring-amber-200',
    navy: 'bg-navy-900 text-white ring-navy-900',
  }
  return <span className={`inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${tones[tone]}`}>{children}</span>
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-navy-900">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  )
}

export function Tabs<T extends string>({ tabs, value, onChange, label }: {
  tabs: { id: T; label: string; icon?: ReactNode }[]
  value: T
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="tablist" aria-label={label} className="mb-5 flex flex-wrap gap-1 rounded-lg bg-slate-100 p-1">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium ${
            value === t.id ? 'bg-white text-navy-900 shadow-sm' : 'text-slate-600 hover:text-navy-900'
          }`}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  )
}

export function SourceTag({ kind }: { kind: 'cv' | 'test' | 'train' | 'all' }) {
  const map = {
    cv: { t: '5-fold CV · training data', tone: 'teal' as const },
    test: { t: 'Test data (114 samples)', tone: 'purple' as const },
    train: { t: 'Training data', tone: 'blue' as const },
    all: { t: 'All 569 samples', tone: 'slate' as const },
  }
  return <Badge tone={map[kind].tone}>{map[kind].t}</Badge>
}
