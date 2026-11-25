import { ShieldAlert } from 'lucide-react'

export const DISCLAIMER =
  'Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.'

/** Persistent research-only notice. `variant="inline"` is used inside prediction results. */
export function Disclaimer({ variant = 'banner' }: { variant?: 'banner' | 'inline' }) {
  if (variant === 'inline') {
    return (
      <p data-testid="disclaimer" className="flex items-start gap-2 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-900">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        {DISCLAIMER}
      </p>
    )
  }
  return (
    <div data-testid="disclaimer" role="note" className="border-b border-indigo-200 bg-indigo-50 px-4 py-2 text-center text-xs font-medium text-indigo-900">
      <ShieldAlert className="mr-1.5 inline h-3.5 w-3.5 align-[-2px]" aria-hidden />
      {DISCLAIMER}
    </div>
  )
}
