export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

/** Probability as a %, without pretending to be exactly 0% or 100%. */
export const prob = (v: number) => (v >= 0.999 ? '> 99.9%' : v <= 0.001 ? '< 0.1%' : pct(v, 1))

export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? '—' : v.toFixed(digits)

export const sig = (v: number | null | undefined, digits = 4) =>
  v == null || Number.isNaN(v) ? '—' : Number(v.toPrecision(digits)).toString()

// Benign / malignant use blue / purple — deliberately not alarm red.
export const COLORS = {
  navy: '#1e2a4a',
  teal: '#0d9488',
  benign: '#3b82f6',
  malignant: '#7c3aed',
  slate: '#94a3b8',
}

export const MODEL_COLORS: Record<string, string> = {
  'Logistic Regression': '#1e2a4a',
  'Decision Tree': '#d97706',
  'Random Forest': '#3b82f6',
  'K-Nearest Neighbors': '#7c3aed',
  'Support Vector Machine': '#0d9488',
  'Naive Bayes': '#64748b',
}

export const LEVEL_STYLES = {
  Low: { text: 'text-teal-700', bg: 'bg-teal-50', ring: 'ring-teal-200', bar: 'bg-teal-500' },
  Medium: { text: 'text-amber-700', bg: 'bg-amber-50', ring: 'ring-amber-200', bar: 'bg-amber-500' },
  High: { text: 'text-violet-700', bg: 'bg-violet-50', ring: 'ring-violet-200', bar: 'bg-violet-600' },
}
