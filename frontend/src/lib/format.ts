export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? '—' : v.toFixed(digits)

export const sig = (v: number | null | undefined, digits = 4) =>
  v == null || Number.isNaN(v) ? '—' : Number(v.toPrecision(digits)).toString()

export const bytes = (b: number | null | undefined) => {
  if (b == null) return '—'
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 ** 2).toFixed(2)} MB`
}

export const prettyFeature = (f: string) =>
  f
    .replace(/_se$/, ' (SE)')
    .replace(/^mean_/, 'mean ')
    .replace(/^worst_/, 'worst ')
    .replace(/_/g, ' ')

export const shortParam = (k: string) => k.replace(/^clf__(estimator__)?/, '')

// Colour roles. Benign / malignant use blue / purple — deliberately not alarm red.
export const COLORS = {
  navy: '#1e2a4a',
  teal: '#0d9488',
  blue: '#3b82f6',
  purple: '#7c3aed',
  slate: '#94a3b8',
  benign: '#3b82f6',
  malignant: '#7c3aed',
}

export const MODEL_COLORS: Record<string, string> = {
  logistic_regression: '#1e2a4a',
  svm: '#0d9488',
  random_forest: '#3b82f6',
  decision_tree: '#d97706',
  knn: '#7c3aed',
  gaussian_nb: '#64748b',
  gradient_boosting: '#db2777',
}
