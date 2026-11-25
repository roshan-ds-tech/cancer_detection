import { sig } from './format'
import type { FeatureSpec } from './types'

type FieldErrors = Record<string, string>

/** Client-side checks mirroring the API schema: required, finite, non-negative. Out-of-range is a warning only. */
export function validateFeatureInputs(values: Record<string, string>, specs: FeatureSpec[]) {
  const errors: FieldErrors = {}
  const warnings: FieldErrors = {}
  const parsed: Record<string, number> = {}
  for (const s of specs) {
    const raw = (values[s.name] ?? '').trim()
    if (raw === '') { errors[s.name] = 'Required — missing values are not imputed'; continue }
    const v = Number(raw)
    if (!Number.isFinite(v)) { errors[s.name] = 'Must be a finite number'; continue }
    if (v < 0) { errors[s.name] = 'Must be ≥ 0'; continue }
    if (v < s.train_min || v > s.train_max) warnings[s.name] = `Outside training range ${sig(s.train_min)}–${sig(s.train_max)}`
    parsed[s.name] = v
  }
  return { errors, warnings, parsed }
}

