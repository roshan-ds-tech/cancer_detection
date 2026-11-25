import { describe, expect, it } from 'vitest'
import { validateFeatureInputs } from '../lib/validation'
import type { FeatureSpec } from '../lib/types'

const spec = (name: string): FeatureSpec => ({
  name, sklearn_name: name, measurement: 'radius', statistic: 'mean', description: '', unit: '',
  train_min: 1, train_max: 10, train_median: 5, input_min: 0,
})

describe('validateFeatureInputs', () => {
  const specs = [spec('a'), spec('b'), spec('c'), spec('d'), spec('e')]
  it('flags missing, non-numeric, negative and non-finite values', () => {
    const r = validateFeatureInputs({ a: '', b: 'abc', c: '-1', d: 'Infinity', e: '5' }, specs)
    expect(Object.keys(r.errors).sort()).toEqual(['a', 'b', 'c', 'd'])
    expect(r.parsed).toEqual({ e: 5 })
  })
  it('warns (but does not block) on out-of-range values', () => {
    const r = validateFeatureInputs({ a: '50', b: '2', c: '2', d: '2', e: '2' }, specs)
    expect(r.errors).toEqual({})
    expect(r.warnings.a).toMatch(/Outside training range/)
    expect(r.parsed.a).toBe(50)
  })
})
