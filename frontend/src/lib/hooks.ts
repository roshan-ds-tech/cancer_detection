import { useCallback, useEffect, useState } from 'react'
import { ApiError } from './api'

export interface AsyncState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/** Fetch on mount (and when `deps` change); exposes loading / error / data and a reload callback. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let alive = true
    // Reset to a loading state when the request is re-issued (reload or changed deps).
    Promise.resolve().then(() => {
      if (alive) {
        setLoading(true)
        setError(null)
      }
    })
    fn()
      .then((d) => alive && setData(d))
      .catch((e: unknown) =>
        alive && setError(e instanceof ApiError ? e : new ApiError(0, String(e))),
      )
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}
