import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  error: string | null
  loading: boolean
}

/**
 * Runs an async function and tracks its state.
 *
 * The request counter exists because a user typing quickly fires several
 * overlapping requests, and they do not necessarily resolve in order. Without
 * it, a slow response to an old query can land after a fast response to the
 * current one and overwrite the right answer with a stale one.
 */
export function useAsync<T>(): AsyncState<T> & {
  run: (fn: () => Promise<T>) => Promise<void>
  reset: () => void
} {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: false })
  const latest = useRef(0)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      // Setting state after unmount is a React warning and a leak.
      mounted.current = false
    }
  }, [])

  const run = useCallback(async (fn: () => Promise<T>) => {
    const id = ++latest.current
    setState((s) => ({ ...s, loading: true, error: null }))
    try {
      const data = await fn()
      if (id === latest.current && mounted.current) {
        setState({ data, error: null, loading: false })
      }
    } catch (err) {
      if (id === latest.current && mounted.current) {
        setState({
          data: null,
          error: err instanceof Error ? err.message : String(err),
          loading: false,
        })
      }
    }
  }, [])

  const reset = useCallback(() => {
    latest.current++
    setState({ data: null, error: null, loading: false })
  }, [])

  return { ...state, run, reset }
}
