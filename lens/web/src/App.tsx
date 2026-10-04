import { useEffect, useMemo, useState } from 'react'
import {
  api,
  type AskResponse,
  type EvalResponse,
  type Retriever,
  type SearchResponse,
  type StatusResponse,
  type SweepResponse,
} from './api'
import { ErrorBox, MetricsTable, OutcomeTable, ResultCard, Spinner } from './components'
import { useAsync } from './useAsync'

type Tab = 'search' | 'ask' | 'evaluate'

const RETRIEVERS: Retriever[] = ['hybrid', 'dense', 'lexical']

const EXAMPLES = [
  'reciprocal rank fusion',
  'efSearch parameter',
  'ANN-4417',
  'shrinking memory by storing fewer bits per dimension',
  'why the slowest request matters more than the typical one',
]

export function App() {
  const [tab, setTab] = useState<Tab>('search')
  const [query, setQuery] = useState('reciprocal rank fusion')
  const [retriever, setRetriever] = useState<Retriever>('hybrid')
  const [k, setK] = useState(5)

  const status = useAsync<StatusResponse>()
  const search = useAsync<SearchResponse>()
  const ask = useAsync<AskResponse>()
  const evaluation = useAsync<EvalResponse>()
  const sweep = useAsync<SweepResponse>()

  useEffect(() => {
    void status.run(() => api.status())
    // Run once on mount; `status.run` is stable via useCallback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function submit() {
    if (!query.trim()) return
    if (tab === 'search') void search.run(() => api.search(query, retriever, k))
    else void ask.run(() => api.ask(query, retriever, k, true))
  }

  const maxScore = useMemo(
    () => Math.max(0, ...(search.data?.results.map((r) => r.score) ?? [])),
    [search.data],
  )

  return (
    <div className="app">
      <header>
        <h1>
          lens <span className="sub">hybrid retrieval &amp; RAG evaluation</span>
        </h1>
        {status.data && (
          <div className="stats">
            <span>{status.data.documents} docs</span>
            <span>{status.data.chunks} chunks</span>
            <span>{status.data.queries} labelled queries</span>
            <span>{status.data.embedding_dim}d embeddings</span>
            <span>{status.data.dense_index} index</span>
            <span>{status.data.vocabulary_size} terms</span>
          </div>
        )}
      </header>

      <nav className="tabs">
        {(['search', 'ask', 'evaluate'] as const).map((t) => (
          <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </nav>

      {tab !== 'evaluate' && (
        <section className="controls">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
            placeholder={tab === 'search' ? 'search the corpus…' : 'ask a question…'}
            aria-label="query"
          />
          <select
            value={retriever}
            onChange={(e) => setRetriever(e.target.value as Retriever)}
            aria-label="retriever"
          >
            {RETRIEVERS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <label className="k">
            k
            <input
              type="number"
              min={1}
              max={50}
              value={k}
              onChange={(e) => setK(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
            />
          </label>
          <button className="primary" onClick={submit}>
            {tab === 'search' ? 'search' : 'ask'}
          </button>
        </section>
      )}

      {tab !== 'evaluate' && (
        <div className="examples">
          {EXAMPLES.map((e) => (
            <button key={e} className="example" onClick={() => setQuery(e)}>
              {e}
            </button>
          ))}
        </div>
      )}

      {tab === 'search' && (
        <section>
          {search.loading && <Spinner label="searching…" />}
          {search.error && <ErrorBox message={search.error} />}
          {search.data && (
            <>
              <div className="meta">
                {search.data.results.length} results in {search.data.latency_ms.toFixed(2)} ms
                {' · '}
                {/* Overlap near 1.0 means the two retrievers agree and hybrid
                    is buying nothing; near 0 means fusion is doing real work. */}
                {typeof search.data.diagnostics['overlap'] === 'number' && (
                  <span title="Jaccard overlap between the dense and lexical result sets">
                    retriever overlap{' '}
                    {(search.data.diagnostics['overlap'] as number).toFixed(2)}
                  </span>
                )}
              </div>
              <ol className="results">
                {search.data.results.map((r, i) => (
                  <ResultCard key={r.chunk_id} result={r} rank={i + 1} maxScore={maxScore} />
                ))}
              </ol>
            </>
          )}
        </section>
      )}

      {tab === 'ask' && (
        <section>
          {ask.loading && <Spinner label="thinking…" />}
          {ask.error && <ErrorBox message={ask.error} />}
          {ask.data && (
            <>
              <blockquote className="answer">{ask.data.answer}</blockquote>
              <div className="meta">
                {ask.data.model} · {ask.data.latency_ms.toFixed(1)} ms ·{' '}
                {ask.data.citations.length} citation(s)
              </div>
              <h3>Cited passages</h3>
              <ol className="results">
                {ask.data.citations.map((c, i) => (
                  <ResultCard key={c.chunk_id} result={c} rank={i + 1} maxScore={1} />
                ))}
              </ol>
              {ask.data.prompt && (
                <details className="prompt">
                  <summary>the exact prompt the model received</summary>
                  <pre>{ask.data.prompt}</pre>
                </details>
              )}
            </>
          )}
        </section>
      )}

      {tab === 'evaluate' && (
        <section>
          <div className="controls">
            <button
              className="primary"
              onClick={() => void sweep.run(() => api.sweep(10, 'ndcg@10'))}
            >
              run sweep
            </button>
            <button onClick={() => void evaluation.run(() => api.evaluate(retriever, 10))}>
              evaluate {retriever}
            </button>
            <select
              value={retriever}
              onChange={(e) => setRetriever(e.target.value as Retriever)}
              aria-label="retriever"
            >
              {RETRIEVERS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          {sweep.loading && <Spinner label="evaluating every configuration…" />}
          {sweep.error && <ErrorBox message={sweep.error} />}
          {sweep.data && (
            <>
              <h3>Configurations ranked by {sweep.data.primary}</h3>
              <MetricsTable rows={sweep.data.rows} best={sweep.data.best} />
            </>
          )}

          {evaluation.loading && <Spinner label="scoring queries…" />}
          {evaluation.error && <ErrorBox message={evaluation.error} />}
          {evaluation.data && (
            <>
              <h3>
                {evaluation.data.report.name} — per query
                {evaluation.data.failures.length > 0 && (
                  <span className="fail-count">
                    {evaluation.data.failures.length} found nothing relevant
                  </span>
                )}
              </h3>
              {/* The per-query view is the point: a mean of 0.8 is consistent
                  with "all mediocre" and with "mostly perfect, a few broken",
                  and those need opposite fixes. */}
              <OutcomeTable outcomes={evaluation.data.outcomes} k={10} />
            </>
          )}
        </section>
      )}
    </div>
  )
}
