import type { ChunkResult, MetricRow, QueryOutcome } from './api'

export function Spinner({ label = 'working…' }: { label?: string }) {
  return (
    <div className="spinner" role="status" aria-live="polite">
      {label}
    </div>
  )
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="error" role="alert">
      {message}
    </div>
  )
}

/** Renders a score bar whose width is relative to the best score shown. */
function ScoreBar({ score, max }: { score: number; max: number }) {
  const width = max > 0 ? Math.max(2, (score / max) * 100) : 0
  return (
    <div className="scorebar" title={score.toFixed(6)}>
      <div className="scorebar-fill" style={{ width: `${width}%` }} />
    </div>
  )
}

export function ResultCard({
  result,
  rank,
  maxScore,
}: {
  result: ChunkResult
  rank: number
  maxScore: number
}) {
  const parts = Object.entries(result.components)
  return (
    <li className="result">
      <div className="result-head">
        <span className="rank">{rank}</span>
        <span className="docid">{result.doc_id}</span>
        {/* Character offsets make a citation checkable rather than decorative. */}
        <span className="offsets">
          chars {result.start}–{result.end}
        </span>
        <span className="score">{result.score.toFixed(4)}</span>
      </div>
      <ScoreBar score={result.score} max={maxScore} />
      {parts.length > 0 && (
        <div className="components">
          {parts.map(([name, value]) => (
            <span key={name} className={`chip chip-${name}`}>
              {name} {value.toFixed(4)}
            </span>
          ))}
        </div>
      )}
      <p className="snippet">{result.text}</p>
    </li>
  )
}

const DISPLAY_METRICS = ['recall@5', 'recall@10', 'mrr', 'ndcg@10', 'map'] as const

export function MetricsTable({ rows, best }: { rows: MetricRow[]; best?: string }) {
  if (rows.length === 0) return null
  // Colour each column relative to its own best value, so a strong cell is
  // visible without having to read every number.
  const columnMax = new Map<string, number>()
  for (const metric of DISPLAY_METRICS) {
    columnMax.set(metric, Math.max(...rows.map((r) => r.means[metric] ?? 0)))
  }

  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>configuration</th>
          {DISPLAY_METRICS.map((m) => (
            <th key={m}>{m}</th>
          ))}
          <th>p95 ms</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.name} className={row.name === best ? 'best' : undefined}>
            <td className="name">
              {row.name}
              {row.name === best && <span className="badge">best</span>}
            </td>
            {DISPLAY_METRICS.map((metric) => {
              const value = row.means[metric] ?? 0
              const top = columnMax.get(metric) ?? 0
              return (
                <td key={metric} className={value >= top && top > 0 ? 'top' : undefined}>
                  {value.toFixed(4)}
                </td>
              )
            })}
            <td>{(row.latency['p95'] ?? 0).toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function OutcomeTable({ outcomes, k }: { outcomes: QueryOutcome[]; k: number }) {
  const metric = `recall@${k}`
  // Worst first: the failures are the actionable part of an evaluation.
  const sorted = [...outcomes].sort(
    (a, b) => (a.scores[metric] ?? 0) - (b.scores[metric] ?? 0),
  )
  return (
    <table className="outcomes">
      <thead>
        <tr>
          <th>query</th>
          <th>{metric}</th>
          <th>mrr</th>
          <th>wanted</th>
          <th>got</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((o) => {
          const recall = o.scores[metric] ?? 0
          const relevant = new Set(o.relevant)
          return (
            <tr key={o.query_id} className={recall === 0 ? 'failed' : undefined}>
              <td className="query">{o.query}</td>
              <td>{o.relevant.length ? recall.toFixed(2) : '—'}</td>
              <td>{o.relevant.length ? (o.scores['mrr'] ?? 0).toFixed(2) : '—'}</td>
              <td className="ids">{o.relevant.join(', ') || '—'}</td>
              <td className="ids">
                {o.retrieved.slice(0, 4).map((id, i) => (
                  <span key={`${id}-${i}`} className={relevant.has(id) ? 'hit' : 'miss'}>
                    {id}
                  </span>
                ))}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
