/**
 * Typed client for the lens API.
 *
 * The response types mirror the Pydantic models on the server. Keeping them in
 * sync by hand is a real cost; the alternative is generating them from the
 * OpenAPI schema FastAPI already publishes at /openapi.json, which is the right
 * move once the surface stops changing.
 */

export interface ChunkResult {
  chunk_id: string
  doc_id: string
  text: string
  start: number
  end: number
  score: number
  source: string
  components: Record<string, number>
  metadata: Record<string, unknown>
}

export interface SearchResponse {
  query: string
  retriever: string
  results: ChunkResult[]
  latency_ms: number
  diagnostics: Record<string, unknown>
}

export interface AskResponse {
  question: string
  answer: string
  citations: ChunkResult[]
  model: string
  latency_ms: number
  usage: Record<string, number>
  prompt: string | null
}

export interface MetricRow {
  name: string
  means: Record<string, number>
  latency: Record<string, number>
  num_queries: number
  skipped: number
}

export interface QueryOutcome {
  query_id: string
  query: string
  retrieved: string[]
  relevant: string[]
  latency_ms: number
  scores: Record<string, number>
}

export interface EvalResponse {
  report: MetricRow
  outcomes: QueryOutcome[]
  failures: QueryOutcome[]
}

export interface SweepResponse {
  rows: MetricRow[]
  table: string
  primary: string
  best: string
}

export interface StatusResponse {
  documents: number
  chunks: number
  queries: number
  chunk_size: number
  chunk_overlap: number
  embedding_dim: number
  vocabulary_size: number
  mean_chunk_chars: number
  dense_index: string
}

export type Retriever = 'dense' | 'lexical' | 'hybrid'

/** Thrown for a non-2xx response, carrying the server's message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function get<T>(path: string, params: Record<string, string | number | boolean> = {}): Promise<T> {
  const query = new URLSearchParams(
    Object.entries(params).map(([k, v]) => [k, String(v)]),
  )
  const url = `/api${path}${query.toString() ? `?${query}` : ''}`
  const response = await fetch(url)

  if (!response.ok) {
    // Surface the server's own explanation rather than a bare status code;
    // FastAPI puts validation failures in `detail`.
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* response had no JSON body; the status text will do */
    }
    throw new ApiError(detail, response.status)
  }
  return (await response.json()) as T
}

export const api = {
  status: () => get<StatusResponse>('/status'),
  search: (q: string, retriever: Retriever, k: number) =>
    get<SearchResponse>('/search', { q, retriever, k }),
  ask: (q: string, retriever: Retriever, k: number, includePrompt = false) =>
    get<AskResponse>('/ask', { q, retriever, k, include_prompt: includePrompt }),
  evaluate: (retriever: Retriever, k: number) => get<EvalResponse>('/eval', { retriever, k }),
  sweep: (k: number, primary: string) => get<SweepResponse>('/sweep', { k, primary }),
}
