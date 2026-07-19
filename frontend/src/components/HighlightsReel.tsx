import { useEffect, useState } from 'react'
import type { BuildStatus } from '../hooks/useBuildStatus'
import type { Highlight } from '../types'

const TERMINAL_STATES = new Set(['completed', 'failed'])

export function HighlightsReel({ status }: { status: BuildStatus }) {
  const [highlights, setHighlights] = useState<Highlight[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!status.runId || !TERMINAL_STATES.has(status.state)) return

    let cancelled = false
    fetch(`/api/highlights/${status.runId}`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<Highlight[]>
      })
      .then((data) => {
        if (!cancelled) setHighlights(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })

    return () => {
      cancelled = true
    }
  }, [status.runId, status.state])

  return (
    <section className="panel panel-wide">
      <h2>Highlights Reel</h2>
      {error && <p className="error">Could not load highlights: {error}</p>}
      {highlights.length === 0 ? (
        <p className="panel-subtitle">Highlights appear here once a build finishes.</p>
      ) : (
        <div className="highlights-strip">
          {highlights.map((highlight, index) => (
            <article key={index} className={`highlight-card highlight-${highlight.kind}`}>
              {highlight.thumbnail_base64 && (
                <img
                  src={`data:image/jpeg;base64,${highlight.thumbnail_base64}`}
                  alt={highlight.label}
                />
              )}
              <p className="highlight-kind">{highlight.kind}</p>
              <p className="highlight-label">{highlight.label}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
