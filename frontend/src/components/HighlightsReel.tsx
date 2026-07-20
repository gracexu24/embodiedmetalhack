import type { BuildStatus } from '../hooks/useBuildStatus'

export function HighlightsReel({ status }: { status: BuildStatus }) {
  const highlights = status.highlights

  return (
    <section className="panel">
      <h2>Highlights Reel</h2>
      <p className="panel-subtitle">
        Key moments stream in live as each layer starts and is placed.
      </p>
      {highlights.length === 0 ? (
        <p className="panel-subtitle">Highlights appear here as the build runs.</p>
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
