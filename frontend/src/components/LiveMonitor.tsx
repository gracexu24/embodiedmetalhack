import { useEffect, useState } from 'react'

export function LiveMonitor() {
  const [viewerUrl, setViewerUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/rerun/viewer_url')
      .then((response) => response.json())
      .then((data) => setViewerUrl(data.viewer_url))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  return (
    <section className="panel panel-wide">
      <h2>Live Build Monitor</h2>
      <p className="panel-subtitle">
        Cam0, cam1, arm joints, and the state/verification log — streamed live from Rerun.
      </p>
      {error && <p className="error">Could not reach the Rerun viewer: {error}</p>}
      {viewerUrl ? (
        <iframe className="rerun-frame" src={viewerUrl} title="Rerun viewer" />
      ) : (
        !error && <p>Connecting to Rerun viewer…</p>
      )}
    </section>
  )
}
