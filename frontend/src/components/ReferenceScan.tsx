import { useState } from 'react'
import type { ScanResponse } from '../types'

// Order the camera sees scanning top-to-bottom on the reference house.
const LAYERS = ['roof', 'wall', 'door'] as const

export function ReferenceScan() {
  const [scan, setScan] = useState<ScanResponse | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleBuildThis() {
    setLoading(true)
    try {
      const response = await fetch('/api/cam2/scan', { method: 'POST' })
      const data = (await response.json()) as ScanResponse
      setScan(data)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h2>Reference Scan (Camera 3)</h2>
      <p className="panel-subtitle">
        Point this camera at the human-built model house, then press Build This (same as the
        voice command) to parse door / wall / roof colors.
      </p>
      <img className="camera-preview" src="/api/cam2/preview" alt="Camera 3 live preview" />
      <button onClick={handleBuildThis} disabled={loading}>
        {loading ? 'Scanning…' : 'Build This'}
      </button>

      {scan?.status === 'camera_unavailable' && (
        <p className="warning">Camera 3 is not available: {scan.error}</p>
      )}

      {scan?.status === 'captured' && (
        <div className="scan-result">
          {scan.image_url && (
            <img className="scan-thumbnail" src={scan.image_url} alt="Captured scan" />
          )}
          <div className="swatches">
            {LAYERS.map((layer) => (
              <div key={layer} className="swatch">
                <span className="swatch-label">{layer}</span>
                <span className={scan.detected ? 'swatch-value' : 'swatch-value pending'}>
                  {scan.detected ? scan.detected[layer] : 'pending'}
                </span>
              </div>
            ))}
          </div>
          {scan.request_sentence && <p className="note">{scan.request_sentence}</p>}
          {scan.error && <p className="error">{scan.error}</p>}
          {scan.note && <p className="note">{scan.note}</p>}
        </div>
      )}
    </section>
  )
}
