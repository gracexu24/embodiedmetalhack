import { useState } from 'react'
import type { BuildStatus } from '../hooks/useBuildStatus'
import type { Color } from '../types'

const COLORS: Color[] = ['red', 'yellow', 'blue']
const IDLE_STATES = new Set(['idle', 'completed', 'failed'])

export function BuildControl({ status }: { status: BuildStatus }) {
  const [mode, setMode] = useState<'sentence' | 'structured'>('sentence')
  const [sentence, setSentence] = useState(
    'Build a house with a red door, yellow walls, and a blue roof.',
  )
  const [door, setDoor] = useState<Color>('red')
  const [wall, setWall] = useState<Color>('yellow')
  const [roof, setRoof] = useState<Color>('blue')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const busy = submitting || !IDLE_STATES.has(status.state)

  async function startBuild() {
    setSubmitting(true)
    setError(null)
    try {
      const body = mode === 'sentence' ? { sentence } : { door, wall, roof }
      const response = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await response.json()
      if (data.error) setError(data.error)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="panel">
      <h2>Build Control</h2>

      <div className="mode-toggle">
        <label>
          <input type="radio" checked={mode === 'sentence'} onChange={() => setMode('sentence')} />
          Sentence
        </label>
        <label>
          <input
            type="radio"
            checked={mode === 'structured'}
            onChange={() => setMode('structured')}
          />
          Colors
        </label>
      </div>

      {mode === 'sentence' ? (
        <textarea
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
          rows={3}
        />
      ) : (
        <div className="color-pickers">
          <ColorPicker label="Door" value={door} onChange={setDoor} />
          <ColorPicker label="Wall" value={wall} onChange={setWall} />
          <ColorPicker label="Roof" value={roof} onChange={setRoof} />
        </div>
      )}

      <button onClick={startBuild} disabled={busy}>
        {busy ? 'Building…' : 'Start Build'}
      </button>

      {error && <p className="error">{error}</p>}

      <dl className="status">
        <dt>State</dt>
        <dd className={`state-badge state-${status.state}`}>{status.state}</dd>
        <dt>Run</dt>
        <dd>{status.runId ?? '—'}</dd>
        <dt>Connection</dt>
        <dd>{status.connected ? 'live' : 'disconnected'}</dd>
      </dl>

      {status.result && (
        <p className={status.result.success ? 'result-success' : 'result-failure'}>
          {status.result.message}
        </p>
      )}

      {status.history.length > 0 && (
        <ol className="history">
          {status.history.map((transition, index) => (
            <li key={index}>
              {transition.from} → {transition.to}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function ColorPicker({
  label,
  value,
  onChange,
}: {
  label: string
  value: Color
  onChange: (color: Color) => void
}) {
  return (
    <label className="color-picker">
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value as Color)}>
        {COLORS.map((color) => (
          <option key={color} value={color}>
            {color}
          </option>
        ))}
      </select>
    </label>
  )
}
