import { useState } from 'react'
import type { BuildStatus } from '../hooks/useBuildStatus'
import type { Color } from '../types'
import { HousePreview3D } from './HousePreview3D'

const DOOR_ROOF_COLORS: Color[] = ['red', 'blue']
const WALL_COLORS: Color[] = ['yellow', 'green']

const VOICE_COMMANDS = [
  { id: 'start', label: 'Start', hint: 'Build door layer' },
  { id: 'build wall', label: 'Build Wall', hint: 'After door' },
  { id: 'build roof', label: 'Build Roof', hint: 'After wall' },
  { id: 'retry last step', label: 'Retry Last Step', hint: 'After a failure' },
  { id: 'stop', label: 'Stop', hint: 'Safe disconnect' },
] as const

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

  const busy = submitting || status.busy

  async function postJson(url: string, body: unknown) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    return response.json()
  }

  function pipelineBody() {
    if (mode === 'sentence') return { sentence }
    return {
      sentence: `Build a house with a ${door} door, ${wall} walls, and a ${roof} roof.`,
    }
  }

  async function buildThisFromForm() {
    setSubmitting(true)
    setError(null)
    try {
      const data = await postJson('/api/build/request', pipelineBody())
      if (data.error) setError(data.error)
      if (typeof data.request_sentence === 'string') setSentence(data.request_sentence)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  async function runOneShot() {
    setSubmitting(true)
    setError(null)
    try {
      const data = await postJson('/api/build', pipelineBody())
      if (data.error) setError(data.error)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  async function runCommand(command: string) {
    setSubmitting(true)
    setError(null)
    try {
      const data = await postJson('/api/build/command', { command })
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
      <p className="panel-subtitle">
        Text and color inputs are normalized to one build phrase, then prepare the same
        staged pipeline as Build This. Step through Start → Build Wall → Build Roof.
      </p>
      <p className="note">
        Human builder: {status.features.humanBuilder ? 'enabled' : 'disabled'} · Camera
        verification: {status.features.cameraVerification ? 'enabled' : 'disabled'}
      </p>

      <HousePreview3D selection={{ door, wall, roof }} />

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
          <ColorPicker label="Door" value={door} options={DOOR_ROOF_COLORS} onChange={setDoor} />
          <ColorPicker label="Wall" value={wall} options={WALL_COLORS} onChange={setWall} />
          <ColorPicker label="Roof" value={roof} options={DOOR_ROOF_COLORS} onChange={setRoof} />
        </div>
      )}

      <div className="button-row">
        <button onClick={buildThisFromForm} disabled={busy}>
          Build This
        </button>
        <button onClick={runOneShot} disabled={busy} className="secondary">
          Run Full Build
        </button>
      </div>

      <h3 className="voice-heading">Voice command buttons</h3>
      <div className="voice-buttons">
        {VOICE_COMMANDS.map((item) => (
          <button
            key={item.id}
            onClick={() => runCommand(item.id)}
            disabled={busy && item.id !== 'stop'}
            title={item.hint}
            className={item.id === 'stop' ? 'danger' : undefined}
          >
            {item.label}
          </button>
        ))}
      </div>

      {error && <p className="error">{error}</p>}

      <dl className="status">
        <dt>State</dt>
        <dd className={`state-badge state-${status.state}`}>{status.state}</dd>
        <dt>Request</dt>
        <dd>{status.requestSentence ?? '—'}</dd>
        <dt>Completed</dt>
        <dd>{status.completedLayers.length ? status.completedLayers.join(', ') : '—'}</dd>
        <dt>Failed</dt>
        <dd>{status.failedLayer ?? '—'}</dd>
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
  options,
  onChange,
}: {
  label: string
  value: Color
  options: Color[]
  onChange: (color: Color) => void
}) {
  return (
    <label className="color-picker">
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value as Color)}>
        {options.map((color) => (
          <option key={color} value={color}>
            {color}
          </option>
        ))}
      </select>
    </label>
  )
}
