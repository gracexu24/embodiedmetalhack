import { useEffect, useRef, useState } from 'react'
import type { BuildEvent, BuildResult, BuildStateName, Transition } from '../types'

export interface BuildStatus {
  runId: string | null
  state: BuildStateName
  history: Transition[]
  result: BuildResult | null
  requestSentence: string | null
  completedLayers: string[]
  failedLayer: string | null
  busy: boolean
  connected: boolean
}

const INITIAL_STATUS: BuildStatus = {
  runId: null,
  state: 'idle',
  history: [],
  result: null,
  requestSentence: null,
  completedLayers: [],
  failedLayer: null,
  busy: false,
  connected: false,
}

/** Connects to /api/build/ws and keeps the current build status up to date. */
export function useBuildStatus(): BuildStatus {
  const [status, setStatus] = useState<BuildStatus>(INITIAL_STATUS)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${window.location.host}/api/build/ws`)
    socketRef.current = socket

    socket.onopen = () => setStatus((prev) => ({ ...prev, connected: true }))
    socket.onclose = () => setStatus((prev) => ({ ...prev, connected: false }))
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as BuildEvent
      setStatus((prev) => {
        switch (data.type) {
          case 'status':
            return {
              ...prev,
              runId: data.run_id,
              state: data.state,
              history: data.history,
              result: data.result,
              requestSentence: data.request_sentence ?? null,
              completedLayers: data.completed_layers ?? [],
              failedLayer: data.failed_layer ?? null,
              busy: Boolean(data.busy),
            }
          case 'transition':
            return {
              ...prev,
              runId: data.run_id,
              state: data.to,
              history: [...prev.history, data],
            }
          case 'result':
            return { ...prev, result: data.result }
          default:
            return prev
        }
      })
    }

    return () => socket.close()
  }, [])

  return status
}
