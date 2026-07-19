import { useEffect, useRef, useState } from 'react'
import type { BuildEvent, BuildResult, BuildStateName, Transition } from '../types'

export interface BuildStatus {
  runId: string | null
  state: BuildStateName
  history: Transition[]
  result: BuildResult | null
  connected: boolean
}

const INITIAL_STATUS: BuildStatus = {
  runId: null,
  state: 'idle',
  history: [],
  result: null,
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
