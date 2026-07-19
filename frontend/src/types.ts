export type BuildStateName =
  | 'idle'
  | 'connecting'
  | 'homing'
  | 'picking'
  | 'placing'
  | 'verifying'
  | 'completed'
  | 'failed'

export type Color = 'red' | 'yellow' | 'blue'

export interface Transition {
  from: BuildStateName
  to: BuildStateName
  step: number
}

export interface BuildResult {
  success: boolean
  completed_layers: string[]
  failed_layer: string | null
  message: string
}

export interface StatusEvent {
  type: 'status'
  run_id: string | null
  state: BuildStateName
  history: Transition[]
  result: BuildResult | null
}

export interface TransitionEvent extends Transition {
  type: 'transition'
  run_id: string | null
}

export interface ResultEvent {
  type: 'result'
  run_id: string
  result: BuildResult
}

export type BuildEvent = StatusEvent | TransitionEvent | ResultEvent

export interface Highlight {
  step: number
  kind: 'verification' | 'instruction'
  label: string
  thumbnail_base64: string | null
}

export interface DetectedColors {
  door: Color
  wall: Color
  roof: Color
}

export interface ScanResponse {
  status: 'captured' | 'camera_unavailable'
  scan_id?: string
  image_url?: string
  detected: DetectedColors | null
  note?: string
  error?: string
}
