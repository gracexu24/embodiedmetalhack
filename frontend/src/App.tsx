import { useBuildStatus } from './hooks/useBuildStatus'
import { BuildControl } from './components/BuildControl'
import { LiveMonitor } from './components/LiveMonitor'
import { HighlightsReel } from './components/HighlightsReel'

function App() {
  const status = useBuildStatus()

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>House Builder Dashboard</h1>
      </header>
      <div className="dashboard-grid">
        <BuildControl status={status} />
        <HighlightsReel status={status} />
        <LiveMonitor />
      </div>
    </div>
  )
}

export default App
