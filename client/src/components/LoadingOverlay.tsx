import '../styles/LoadingOverlay.css'

interface LoadingOverlayProps {
  progress: number
  status: string
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ progress, status }) => {
  return (
    <div className="loading-overlay">
      <div className="loading-content">
        <div className="neural-animation">
          <div className="neuron-col">
            <div className="neuron"></div>
            <div className="neuron"></div>
            <div className="neuron"></div>
          </div>
          <div className="connections">
            <div className="conn"></div>
            <div className="conn"></div>
          </div>
          <div className="neuron-col">
            <div className="neuron big"></div>
            <div className="neuron big"></div>
            <div className="neuron big"></div>
            <div className="neuron big"></div>
          </div>
          <div className="connections">
            <div className="conn"></div>
            <div className="conn"></div>
          </div>
          <div className="neuron-col">
            <div className="neuron"></div>
            <div className="neuron"></div>
            <div className="neuron"></div>
          </div>
        </div>
        
        <div className="loading-text">
          <span className="main-text">Red Neuronal</span>
          <span className="status-text">{status}</span>
        </div>
        
        <div className="progress-container">
          <div className="progress-bar-bg">
            <div 
              className="progress-bar-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <span className="progress-percent">{Math.round(progress)}%</span>
        </div>
      </div>
    </div>
  )
}

export default LoadingOverlay
