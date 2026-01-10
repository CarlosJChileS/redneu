import { PredictionPanelProps } from '../types'
import '../styles/PredictionPanel.css'

const PredictionPanel: React.FC<PredictionPanelProps> = ({ prediction, thumbnailData }) => {
  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return '#28a745'
    if (confidence >= 0.7) return '#667eea'
    if (confidence >= 0.5) return '#ffc107'
    return '#dc3545'
  }

  const getConfidenceLabel = (confidence: number): string => {
    if (confidence >= 0.9) return 'MUY ALTA'
    if (confidence >= 0.7) return 'ALTA'
    if (confidence >= 0.5) return 'MEDIA'
    return 'BAJA'
  }

  return (
    <div className="prediction-panel">
      <span className="panel-label">PREDICCIÓN</span>
      
      <div className="prediction-display">
        {prediction ? (
          <>
            <div 
              className="predicted-digit"
              style={{ color: getConfidenceColor(prediction.confidence) }}
            >
              {prediction.predictedDigit}
            </div>
            <div className="confidence-section">
              <div 
                className="confidence-badge"
                style={{ background: getConfidenceColor(prediction.confidence) }}
              >
                {getConfidenceLabel(prediction.confidence)}
              </div>
              <div className="confidence-value">
                {(prediction.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </>
        ) : (
          <div className="waiting">
            <span className="waiting-icon">✏️</span>
            <span>Dibuja un número</span>
          </div>
        )}
      </div>

      {prediction && (
        <div className="probabilities">
          <span className="probs-label">PROBABILIDADES</span>
          <div className="prob-grid">
            {prediction.probabilities.map((prob, idx) => (
              <div 
                key={idx} 
                className={`prob-item ${idx === prediction.predictedDigit ? 'winner' : ''}`}
              >
                <span className="prob-digit">{idx}</span>
                <div className="prob-bar-container">
                  <div 
                    className="prob-bar"
                    style={{ 
                      width: `${prob * 100}%`,
                      background: idx === prediction.predictedDigit 
                        ? getConfidenceColor(prediction.confidence)
                        : '#667eea'
                    }}
                  />
                </div>
                <span className="prob-value">{(prob * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {thumbnailData && (
        <div className="thumbnail-section">
          <span className="thumbnail-label">ENTRADA 28×28</span>
          <img src={thumbnailData} alt="Input" className="thumbnail" />
        </div>
      )}
    </div>
  )
}

export default PredictionPanel
