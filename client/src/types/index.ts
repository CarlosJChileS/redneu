export interface PredictionResult {
  probabilities: number[]
  predictedDigit: number
  confidence: number
  hidden1: number[]
  hidden2: number[]
  hidden3?: number[]
  hidden4?: number[]
  hidden5?: number[]
  hidden6?: number[]
  hidden7?: number[]
  hidden8?: number[]
  hidden9?: number[]
  hidden10?: number[]
  inputSample: number[]
}

export interface DrawingCanvasProps {
  onDraw: (canvas: HTMLCanvasElement, pixels: number[]) => void
  gridSize: number
  cellSize: number
  disabled?: boolean
}

export interface PredictionPanelProps {
  prediction: PredictionResult | null
  thumbnailData: string | null
}

export interface NetworkVisualizerProps {
  prediction: PredictionResult | null
}
