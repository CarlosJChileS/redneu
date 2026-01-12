import { useState, useEffect, useRef, useCallback } from 'react'
import DrawingCanvas from './components/DrawingCanvas'
import PredictionPanel from './components/PredictionPanel'
import NetworkVisualizer from './components/NetworkVisualizer'
import LoadingOverlay from './components/LoadingOverlay'
import { NeuralNetwork } from './services/NeuralNetwork'
import { PredictionResult } from './types'
import './styles/App.css'

function App() {
  const networkRef = useRef<NeuralNetwork | null>(null)
  const drawingCanvasRef = useRef<{ clear: () => void }>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isReady, setIsReady] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [thumbnailData, setThumbnailData] = useState<string | null>(null)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingStatus, setLoadingStatus] = useState('Iniciando...')
  const [isRetraining, setIsRetraining] = useState(false)

  useEffect(() => {
    let mounted = true
    
    const initNetwork = async () => {
      try {
        // Usar singleton para evitar múltiples instancias
        const nn = NeuralNetwork.getInstance()
        networkRef.current = nn
        
        // Si ya está listo, no reinicializar
        if (nn.isReady) {
          console.log('✅ Red neuronal ya estaba lista')
          if (mounted) {
            setIsReady(true)
            setIsLoading(false)
          }
          return
        }
        
        // Callback para actualizar progreso
        nn.onProgress = (progress: number, status: string) => {
          if (mounted) {
            setLoadingProgress(progress)
            setLoadingStatus(status)
          }
        }
        
        const success = await nn.initialize()
        
        if (mounted && success) {
          setIsReady(true)
        }
      } catch (error) {
        console.error('Error:', error)
      } finally {
        if (mounted) {
          setIsLoading(false)
        }
      }
    }
    initNetwork()
    
    return () => {
      mounted = false
    }
  }, [])

  const handleDraw = useCallback(async (canvas: HTMLCanvasElement, pixels: number[]) => {
    const network = networkRef.current
    if (!network || !isReady) return

    const thumbCanvas = document.createElement('canvas')
    thumbCanvas.width = 28
    thumbCanvas.height = 28
    const ctx = thumbCanvas.getContext('2d')
    if (ctx) {
      ctx.fillStyle = 'white'
      ctx.fillRect(0, 0, 28, 28)
      ctx.drawImage(canvas, 0, 0, 28, 28)
      setThumbnailData(thumbCanvas.toDataURL())
    }

    try {
      const result = await network.predict(pixels)
      setPrediction(result)
    } catch (error) {
      console.error('Error:', error)
    }
  }, [isReady])

  const handleClear = useCallback(() => {
    drawingCanvasRef.current?.clear()
    setPrediction(null)
    setThumbnailData(null)
  }, [])

  const handleRetrain = useCallback(async () => {
    console.log('🔄 Botón RE-ENTRENAR presionado')
    
    const nn = NeuralNetwork.getInstance()
    networkRef.current = nn
    
    if (isRetraining) {
      console.log('⚠️ Ya está re-entrenando')
      return
    }
    
    setIsRetraining(true)
    setIsLoading(true)
    setLoadingProgress(0)
    setLoadingStatus('Preparando re-entrenamiento...')
    setPrediction(null)
    setThumbnailData(null)
    setIsReady(false)
    
    try {
      // Reconectar el callback de progreso
      nn.onProgress = (progress: number, status: string) => {
        console.log(`📊 Progreso: ${progress}% - ${status}`)
        setLoadingProgress(progress)
        setLoadingStatus(status)
      }
      
      console.log('🚀 Iniciando retrainModel()...')
      const success = await nn.retrainModel()
      console.log(`✅ retrainModel() completado: ${success}`)
      
      if (success) {
        setIsReady(true)
      }
    } catch (error) {
      console.error('❌ Error re-entrenando:', error)
    } finally {
      setIsLoading(false)
      setIsRetraining(false)
    }
  }, [isRetraining])

  return (
    <div className="app">
      {isLoading && <LoadingOverlay progress={loadingProgress} status={loadingStatus} />}
      
      <div className="container">
        <div className="left-section">
          <PredictionPanel 
            prediction={prediction} 
            thumbnailData={thumbnailData}
          />
          
          <div className="drawing-section">
            <DrawingCanvas 
              ref={drawingCanvasRef}
              onDraw={handleDraw}
              gridSize={28}
              cellSize={10}
              disabled={!isReady}
            />
            <div className="button-row">
              <button className="btn-clear" onClick={handleClear} disabled={!isReady}>
                🔄 LIMPIAR
              </button>
              <button 
                className="btn-retrain" 
                onClick={handleRetrain} 
                disabled={isRetraining || isLoading}
                title="Re-entrena el modelo desde cero"
              >
                {isRetraining ? '⏳ ENTRENANDO...' : '🧠 RE-ENTRENAR'}
              </button>
            </div>
          </div>
        </div>
        
        <NetworkVisualizer prediction={prediction} />
      </div>
    </div>
  )
}

export default App
