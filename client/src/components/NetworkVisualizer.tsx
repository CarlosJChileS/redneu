import { useRef, useEffect, useCallback, useState } from 'react'
import { NetworkVisualizerProps } from '../types'
import '../styles/NetworkVisualizer.css'

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ prediction }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>(0)
  const [animPhase, setAnimPhase] = useState(0)

  // Capas con menos neuronas para mejor visualización
  const layers = [
    { name: 'Entrada (784px)', units: 8, color: '#FF6B6B', desc: 'Imagen' },
    { name: 'Oculta 1 (128)', units: 6, color: '#FFB84C', desc: '128 neuronas' },
    { name: 'Oculta 2 (64)', units: 5, color: '#00C9B1', desc: '64 neuronas' },
    { name: 'Salida (10)', units: 10, color: '#667eea', desc: 'Dígitos 0-9' }
  ]

  // Animación continua para mostrar flujo de datos
  useEffect(() => {
    let frame = 0
    const animate = () => {
      frame++
      if (frame % 3 === 0) {
        setAnimPhase(prev => (prev + 1) % 100)
      }
      animationRef.current = requestAnimationFrame(animate)
    }
    animationRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationRef.current)
  }, [])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const w = rect.width
    const h = rect.height
    ctx.clearRect(0, 0, w, h)

    const padX = 60
    const padY = 50
    const spacing = (w - 2 * padX) / (layers.length - 1)

    // Calcular posiciones de neuronas
    const getNeuronY = (idx: number, total: number): number => {
      const availH = h - padY * 2 - 30
      const neuronSpacing = Math.min(35, availH / (total + 1))
      const totalH = (total - 1) * neuronSpacing
      return padY + 15 + (availH - totalH) / 2 + idx * neuronSpacing
    }

    // Dibujar conexiones con efecto de flujo
    for (let i = 1; i < layers.length; i++) {
      const x1 = padX + (i - 1) * spacing
      const x2 = padX + i * spacing
      const prevLayer = layers[i - 1]
      const currLayer = layers[i]

      for (let j = 0; j < currLayer.units; j++) {
        for (let k = 0; k < prevLayer.units; k++) {
          const y1 = getNeuronY(k, prevLayer.units)
          const y2 = getNeuronY(j, currLayer.units)

          // Calcular intensidad de conexión basada en predicción
          let connectionStrength = 0.15
          if (prediction) {
            if (i === layers.length - 1) {
              // Conexiones a la capa de salida
              connectionStrength = prediction.probabilities[j] * 0.8
            } else {
              connectionStrength = 0.2 + Math.random() * 0.3
            }
          }

          // Gradiente para la conexión
          const gradient = ctx.createLinearGradient(x1, y1, x2, y2)
          const alpha = Math.max(0.08, connectionStrength)
          gradient.addColorStop(0, `${prevLayer.color}`)
          gradient.addColorStop(1, `${currLayer.color}`)

          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.strokeStyle = gradient
          ctx.globalAlpha = alpha
          ctx.lineWidth = 1 + connectionStrength * 2
          ctx.stroke()

          // Partícula animada mostrando flujo de datos (solo cuando hay predicción)
          if (prediction && connectionStrength > 0.3) {
            const progress = ((animPhase + k * 10 + j * 5) % 100) / 100
            const px = x1 + (x2 - x1) * progress
            const py = y1 + (y2 - y1) * progress
            
            ctx.beginPath()
            ctx.arc(px, py, 3, 0, Math.PI * 2)
            ctx.fillStyle = currLayer.color
            ctx.globalAlpha = 0.8 * (1 - Math.abs(progress - 0.5) * 2)
            ctx.fill()
          }
        }
      }
    }
    ctx.globalAlpha = 1

    // Dibujar neuronas
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i]
      const x = padX + i * spacing
      const isOutput = i === layers.length - 1

      for (let j = 0; j < layer.units; j++) {
        const y = getNeuronY(j, layer.units)
        const baseRadius = isOutput ? 18 : 12

        // Calcular activación
        let activation = 0.2
        if (prediction) {
          if (isOutput) {
            activation = prediction.probabilities[j] || 0.1
          } else if (i === 1) {
            activation = prediction.hidden1?.[j % prediction.hidden1.length] || 0.3
          } else if (i === 2) {
            activation = prediction.hidden2?.[j % (prediction.hidden2?.length || 1)] || 0.3
          } else {
            activation = 0.5 + Math.sin(animPhase / 10 + j) * 0.2
          }
        }

        const radius = baseRadius + (isOutput && prediction && j === prediction.predictedDigit ? 4 : 0)

        // Sombra/glow de la neurona
        if (activation > 0.3) {
          ctx.beginPath()
          ctx.arc(x, y, radius + 6, 0, Math.PI * 2)
          const glowGradient = ctx.createRadialGradient(x, y, radius, x, y, radius + 10)
          glowGradient.addColorStop(0, `${layer.color}40`)
          glowGradient.addColorStop(1, 'transparent')
          ctx.fillStyle = glowGradient
          ctx.fill()
        }

        // Círculo exterior (borde)
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, Math.PI * 2)
        ctx.fillStyle = '#ffffff'
        ctx.fill()
        ctx.strokeStyle = layer.color
        ctx.lineWidth = 3
        ctx.stroke()

        // Relleno interior según activación
        if (activation > 0.1) {
          ctx.beginPath()
          ctx.arc(x, y, radius - 3, 0, Math.PI * 2)
          
          // Color de activación
          const r = parseInt(layer.color.slice(1, 3), 16)
          const g = parseInt(layer.color.slice(3, 5), 16)
          const b = parseInt(layer.color.slice(5, 7), 16)
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${activation * 0.9})`
          ctx.fill()
        }

        // Texto en las neuronas de salida
        if (isOutput) {
          ctx.fillStyle = activation > 0.5 ? '#ffffff' : '#2D3436'
          ctx.font = 'bold 14px Poppins, sans-serif'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(j.toString(), x, y)

          // Porcentaje al lado si hay predicción
          if (prediction) {
            const pct = (prediction.probabilities[j] * 100).toFixed(0)
            ctx.fillStyle = j === prediction.predictedDigit ? '#00C9B1' : '#718096'
            ctx.font = j === prediction.predictedDigit ? 'bold 11px Poppins' : '10px Poppins'
            ctx.textAlign = 'left'
            ctx.fillText(`${pct}%`, x + radius + 6, y)
          }
        }
      }

      // Etiqueta de la capa
      ctx.fillStyle = layer.color
      ctx.font = 'bold 12px Poppins, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(layer.name, x, h - 20)
      
      // Indicador de "más neuronas" para capas ocultas
      if (i > 0 && i < layers.length - 1) {
        ctx.fillStyle = '#a0aec0'
        ctx.font = '10px Poppins'
        ctx.fillText('⋮', x, getNeuronY(layer.units - 1, layer.units) + 25)
      }
    }

    // Leyenda de flujo de datos
    if (prediction) {
      ctx.fillStyle = '#4a5568'
      ctx.font = '11px Poppins'
      ctx.textAlign = 'left'
      ctx.fillText('→ Flujo de datos activo', 10, 15)
    }

  }, [prediction, animPhase])

  useEffect(() => {
    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [draw])

  return (
    <div className="network-panel">
      <span className="panel-label">🧠 RED NEURONAL — VISUALIZACIÓN EN TIEMPO REAL</span>
      <canvas ref={canvasRef} />
      <div className="network-legend">
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#FF6B6B' }}></span>
          <span>Entrada</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#FFB84C' }}></span>
          <span>Capa Oculta 1</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#00C9B1' }}></span>
          <span>Capa Oculta 2</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#667eea' }}></span>
          <span>Salida</span>
        </div>
      </div>
    </div>
  )
}

export default NetworkVisualizer

