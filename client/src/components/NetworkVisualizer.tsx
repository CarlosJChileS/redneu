import { useRef, useEffect, useCallback } from 'react'
import { NetworkVisualizerProps } from '../types'
import '../styles/NetworkVisualizer.css'

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ prediction }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const layers = [
    { name: 'Input', units: 20 },
    { name: '128', units: 16 },
    { name: '64', units: 12 },
    { name: 'Output', units: 10 }
  ]

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * 2
    canvas.height = rect.height * 2
    ctx.scale(2, 2)

    const w = rect.width
    const h = rect.height
    ctx.clearRect(0, 0, w, h)

    const pad = 50
    const spacing = (w - 2 * pad) / (layers.length - 1)

    // Dibujar conexiones
    ctx.globalAlpha = 0.2
    for (let i = 1; i < layers.length; i++) {
      const x1 = pad + (i - 1) * spacing
      const x2 = pad + i * spacing
      const prev = layers[i - 1].units
      const curr = layers[i].units

      for (let j = 0; j < curr; j += 2) {
        for (let k = 0; k < prev; k += 2) {
          const y1 = getY(k, prev, h - 40)
          const y2 = getY(j, curr, h - 40)
          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.strokeStyle = '#667eea'
          ctx.lineWidth = 0.5
          ctx.stroke()
        }
      }
    }
    ctx.globalAlpha = 1

    // Dibujar neuronas
    for (let i = 0; i < layers.length; i++) {
      const x = pad + i * spacing
      const n = layers[i].units
      const isOut = i === layers.length - 1

      for (let j = 0; j < n; j++) {
        const y = getY(j, n, h - 40)
        const r = isOut ? 14 : 5
        
        // Activación
        let act = 0.2
        if (prediction) {
          if (isOut) {
            act = prediction.probabilities[j] || 0.1
          } else {
            act = (prediction.hidden1?.[j % 10] || 0.2)
          }
        }

        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fillStyle = '#1a1a2e'
        ctx.fill()
        ctx.strokeStyle = isOut ? '#667eea' : '#444'
        ctx.lineWidth = isOut ? 2 : 1
        ctx.stroke()

        // Relleno de activación
        if (act > 0.1) {
          ctx.beginPath()
          ctx.arc(x, y, r - 1, 0, Math.PI * 2)
          if (isOut && prediction && j === prediction.predictedDigit) {
            ctx.fillStyle = `rgba(40, 167, 69, ${0.5 + act * 0.5})`
          } else {
            ctx.fillStyle = `rgba(102, 126, 234, ${act * 0.8})`
          }
          ctx.fill()
        }

        // Número en salida
        if (isOut) {
          ctx.fillStyle = '#fff'
          ctx.font = 'bold 10px sans-serif'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(j.toString(), x, y)
        }
      }

      // Etiqueta
      ctx.fillStyle = '#666'
      ctx.font = '11px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(layers[i].name, x, h - 10)
    }
  }, [prediction])

  const getY = (idx: number, total: number, availH: number): number => {
    const space = Math.min(12, (availH - 40) / Math.max(total - 1, 1))
    const totalH = (total - 1) * space
    return 30 + (availH - totalH) / 2 + idx * space
  }

  useEffect(() => {
    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [draw])

  return (
    <div className="network-panel">
      <span className="panel-label">RED NEURONAL (784 → 128 → 64 → 10)</span>
      <canvas ref={canvasRef} />
    </div>
  )
}

export default NetworkVisualizer
