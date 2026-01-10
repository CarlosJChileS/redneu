import { useRef, useEffect, useCallback, useState, useImperativeHandle, forwardRef } from 'react'
import { DrawingCanvasProps } from '../types'
import '../styles/DrawingCanvas.css'

interface DrawingCanvasRef {
  clear: () => void
}

const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(({
  onDraw,
  gridSize = 32,
  cellSize = 12,
  disabled = false
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const debounceRef = useRef<NodeJS.Timeout | null>(null)

  const canvasWidth = gridSize * cellSize
  const canvasHeight = gridSize * cellSize

  const drawGrid = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.strokeStyle = '#e0e0e0'
    ctx.lineWidth = 0.3

    for (let i = 0; i <= gridSize; i++) {
      ctx.beginPath()
      ctx.moveTo(i * cellSize, 0)
      ctx.lineTo(i * cellSize, canvasHeight)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, i * cellSize)
      ctx.lineTo(canvasWidth, i * cellSize)
      ctx.stroke()
    }
  }, [gridSize, cellSize, canvasWidth, canvasHeight])

  const clear = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvasWidth, canvasHeight)
    drawGrid()
  }, [canvasWidth, canvasHeight, drawGrid])

  useImperativeHandle(ref, () => ({
    clear
  }))

  useEffect(() => {
    clear()
  }, [clear])

  const getPixels = useCallback((): number[] => {
    const canvas = canvasRef.current
    if (!canvas) return []

    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = 28
    tempCanvas.height = 28
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) return []

    tempCtx.fillStyle = 'white'
    tempCtx.fillRect(0, 0, 28, 28)
    tempCtx.drawImage(canvas, 0, 0, 28, 28)

    const imageData = tempCtx.getImageData(0, 0, 28, 28)
    const pixels: number[] = []

    for (let i = 0; i < imageData.data.length; i += 4) {
      const gray = 1 - (imageData.data[i] / 255)
      pixels.push(gray)
    }
    return pixels
  }, [])

  const paintCell = useCallback((x: number, y: number) => {
    const canvas = canvasRef.current
    if (!canvas || disabled) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cellX = Math.floor(x / cellSize)
    const cellY = Math.floor(y / cellSize)

    if (cellX < 0 || cellX >= gridSize || cellY < 0 || cellY >= gridSize) return

    ctx.fillStyle = 'black'
    ctx.fillRect(cellX * cellSize, cellY * cellSize, cellSize, cellSize)

    drawGrid()

    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }
    debounceRef.current = setTimeout(() => {
      const pixels = getPixels()
      onDraw(canvas, pixels)
    }, 100)
  }, [cellSize, gridSize, drawGrid, getPixels, onDraw, disabled])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled) return
    setIsDrawing(true)
    const rect = canvasRef.current?.getBoundingClientRect()
    if (rect) {
      paintCell(e.clientX - rect.left, e.clientY - rect.top)
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || disabled) return
    const rect = canvasRef.current?.getBoundingClientRect()
    if (rect) {
      paintCell(e.clientX - rect.left, e.clientY - rect.top)
    }
  }

  const handleMouseUp = () => {
    if (disabled) return
    setIsDrawing(false)
    const canvas = canvasRef.current
    if (canvas) {
      const pixels = getPixels()
      onDraw(canvas, pixels)
    }
  }

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault()
    if (disabled) return
    setIsDrawing(true)
    const rect = canvasRef.current?.getBoundingClientRect()
    const touch = e.touches[0]
    if (rect) {
      paintCell(touch.clientX - rect.left, touch.clientY - rect.top)
    }
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    e.preventDefault()
    if (!isDrawing || disabled) return
    const rect = canvasRef.current?.getBoundingClientRect()
    const touch = e.touches[0]
    if (rect) {
      paintCell(touch.clientX - rect.left, touch.clientY - rect.top)
    }
  }

  const handleTouchEnd = (e: React.TouchEvent) => {
    e.preventDefault()
    if (disabled) return
    setIsDrawing(false)
    const canvas = canvasRef.current
    if (canvas) {
      const pixels = getPixels()
      onDraw(canvas, pixels)
    }
  }

  return (
    <div className={`drawing-panel ${disabled ? 'disabled' : ''}`}>
      <span className="panel-label">DIBUJA UN DÍGITO (0-9)</span>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => setIsDrawing(false)}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      />
    </div>
  )
})

export default DrawingCanvas
