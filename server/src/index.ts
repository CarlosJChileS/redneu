import dotenv from 'dotenv'
// Cargar variables de entorno PRIMERO
dotenv.config()

import express from 'express'
import cors from 'cors'
import { groqRouter } from './routes/groq.js'

const app = express()
const PORT = process.env.PORT || 4000

// Middlewares
app.use(cors())
app.use(express.json({ limit: '50mb' }))

// Rutas
app.use('/api/groq', groqRouter)

// Health check
app.get('/api/health', (_, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    groqKeySet: !!process.env.GROQ_API_KEY
  })
})

// Iniciar servidor - Render requiere escuchar en 0.0.0.0
const HOST = '0.0.0.0'
app.listen(Number(PORT), HOST, () => {
  console.log(`🚀 Servidor corriendo en http://${HOST}:${PORT}`)
  console.log(`📡 API Groq disponible en /api/groq/analyze`)
  console.log(`🔑 API Key: ${process.env.GROQ_API_KEY ? 'Configurada ✓' : 'NO CONFIGURADA ❌'}`)
})
