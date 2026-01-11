import dotenv from 'dotenv'
// Cargar variables de entorno PRIMERO
dotenv.config()

import express from 'express'
import cors from 'cors'
import { groqRouter } from './routes/groq.js'
import { modelRouter } from './routes/model.js'
import { connectDB } from './db/connection.js'

const app = express()
const PORT = process.env.PORT || 4000

// Middlewares
app.use(cors())
app.use(express.json({ limit: '50mb' }))  // Aumentado para modelos grandes

// Conectar a MongoDB
connectDB()

// Rutas
app.use('/api/groq', groqRouter)
app.use('/api/model', modelRouter)

// Health check
app.get('/api/health', (_, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    groqKeySet: !!process.env.GROQ_API_KEY,
    mongoUri: process.env.MONGODB_URI ? 'Configurado' : 'Local (localhost:27017)'
  })
})

// Iniciar servidor
app.listen(PORT, () => {
  console.log(`🚀 Servidor corriendo en http://localhost:${PORT}`)
  console.log(`📡 API Groq disponible en /api/groq/analyze`)
  console.log(`🧠 API Modelo disponible en /api/model`)
  console.log(`🔑 API Key: ${process.env.GROQ_API_KEY ? 'Configurada ✓' : 'NO CONFIGURADA ❌'}`)
})
