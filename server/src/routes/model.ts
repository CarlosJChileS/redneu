import { Router, Request, Response } from 'express'
import { NeuralModel } from '../models/NeuralModel.js'

const router = Router()
const MODEL_NAME = 'digit-recognition-cnn'

// GET /api/model - Obtener el modelo guardado
router.get('/', async (_req: Request, res: Response) => {
  try {
    const model = await NeuralModel.findOne({ name: MODEL_NAME })
    
    if (!model) {
      return res.status(404).json({ 
        success: false, 
        message: 'No hay modelo guardado en la base de datos' 
      })
    }

    // Convertir Buffer a base64 para enviar al cliente
    const weightsBase64 = model.weightsData.toString('base64')

    res.json({
      success: true,
      model: {
        name: model.name,
        version: model.version,
        modelJson: model.modelJson,
        weightsBase64: weightsBase64,
        accuracy: model.accuracy,
        updatedAt: model.updatedAt
      }
    })
  } catch (error) {
    console.error('Error obteniendo modelo:', error)
    res.status(500).json({ success: false, message: 'Error del servidor' })
  }
})

// POST /api/model - Guardar un nuevo modelo
router.post('/', async (req: Request, res: Response) => {
  try {
    const { modelJson, weightsBase64, accuracy, version } = req.body

    if (!modelJson || !weightsBase64) {
      return res.status(400).json({ 
        success: false, 
        message: 'Se requiere modelJson y weightsBase64' 
      })
    }

    // Convertir base64 a Buffer
    const weightsBuffer = Buffer.from(weightsBase64, 'base64')

    // Actualizar o crear el modelo
    const model = await NeuralModel.findOneAndUpdate(
      { name: MODEL_NAME },
      {
        name: MODEL_NAME,
        version: version || '1.0.0',
        modelJson: typeof modelJson === 'string' ? modelJson : JSON.stringify(modelJson),
        weightsData: weightsBuffer,
        accuracy: accuracy || 0
      },
      { upsert: true, new: true }
    )

    console.log(`💾 Modelo guardado: ${model.name} v${model.version} (${(weightsBuffer.length / 1024 / 1024).toFixed(2)} MB)`)

    res.json({
      success: true,
      message: 'Modelo guardado correctamente',
      model: {
        name: model.name,
        version: model.version,
        accuracy: model.accuracy,
        size: `${(weightsBuffer.length / 1024 / 1024).toFixed(2)} MB`
      }
    })
  } catch (error) {
    console.error('Error guardando modelo:', error)
    res.status(500).json({ success: false, message: 'Error del servidor' })
  }
})

// DELETE /api/model - Eliminar el modelo guardado
router.delete('/', async (_req: Request, res: Response) => {
  try {
    const result = await NeuralModel.findOneAndDelete({ name: MODEL_NAME })
    
    if (!result) {
      return res.status(404).json({ 
        success: false, 
        message: 'No hay modelo para eliminar' 
      })
    }

    res.json({ success: true, message: 'Modelo eliminado' })
  } catch (error) {
    console.error('Error eliminando modelo:', error)
    res.status(500).json({ success: false, message: 'Error del servidor' })
  }
})

// GET /api/model/info - Solo info del modelo (sin pesos)
router.get('/info', async (_req: Request, res: Response) => {
  try {
    const model = await NeuralModel.findOne({ name: MODEL_NAME })
    
    if (!model) {
      return res.status(404).json({ 
        success: false, 
        exists: false 
      })
    }

    res.json({
      success: true,
      exists: true,
      info: {
        name: model.name,
        version: model.version,
        accuracy: model.accuracy,
        size: `${(model.weightsData.length / 1024 / 1024).toFixed(2)} MB`,
        updatedAt: model.updatedAt
      }
    })
  } catch (error) {
    res.status(500).json({ success: false, message: 'Error del servidor' })
  }
})

export const modelRouter = router


