import mongoose, { Schema, Document } from 'mongoose'

export interface INeuralModel extends Document {
  name: string
  version: string
  modelJson: string  // La arquitectura del modelo (model.json)
  weightsData: Buffer  // Los pesos del modelo en formato binario
  accuracy: number
  createdAt: Date
  updatedAt: Date
}

const NeuralModelSchema = new Schema<INeuralModel>({
  name: { 
    type: String, 
    required: true, 
    unique: true,
    default: 'digit-recognition-cnn'
  },
  version: { 
    type: String, 
    required: true,
    default: '1.0.0'
  },
  modelJson: { 
    type: String, 
    required: true 
  },
  weightsData: { 
    type: Buffer, 
    required: true 
  },
  accuracy: { 
    type: Number, 
    default: 0 
  }
}, {
  timestamps: true
})

export const NeuralModel = mongoose.model<INeuralModel>('NeuralModel', NeuralModelSchema)



