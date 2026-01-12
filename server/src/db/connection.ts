import mongoose from 'mongoose'

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/digit-recognition'

export async function connectDB(): Promise<void> {
  try {
    await mongoose.connect(MONGODB_URI)
    console.log('✅ Conectado a MongoDB')
  } catch (error) {
    console.error('❌ Error conectando a MongoDB:', error)
    console.log('💡 Asegúrate de tener MongoDB corriendo localmente o configura MONGODB_URI')
  }
}

export default mongoose



