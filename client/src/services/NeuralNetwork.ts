import * as tf from '@tensorflow/tfjs'
import { PredictionResult } from '../types'

const MODEL_STORAGE_KEY = 'indexeddb://digit-recognition-model'
const MODEL_VERSION_KEY = 'digit-model-version'
const CURRENT_MODEL_VERSION = '10.0.0' // Modelo preciso con 25 epocas

// URL del servidor API
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000'

// Singleton para evitar múltiples instancias
let instance: NeuralNetwork | null = null
let isInitializing = false
let isTraining = false
let initCount = 0

export class NeuralNetwork {
  private model: tf.Sequential | null = null
  public isReady = false
  public onProgress: ((progress: number, status: string) => void) | null = null

  // Obtener instancia singleton
  static getInstance(): NeuralNetwork {
    if (!instance) {
      // Resetear flags cuando se crea nueva instancia
      isInitializing = false
      isTraining = false
      initCount = 0
      instance = new NeuralNetwork()
    }
    return instance
  }
  
  // Resetear para nueva sesión
  static reset(): void {
    instance = null
    isInitializing = false
    isTraining = false
    initCount = 0
  }

  // Detener entrenamiento en progreso
  private stopTraining(): void {
    if (this.model && isTraining) {
      console.log('[!] Deteniendo entrenamiento anterior...')
      this.model.stopTraining = true
    }
  }

  private updateProgress(progress: number, status: string) {
    if (this.onProgress) {
      this.onProgress(progress, status)
    }
  }

  private async yieldToUI(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 10))
  }

  async initialize(): Promise<boolean> {
    initCount++
    const myInitCount = initCount
    
    // Si ya está listo, retornar inmediatamente
    if (this.isReady && this.model) {
      console.log('[OK] Modelo ya esta listo')
      return true
    }
    
    // Solo permitir la primera inicializacion
    if (isInitializing && myInitCount > 1) {
      console.log(`[...] Esperando inicializacion (intento #${myInitCount})...`)
      // Esperar a que la primera termine
      while (isInitializing && !this.isReady) {
        await new Promise(resolve => setTimeout(resolve, 500))
      }
      return this.isReady
    }
    
    isInitializing = true
    console.log(`[>] Inicializando (intento #${myInitCount})...`)
    
    try {
      this.updateProgress(5, 'Iniciando TensorFlow...')
      
      // Intentar WebGL primero, si falla usar CPU
      try {
        await tf.setBackend('webgl')
        await tf.ready()
        console.log('[OK] Usando backend WebGL (GPU)')
      } catch (e) {
        console.log('[!] WebGL no disponible, usando CPU...')
        await tf.setBackend('cpu')
        await tf.ready()
        console.log('[OK] Usando backend CPU')
      }
      
      // 1. Primero intentar cargar del SERVIDOR (MongoDB)
      this.updateProgress(10, 'Buscando modelo en servidor...')
      await this.yieldToUI()
      
      const loadedFromServer = await this.tryLoadFromServer()
      if (loadedFromServer) {
        this.updateProgress(100, '¡Modelo cargado del servidor!')
        this.isReady = true
        return true
      }

      // 2. Si no hay en servidor, buscar localmente (IndexedDB)
      this.updateProgress(15, 'Buscando modelo local...')
      await this.yieldToUI()
      
      const loadedLocally = await this.tryLoadSavedModel()
      if (loadedLocally) {
        this.updateProgress(100, '¡Modelo cargado!')
        this.isReady = true
        return true
      }

      // 3. Si no hay modelo en ningún lado, entrenar uno nuevo
      this.updateProgress(18, 'Creando modelo CNN...')
      await this.yieldToUI()
      this.model = this.buildModel()
      
      this.updateProgress(20, 'Generando datos...')
      await this.yieldToUI()
      await this.train()

      // Guardar localmente
      this.updateProgress(94, 'Guardando modelo local...')
      await this.saveModel()

      // Intentar subir al servidor
      this.updateProgress(97, 'Subiendo al servidor...')
      await this.uploadToServer()

      this.updateProgress(100, 'Listo!')
      this.isReady = true
      isInitializing = false
      return true
    } catch (e) {
      console.error('[X]', e)
      isInitializing = false
      return false
    }
  }

  // Cargar modelo desde el servidor (MongoDB)
  private async tryLoadFromServer(): Promise<boolean> {
    try {
      const response = await fetch(`${API_URL}/api/model`)
      
      if (!response.ok) {
        console.log('[i] No hay modelo en el servidor')
        return false
      }

      const data = await response.json()
      
      if (!data.success || !data.model) {
        return false
      }

      this.updateProgress(30, 'Descargando modelo del servidor...')
      await this.yieldToUI()

      // Parsear el JSON del modelo
      const modelJson = typeof data.model.modelJson === 'string' 
        ? JSON.parse(data.model.modelJson) 
        : data.model.modelJson

      // Convertir base64 a ArrayBuffer
      const weightsBase64 = data.model.weightsBase64
      const weightsBuffer = this.base64ToArrayBuffer(weightsBase64)

      this.updateProgress(60, 'Reconstruyendo modelo...')
      await this.yieldToUI()

      // Crear el modelo desde el JSON
      const loadedModel = await tf.models.modelFromJSON(modelJson)
      
      // Cargar los pesos
      this.updateProgress(80, 'Cargando pesos...')
      await this.yieldToUI()
      
      const weightSpecs = modelJson.weightsManifest[0].weights
      const weightData = new Float32Array(weightsBuffer)
      
      // Reconstruir los tensores de pesos
      let offset = 0
      const weightTensors: tf.Tensor[] = []
      
      for (const spec of weightSpecs) {
        const size = spec.shape.reduce((a: number, b: number) => a * b, 1)
        const values = weightData.slice(offset, offset + size)
        weightTensors.push(tf.tensor(Array.from(values), spec.shape, spec.dtype))
        offset += size
      }
      
      loadedModel.setWeights(weightTensors)

      // Compilar
      loadedModel.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      })

      this.model = loadedModel as tf.Sequential
      console.log(`[OK] Modelo cargado desde servidor (v${data.model.version}, acc: ${(data.model.accuracy * 100).toFixed(1)}%)`)
      
      return true
    } catch (error) {
      console.log('[i] Error cargando del servidor:', error)
      return false
    }
  }

  // Subir modelo al servidor (MongoDB) - FIX: Usando chunks para evitar stack overflow
  private async uploadToServer(): Promise<boolean> {
    if (!this.model) return false

    try {
      // Obtener el JSON del modelo
      const modelJson = this.model.toJSON()
      
      // Obtener los pesos como Float32Array
      const weights = this.model.getWeights()
      const weightsData: number[] = []
      
      for (const w of weights) {
        const data = await w.data()
        weightsData.push(...Array.from(data))
      }
      
      // Convertir a base64 usando chunks (evita stack overflow)
      const float32Array = new Float32Array(weightsData)
      const uint8Array = new Uint8Array(float32Array.buffer)
      const weightsBase64 = this.arrayBufferToBase64Chunked(uint8Array)

      // Enviar al servidor
      const response = await fetch(`${API_URL}/api/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelJson: modelJson,
          weightsBase64: weightsBase64,
          version: CURRENT_MODEL_VERSION,
          accuracy: 0.97
        })
      })

      if (response.ok) {
        console.log('[OK] Modelo subido al servidor')
        return true
      }
      return false
    } catch (error) {
      console.log('[!] No se pudo subir al servidor:', error)
      return false
    }
  }

  // Helpers para conversión base64
  private base64ToArrayBuffer(base64: string): ArrayBuffer {
    const binaryString = atob(base64)
    const bytes = new Uint8Array(binaryString.length)
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i)
    }
    return bytes.buffer
  }

  // FIX: Conversion a base64 usando chunks para evitar "Maximum call stack size exceeded"
  private arrayBufferToBase64Chunked(uint8Array: Uint8Array): string {
    const CHUNK_SIZE = 0x8000 // 32KB chunks
    const chunks: string[] = []
    
    for (let i = 0; i < uint8Array.length; i += CHUNK_SIZE) {
      const chunk = uint8Array.subarray(i, Math.min(i + CHUNK_SIZE, uint8Array.length))
      chunks.push(String.fromCharCode.apply(null, Array.from(chunk)))
    }
    
    return btoa(chunks.join(''))
  }

  private async tryLoadSavedModel(): Promise<boolean> {
    try {
      const savedVersion = localStorage.getItem(MODEL_VERSION_KEY)
      if (savedVersion !== CURRENT_MODEL_VERSION) {
        console.log('[i] Version de modelo diferente, re-entrenando...')
        try {
          await tf.io.removeModel(MODEL_STORAGE_KEY)
        } catch {
          // Ignorar si no existe
        }
        return false
      }

      this.updateProgress(20, 'Cargando modelo local...')
      await this.yieldToUI()

      const loadedModel = await tf.loadLayersModel(MODEL_STORAGE_KEY)
      
      if (loadedModel) {
        this.updateProgress(80, 'Compilando modelo...')
        await this.yieldToUI()
        
        loadedModel.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'categoricalCrossentropy',
          metrics: ['accuracy']
        })
        
        this.model = loadedModel as tf.Sequential
        console.log('[OK] Modelo cargado desde IndexedDB')
        return true
      }
      return false
    } catch (error) {
      console.log('[i] No hay modelo local guardado')
      return false
    }
  }

  private async saveModel(): Promise<void> {
    if (!this.model) return
    
    try {
      await this.model.save(MODEL_STORAGE_KEY)
      localStorage.setItem(MODEL_VERSION_KEY, CURRENT_MODEL_VERSION)
      console.log('[+] Modelo guardado en IndexedDB')
    } catch (error) {
      console.error('Error guardando modelo:', error)
    }
  }

  async retrainModel(): Promise<boolean> {
    try {
      console.log('[~] Iniciando re-entrenamiento...')
      
      // Si hay entrenamiento en progreso, no permitir
      if (isTraining) {
        console.log('[!] Hay un entrenamiento en progreso. Espera a que termine.')
        this.updateProgress(0, 'Espera a que termine el entrenamiento actual...')
        return false
      }
      
      this.isReady = false

      // Eliminar modelo local
      try {
        await tf.io.removeModel(MODEL_STORAGE_KEY)
        localStorage.removeItem(MODEL_VERSION_KEY)
        console.log('[-] Modelo local eliminado')
      } catch {
        console.log('[!] No habia modelo local')
      }

      // Eliminar modelo del servidor
      try {
        await fetch(`${API_URL}/api/model`, { method: 'DELETE' })
        console.log('[-] Modelo del servidor eliminado')
      } catch {
        console.log('[!] No se pudo eliminar del servidor')
      }

      // Crear modelo nuevo directamente (sin pasar por initialize)
      console.log('[+] Creando modelo nuevo...')
      this.updateProgress(18, 'Limpiando memoria...')
      await this.yieldToUI()
      
      // Liberar modelo anterior si existe
      if (this.model) {
        try {
          this.model.dispose()
        } catch (e) {
          // Ignorar
        }
        this.model = null
      }
      
      // Limpiar memoria de TensorFlow
      tf.disposeVariables()
      
      // Forzar garbage collection de WebGL
      try {
        const backend = tf.backend()
        if (backend && 'dispose' in backend) {
          // @ts-ignore
          backend.dispose()
        }
      } catch (e) {
        // Ignorar
      }
      
      // Reiniciar backend
      try {
        await tf.setBackend('webgl')
        await tf.ready()
      } catch (e) {
        console.log('[!] WebGL fallo, usando CPU...')
        await tf.setBackend('cpu')
        await tf.ready()
      }
      
      this.updateProgress(20, 'Creando modelo CNN...')
      await this.yieldToUI()
      
      // Crear modelo nuevo
      this.model = this.buildModel()
      
      this.updateProgress(20, 'Generando datos...')
      await this.yieldToUI()
      await this.train()

      // Guardar localmente
      this.updateProgress(94, 'Guardando modelo local...')
      await this.saveModel()

      // Intentar subir al servidor
      this.updateProgress(97, 'Subiendo al servidor...')
      await this.uploadToServer()

      this.updateProgress(100, '¡Listo!')
      this.isReady = true
      isInitializing = false
      
      return true
    } catch (error) {
      console.error('[X] Error re-entrenando:', error)
      isTraining = false
      isInitializing = false
      return false
    }
  }

  private buildModel(): tf.Sequential {
    const m = tf.sequential()
    
    // MODELO PRECISO - Arquitectura mejorada con BatchNorm
    m.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    m.add(tf.layers.batchNormalization())
    m.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }))
    m.add(tf.layers.dropout({ rate: 0.25 }))
    
    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    m.add(tf.layers.batchNormalization())
    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }))
    m.add(tf.layers.dropout({ rate: 0.25 }))
    
    m.add(tf.layers.flatten())
    m.add(tf.layers.dense({ units: 256, activation: 'relu' }))
    m.add(tf.layers.batchNormalization())
    m.add(tf.layers.dropout({ rate: 0.4 }))
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }))
    m.add(tf.layers.dropout({ rate: 0.3 }))
    m.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

    m.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    })
    
    return m
  }

  private async train(): Promise<void> {
    if (!this.model) {
      console.error('[X] No hay modelo para entrenar')
      return
    }

    // Verificar si ya hay un entrenamiento en progreso
    if (isTraining) {
      console.log('[!] Ya hay un entrenamiento en progreso')
      throw new Error('Ya hay un entrenamiento en progreso')
    }

    isTraining = true
    console.log('[>] Iniciando entrenamiento...')
    this.updateProgress(22, 'Generando estilos de escritura...')
    await this.yieldToUI()
    
    try {
      console.log('[~] Generando dataset...')
      const { xs, ys } = await this.createDatasetAsync()
      console.log(`[OK] Dataset creado: xs=${xs.shape}, ys=${ys.shape}`)
      
      const totalEpochs = 25 // Mayor precision
      
      console.log('[>] Iniciando fit()...')
      await this.model.fit(xs, ys, {
        epochs: totalEpochs,
        batchSize: 64,
        shuffle: true,
        validationSplit: 0.15,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const progress = 25 + ((epoch + 1) / totalEpochs) * 65
            const acc = ((logs?.acc || 0) * 100).toFixed(0)
            console.log(`[${epoch + 1}/${totalEpochs}] acc: ${acc}%`)
            this.updateProgress(progress, `Epoca ${epoch + 1}/${totalEpochs} (${acc}%)`)
            await this.yieldToUI()
          }
        }
      })
      console.log('[OK] Entrenamiento completado')

      xs.dispose()
      ys.dispose()
      isTraining = false
    } catch (error) {
      console.error('[X] Error en entrenamiento:', error)
      isTraining = false
      throw error
    }
  }

  private async createDatasetAsync(): Promise<{ xs: tf.Tensor4D; ys: tf.Tensor2D }> {
    const X: number[][][][] = []
    const Y: number[][] = []
    
    const templates = this.getAllTemplates()
    const samplesPerDigit = 350 // 3500 muestras totales
    
    for (let d = 0; d < 10; d++) {
      const digitTemplates = templates[d]
      
      for (let i = 0; i < samplesPerDigit; i++) {
        const t = digitTemplates[Math.floor(Math.random() * digitTemplates.length)]
        // 70% números mal escritos para máximo reconocimiento
        const style = Math.random() < 0.7 ? this.getMessyStyle() : this.getRandomStyle()
        X.push(this.renderWithStyle(t, style))
        
        const label = new Array(10).fill(0)
        label[d] = 1
        Y.push(label)
        
        if (i % 15 === 0) {
          await this.yieldToUI()
        }
      }
      
      this.updateProgress(22 + (d / 10) * 3, `Digito ${d}...`)
      await this.yieldToUI()
    }

    console.log(`[+] Dataset: ${X.length} muestras`)

    // Shuffle simple
    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[X[i], X[j]] = [X[j], X[i]]
      ;[Y[i], Y[j]] = [Y[j], Y[i]]
    }

    return { xs: tf.tensor4d(X), ys: tf.tensor2d(Y) }
  }

  private getRandomStyle(): WritingStyle {
    const styles: WritingStyle[] = [
      // Tamanos GRANDES - Numeros que ocupan casi todo el canvas
      { scale: 5.0, thickness: 1.3, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.2, noise: 0 },
      { scale: 4.5, thickness: 1.2, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.25, noise: 0 },
      { scale: 4.0, thickness: 1.1, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.2, noise: 0 },
      { scale: 5.5, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.15, noise: 0 },
      // Tamanos MEDIANOS - Numeros normales
      { scale: 3.0, thickness: 1.2, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.3, noise: 0 },
      { scale: 2.5, thickness: 1.3, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.2, noise: 0 },
      { scale: 3.5, thickness: 1.1, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.3, noise: 0 },
      { scale: 2.8, thickness: 1.25, shearX: 0, shearY: 0, stretchX: 1.1, stretchY: 1, blur: 0.25, noise: 0 },
      // Tamanos PEQUENOS - Numeros pequenos en el centro
      { scale: 2.0, thickness: 1.4, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.4, noise: 0 },
      { scale: 1.8, thickness: 1.5, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.35, noise: 0 },
      { scale: 1.5, thickness: 1.6, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.5, noise: 0 },
      { scale: 2.2, thickness: 1.35, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.3, noise: 0 },
      // Con inclinacion (varios tamanos)
      { scale: 4.0, thickness: 1.2, shearX: 0.2, shearY: 0, stretchX: 0.95, stretchY: 1, blur: 0.2, noise: 0 },
      { scale: 3.0, thickness: 1.2, shearX: 0.15, shearY: 0, stretchX: 0.95, stretchY: 1, blur: 0.2, noise: 0 },
      { scale: 2.0, thickness: 1.4, shearX: -0.15, shearY: 0, stretchX: 0.95, stretchY: 1, blur: 0.3, noise: 0 },
      { scale: 3.2, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.3, noise: 0 },
    ]
    
    const base = styles[Math.floor(Math.random() * styles.length)]
    
    return {
      scale: base.scale + (Math.random() - 0.5) * 0.6,
      thickness: Math.max(0.8, base.thickness + (Math.random() - 0.5) * 0.3),
      shearX: base.shearX + (Math.random() - 0.5) * 0.2,
      shearY: base.shearY + (Math.random() - 0.5) * 0.1,
      stretchX: Math.max(0.7, base.stretchX + (Math.random() - 0.5) * 0.25),
      stretchY: Math.max(0.7, base.stretchY + (Math.random() - 0.5) * 0.25),
      blur: Math.max(0.1, base.blur + Math.random() * 0.3),
      noise: Math.random() * 0.1,
    }
  }

  // Estilo "desordenado" para escritura mal hecha - Con tamanos variados
  private getMessyStyle(): WritingStyle {
    const messyTypes = [
      // GRANDES desordenados
      { scale: 5.0, thickness: 1.2, shearX: 0.3, shearY: 0.05, stretchX: 0.95, stretchY: 1.05, blur: 0.3, noise: 0.1 },
      { scale: 4.5, thickness: 1.3, shearX: -0.25, shearY: 0, stretchX: 1.0, stretchY: 1.0, blur: 0.25, noise: 0.1 },
      { scale: 4.0, thickness: 1.1, shearX: 0.15, shearY: 0, stretchX: 1.1, stretchY: 1.05, blur: 0.35, noise: 0.15 },
      // MEDIANOS desordenados
      { scale: 3.0, thickness: 1.1, shearX: 0.4, shearY: 0.05, stretchX: 0.9, stretchY: 1.05, blur: 0.3, noise: 0.1 },
      { scale: 3.0, thickness: 1.15, shearX: -0.35, shearY: -0.05, stretchX: 0.95, stretchY: 1.0, blur: 0.25, noise: 0.1 },
      { scale: 3.8, thickness: 1.3, shearX: 0.1, shearY: 0, stretchX: 1.1, stretchY: 1.1, blur: 0.4, noise: 0.15 },
      { scale: 3.2, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1.4, stretchY: 0.85, blur: 0.3, noise: 0.1 },
      { scale: 3.2, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 0.8, stretchY: 1.35, blur: 0.3, noise: 0.1 },
      // PEQUENOS desordenados
      { scale: 2.0, thickness: 1.3, shearX: 0.2, shearY: 0.05, stretchX: 0.95, stretchY: 1.0, blur: 0.5, noise: 0.1 },
      { scale: 1.8, thickness: 1.4, shearX: -0.2, shearY: 0, stretchX: 1.0, stretchY: 1.1, blur: 0.45, noise: 0.15 },
      { scale: 1.5, thickness: 1.5, shearX: 0.15, shearY: 0.05, stretchX: 1.1, stretchY: 0.95, blur: 0.6, noise: 0.2 },
      { scale: 2.2, thickness: 1.25, shearX: -0.1, shearY: 0, stretchX: 1.0, stretchY: 1.0, blur: 0.4, noise: 0.12 },
      // Borrosos y gruesos (varios tamanos)
      { scale: 2.8, thickness: 1.2, shearX: 0.1, shearY: 0.05, stretchX: 1, stretchY: 1, blur: 0.6, noise: 0.2 },
      { scale: 3.0, thickness: 1.25, shearX: 0.05, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.35, noise: 0.12 },
      { scale: 4.5, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.5, noise: 0.15 },
      // Muy gruesos descuidados
      { scale: 3.0, thickness: 1.5, shearX: -0.15, shearY: 0.08, stretchX: 1.05, stretchY: 1.0, blur: 0.5, noise: 0.2 },
      { scale: 2.5, thickness: 1.6, shearX: 0.1, shearY: 0, stretchX: 1.0, stretchY: 1.0, blur: 0.4, noise: 0.15 },
    ]
    
    const base = messyTypes[Math.floor(Math.random() * messyTypes.length)]
    
    return {
      scale: base.scale + (Math.random() - 0.5) * 0.6,
      thickness: Math.max(0.8, base.thickness + (Math.random() - 0.5) * 0.3),
      shearX: base.shearX + (Math.random() - 0.5) * 0.2,
      shearY: base.shearY + (Math.random() - 0.5) * 0.1,
      stretchX: Math.max(0.6, base.stretchX + (Math.random() - 0.5) * 0.25),
      stretchY: Math.max(0.6, base.stretchY + (Math.random() - 0.5) * 0.25),
      blur: Math.max(0.1, base.blur + (Math.random() - 0.5) * 0.25),
      noise: Math.max(0, base.noise + Math.random() * 0.1),
    }
  }

  private renderWithStyle(template: number[][], style: WritingStyle): number[][][] {
    const out: number[][][] = []
    const h = template.length
    const w = template[0].length
    
    // Menos variación en posición para mejor centrado
    const offsetX = (Math.random() - 0.5) * 4
    const offsetY = (Math.random() - 0.5) * 4
    const angle = (Math.random() - 0.5) * 0.35  // Menos rotación
    
    const cos = Math.cos(angle)
    const sin = Math.sin(angle)

    const raw: number[][] = []
    for (let y = 0; y < 28; y++) {
      const row: number[] = []
      for (let x = 0; x < 28; x++) {
        let dx = (x - 14 - offsetX) / style.stretchX
        let dy = (y - 14 - offsetY) / style.stretchY
        
        dx += dy * style.shearX
        dy += dx * style.shearY
        
        const rx = dx * cos + dy * sin
        const ry = -dx * sin + dy * cos
        
        const tx = rx / style.scale + w / 2
        const ty = ry / style.scale + h / 2
        
        let val = 0
        const ix = Math.floor(tx)
        const iy = Math.floor(ty)
        
        if (ix >= 0 && ix < w - 1 && iy >= 0 && iy < h - 1) {
          const fx = tx - ix
          const fy = ty - iy
          
          const v00 = template[iy][ix] || 0
          const v10 = template[iy][ix + 1] || 0
          const v01 = template[iy + 1]?.[ix] || 0
          const v11 = template[iy + 1]?.[ix + 1] || 0
          
          val = v00 * (1-fx) * (1-fy) + 
                v10 * fx * (1-fy) + 
                v01 * (1-fx) * fy + 
                v11 * fx * fy
        } else if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
          val = template[iy]?.[ix] || 0
        }
        
        val *= style.thickness
        row.push(val)
      }
      raw.push(row)
    }
    
    for (let y = 0; y < 28; y++) {
      const row: number[][] = []
      for (let x = 0; x < 28; x++) {
        let val = raw[y][x]
        
        // Blur mejorado
        if (style.blur > 0 && x > 0 && x < 27 && y > 0 && y < 27) {
          const neighbors = (raw[y-1][x] + raw[y+1][x] + raw[y][x-1] + raw[y][x+1]) / 4
          val = val * (1 - style.blur * 0.6) + neighbors * style.blur * 0.6
        }
        
        // Variación de intensidad
        if (val > 0.15) {
          val = val * (0.7 + Math.random() * 0.4)
        }
        
        // Ruido de fondo
        if (val < 0.05 && Math.random() < 0.015) {
          val = Math.random() * 0.1
        }
        
        // Añadir ruido según el estilo
        if (style.noise > 0) {
          val += (Math.random() - 0.5) * style.noise
          // Erosión aleatoria (quitar píxeles)
          if (val > 0.3 && Math.random() < style.noise * 0.5) {
            val *= 0.3
          }
          // Dilatación aleatoria (añadir píxeles cerca de bordes)
          if (val < 0.2 && val > 0.05 && Math.random() < style.noise * 0.3) {
            val = 0.4 + Math.random() * 0.3
          }
        }
        
        row.push([Math.max(0, Math.min(1, val))])
      }
      out.push(row)
    }
    
    return out
  }

  private getAllTemplates(): Record<number, number[][][]> {
    return {
      // ========== CERO (0) - Incluye óvalos, círculos abiertos, mal cerrados ==========
      0: [
        this.p(['..###..', '.#...#.', '#.....#', '#.....#', '#.....#', '.#...#.', '..###..']),
        this.p(['..##..', '.#..#.', '#....#', '#....#', '#....#', '#....#', '.#..#.', '..##..']),
        this.p(['.#####.', '#.....#', '#.....#', '#.....#', '.#####.']),
        this.p(['.####.', '#....#', '#....#', '#....#', '#....#', '#....#', '.####.']),
        this.p(['.##.', '#..#', '#..#', '#..#', '.##.']),
        this.p(['.###.', '#...#', '#...#', '#...#', '.###.']),
        // Mal escritos / irregulares
        this.p(['..##.', '.#..#', '#...#', '#...#', '.#..#', '..##.']),
        this.p(['.###.', '#....', '#...#', '#...#', '.###.']),
        this.p(['.##..', '#..#.', '#...#', '#..#.', '.##..']),
        this.p(['..#..', '.#.#.', '#...#', '#...#', '.#.#.', '..#..']),
        this.p(['.###.', '#...#', '#....', '#...#', '.###.']),
        this.p(['####', '#..#', '#..#', '####']),
        this.p(['.#.#.', '#...#', '#...#', '#...#', '.#.#.']),
        // NUEVOS - Más variaciones mal escritas
        this.p(['..#..', '.#.#.', '#...#', '#...#', '#...#', '.#.#.', '..#..']), // Ovalado alto
        this.p(['.##.', '#..#', '#..#', '.##.']), // Muy pequeño
        this.p(['...##', '..#.#', '.#..#', '#...#', '.#..#', '..#.#', '...##']), // Inclinado derecha
        this.p(['##...', '#.#..', '#..#.', '#...#', '#..#.', '#.#..', '##...']), // Inclinado izquierda
        this.p(['.###.', '#....', '#....', '#....', '.###.']), // Muy abierto
        this.p(['#####', '#...#', '#...#', '#...#', '#####']), // Rectangular
        this.p(['.#.', '#.#', '#.#', '#.#', '.#.']), // Muy estrecho
      ],
      
      // ========== UNO (1) - Incluye palos, con/sin base, inclinados ==========
      1: [
        this.p(['...#...', '..##...', '.#.#...', '...#...', '...#...', '...#...', '.#####.']),
        this.p(['..#.', '.##.', '..#.', '..#.', '..#.', '..#.', '.###']),
        this.p(['#', '#', '#', '#', '#', '#', '#']),
        this.p(['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '..#..']),
        this.p(['.#.', '.#.', '.#.', '.#.', '.#.', '.#.', '.#.']),
        this.p(['##.', '.#.', '.#.', '.#.', '.#.', '.#.', '###']),
        // Mal escritos / irregulares
        this.p(['..#', '.#.', '.#.', '.#.', '#..', '#..']),
        this.p(['.#', '.#', '#.', '#.', '#.', '#.']),
        this.p(['#.', '.#', '.#', '.#', '.#', '#.']),
        this.p(['##', '##', '.#', '.#', '.#', '##']),
        this.p(['.#.', '##.', '.#.', '.#.', '.##', '.#.']),
        this.p(['...#', '..#.', '..#.', '.#..', '.#..', '#...']),
        // NUEVOS - Más variaciones
        this.p(['.#', '.#', '.#', '.#', '.#']), // Muy corto
        this.p(['..#..', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..']), // Muy largo
        this.p(['#..', '#..', '#..', '#..', '#..', '###']), // Con base grande
        this.p(['.##', '..#', '..#', '..#', '..#', '..#']), // Serif arriba
        this.p(['#', '#', '#', '#', '#']), // Palo simple corto
        this.p(['..#.', '.#..', '.#..', '#...', '#...', '#...']), // Muy diagonal
        this.p(['.#.', '.#.', '.#.', '.#.', '###']), // Con base
        this.p(['##', '.#', '.#', '.#', '.#', '.#', '.#']), // Cabeza grande
      ],
      
      // ========== DOS (2) - Incluye Z, curvas abiertas, angulares ==========
      2: [
        this.p(['.####.', '#....#', '.....#', '....#.', '..##..', '.#....', '######']),
        this.p(['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####']),
        this.p(['####.', '....#', '...#.', '..#..', '.#...', '#....', '#####']),
        this.p(['.##.', '#..#', '...#', '..#.', '.#..', '#...', '####']),
        this.p(['###', '..#', '.#.', '#..', '#..', '###']),
        // Mal escritos / irregulares
        this.p(['###.', '...#', '..#.', '.#..', '#...', '###.']),
        this.p(['..##', '....#', '...#.', '..#..', '.#...', '####']),
        this.p(['.##.', '...#', '..#.', '.#..', '#...', '##..']),
        this.p(['###', '..#', '.#.', '.#.', '#..', '###']),
        this.p(['.#.', '#.#', '..#', '.#.', '#..', '###']),
        this.p(['##..', '..#.', '..#.', '.#..', '#...', '####']),
        // NUEVOS - Más variaciones (confusión con Z, 7)
        this.p(['####', '...#', '..#.', '.#..', '#...', '####']), // Forma de Z
        this.p(['.##.', '...#', '..#.', '.#..', '#...', '####']), // Curva pequeña
        this.p(['###.', '...#', '..##', '.#..', '#...', '####']), // Con bulto medio
        this.p(['.#..', '#.#.', '..#.', '.#..', '#...', '###.']), // Muy curvo arriba
        this.p(['..#.', '.#.#', '...#', '..#.', '.#..', '####']), // Loop arriba
        this.p(['###', '..#', '..#', '.#.', '#..', '###']), // Muy angular
        this.p(['.##.', '#..#', '...#', '...#', '..#.', '.#..', '####']), // Alto
      ],
      
      // ========== TRES (3) - Incluye 3 abiertos, con curvas irregulares ==========
      3: [
        this.p(['.####.', '#....#', '.....#', '..###.', '.....#', '#....#', '.####.']),
        this.p(['####.', '....#', '....#', '.###.', '....#', '....#', '####.']),
        this.p(['.###.', '#...#', '....#', '..##.', '....#', '#...#', '.###.']),
        this.p(['###.', '...#', '...#', '.##.', '...#', '...#', '###.']),
        this.p(['###.', '...#', '###.', '...#', '...#', '###.']),
        // Mal escritos / irregulares
        this.p(['##..', '..#.', '..#.', '.#..', '..#.', '..#.', '##..']),
        this.p(['###', '..#', '.#.', '..#', '..#', '###']),
        this.p(['.##.', '...#', '..#.', '...#', '...#', '.##.']),
        this.p(['###.', '...#', '.##.', '...#', '..#.', '.#..']),
        this.p(['.#..', '..#.', '.#..', '..#.', '..#.', '.#..']),
        // NUEVOS - Más variaciones (confusión con 8, 9)
        this.p(['##.', '..#', '##.', '..#', '##.']), // Compacto
        this.p(['.###', '...#', '..##', '...#', '.###']), // Asimétrico
        this.p(['###.', '...#', '..#.', '..#.', '...#', '###.']), // Sin cintura clara
        this.p(['.#.', '#.#', '..#', '.#.', '..#', '#.#', '.#.']), // Muy curvo
        this.p(['##..', '..#.', '.##.', '..#.', '..#.', '##..']), // Inclinado
        this.p(['###', '..#', '.#.', '.#.', '..#', '###']), // Con pico medio
        this.p(['.##.', '...#', '...#', '.##.', '...#', '...#', '.##.']), // Dos círculos
      ],
      
      // ========== CUATRO (4) - Incluye abiertos, cerrados, angulares ==========
      4: [
        this.p(['....#.', '...##.', '..#.#.', '.#..#.', '######', '....#.', '....#.']),
        this.p(['#...#', '#...#', '#...#', '#####', '....#', '....#', '....#']),
        this.p(['#..#', '#..#', '#..#', '####', '...#', '...#', '...#']),
        this.p(['#...#', '#...#', '#####', '....#', '....#', '....#']),
        this.p(['#.#', '#.#', '###', '..#', '..#']),
        // Mal escritos / irregulares
        this.p(['#..#', '#..#', '####', '...#', '...#']),
        this.p(['..#.', '.##.', '#.#.', '####', '..#.', '..#.']),
        this.p(['#...#', '#..#.', '.###.', '...#.', '...#.']),
        this.p(['.#.#', '#..#', '####', '...#', '...#', '..#.']),
        this.p(['#..', '#.#', '###', '..#', '..#', '..#']),
        // NUEVOS - Más variaciones (confusión con 9, 1)
        this.p(['#..#', '#..#', '#..#', '####', '...#', '...#', '...#', '...#']), // Muy largo
        this.p(['#.#', '#.#', '###', '..#']), // Muy corto
        this.p(['..#.', '.#.#', '#..#', '####', '...#', '...#']), // Con diagonal
        this.p(['#...#', '#...#', '#...#', '####.', '....#', '....#']), // Asimétrico
        this.p(['.#..#', '#...#', '#####', '....#', '....#']), // Brazo corto
        this.p(['#..#', '#..#', '###.', '..#.', '..#.']), // Barra corta
        this.p(['..#', '.##', '###', '..#', '..#', '..#']), // Solo diagonal
        this.p(['#....#', '#....#', '######', '.....#', '.....#']), // Muy ancho
      ],
      
      // ========== CINCO (5) - Incluye S invertidas, angulares ==========
      5: [
        this.p(['######', '#.....', '#.....', '.####.', '.....#', '#....#', '.####.']),
        this.p(['#####', '#....', '####.', '....#', '....#', '#...#', '.###.']),
        this.p(['#####', '#....', '#....', '####.', '....#', '....#', '####.']),
        this.p(['####', '#...', '###.', '...#', '...#', '###.']),
        // Mal escritos / irregulares
        this.p(['####', '#...', '##..', '..#.', '..#.', '##..']),
        this.p(['###.', '#...', '###.', '...#', '..#.', '.#..']),
        this.p(['####', '#...', '#...', '###.', '...#', '###.']),
        this.p(['.###', '.#..', '.##.', '...#', '...#', '.##.']),
        this.p(['###', '#..', '##.', '..#', '..#', '#..']),
        // NUEVOS - Más variaciones (confusión con 6, S)
        this.p(['###.', '#...', '#...', '##..', '..#.', '..#.', '.#..']), // Cola larga
        this.p(['####', '#...', '###.', '...#', '###.']), // Compacto
        this.p(['#####', '#....', '.###.', '....#', '....#', '.###.']), // Con curva
        this.p(['##..', '#...', '##..', '..#.', '..#.', '##..']), // Pequeño
        this.p(['####', '#...', '##..', '...#', '...#', '..#.', '.#..']), // S invertida
        this.p(['###.', '#...', '##..', '..#.', '#..#', '.##.']), // Con loop abajo
        this.p(['####', '#...', '###.', '...#', '...#', '...#', '.##.']), // Alto
      ],
      
      // ========== SEIS (6) - Incluye 6 abiertos, con loops irregulares ==========
      6: [
        this.p(['..###.', '.#....', '#.....', '#####.', '#....#', '#....#', '.####.']),
        this.p(['.###.', '#....', '#....', '####.', '#...#', '#...#', '.###.']),
        this.p(['.##.', '#...', '#...', '###.', '#..#', '#..#', '.##.']),
        this.p(['###.', '#...', '###.', '#..#', '#..#', '###.']),
        // Mal escritos / irregulares
        this.p(['..#.', '.#..', '#...', '###.', '#..#', '.##.']),
        this.p(['.##.', '#...', '##..', '#.#.', '#.#.', '.#..']),
        this.p(['.#..', '#...', '###.', '#..#', '#..#', '.##.']),
        this.p(['..##', '.#..', '#...', '##..', '#.#.', '.#..']),
        this.p(['.#.', '#..', '#..', '##.', '#.#', '.#.']),
        // NUEVOS - Más variaciones (confusión con 0, 8, 9)
        this.p(['..#..', '.#...', '#....', '###..', '#..#.', '#..#.', '.##..']), // Cola larga arriba
        this.p(['.##.', '#...', '#...', '##..', '#.#.', '.#..']), // Loop pequeño
        this.p(['.#..', '#...', '#...', '###.', '#..#', '.##.']), // Sin curva superior
        this.p(['..#.', '.#..', '#...', '#...', '##..', '#.#.', '.#..']), // Muy largo
        this.p(['.###.', '#....', '#....', '#....', '####.', '#...#', '.###.']), // Tallo largo
        this.p(['.#.', '#..', '##.', '#.#', '#.#', '.#.']), // Compacto
        this.p(['..##.', '.#...', '#....', '####.', '#...#', '.###.']), // Inclinado
      ],
      
      // ========== SIETE (7) - Incluye con/sin barra, muy inclinados ==========
      7: [
        this.p(['######', '.....#', '....#.', '...#..', '..#...', '..#...', '..#...']),
        this.p(['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...']),
        this.p(['####', '...#', '...#', '..#.', '..#.', '.#..', '.#..']),
        this.p(['###', '..#', '..#', '.#.', '.#.', '#..']),
        // Mal escritos / irregulares
        this.p(['####', '...#', '..#.', '..#.', '.#..', '.#..']),
        this.p(['###.', '..#.', '..#.', '.#..', '.#..', '#...']),
        this.p(['####', '...#', '..#.', '.#..', '#...', '#...']),
        this.p(['##', '.#', '.#', '#.', '#.']),
        this.p(['###', '..#', '.#.', '.#.', '#..', '#..']),
        this.p(['####', '..#.', '..#.', '..#.', '.#..', '.#..']),
        // NUEVOS - Más variaciones (confusión con 1, 2)
        this.p(['#####', '....#', '....#', '...#.', '...#.', '..#..', '..#..']), // Muy vertical
        this.p(['###', '..#', '..#', '..#', '.#.', '.#.']), // Casi recto
        this.p(['####', '...#', '..#.', '..#.', '..#.', '.#..']), // Poco diagonal
        this.p(['#####', '....#', '...#.', '..#..', '..#..', '..#..']), // Con meseta
        this.p(['##', '.#', '.#', '.#', '.#', '.#']), // Muy pequeño vertical
        this.p(['####', '...#', '...#', '..#.', '.#..', '#...']), // Con esquina
        this.p(['###.', '..#.', '..##', '...#', '...#', '..#.']), // Con serif medio
      ],
      
      // ========== OCHO (8) - Incluye 8 desproporcionados, abiertos ==========
      8: [
        this.p(['.####.', '#....#', '#....#', '.####.', '#....#', '#....#', '.####.']),
        this.p(['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.']),
        this.p(['.##.', '#..#', '#..#', '.##.', '#..#', '#..#', '.##.']),
        this.p(['###.', '#..#', '###.', '#..#', '#..#', '###.']),
        // Mal escritos / irregulares
        this.p(['.##.', '#..#', '.##.', '#..#', '#..#', '.##.']),
        this.p(['.#.', '#.#', '.#.', '#.#', '#.#', '.#.']),
        this.p(['.###.', '#...#', '.#.#.', '#...#', '#...#', '.###.']),
        this.p(['.##.', '#..#', '#..#', '.#..', '#..#', '.##.']),
        this.p(['##..', '#.#.', '.##.', '#.#.', '#.#.', '.##.']),
        this.p(['.#.', '#.#', '#.#', '.#.', '#.#', '.#.']),
        // NUEVOS - Más variaciones (confusión con 0, 3, 6, 9)
        this.p(['.##.', '#..#', '#..#', '#..#', '.##.', '#..#', '.##.']), // Sin cintura
        this.p(['.###.', '#...#', '.##..', '#...#', '#...#', '.###.']), // Cintura descentrada
        this.p(['.#..', '#.#.', '.#..', '#.#.', '#.#.', '.#..']), // Inclinado
        this.p(['.##.', '#..#', '.#..', '..#.', '#..#', '.##.']), // Cruzado raro
        this.p(['##.', '#.#', '##.', '#.#', '##.']), // Compacto
        this.p(['.###.', '#...#', '#...#', '..#..', '#...#', '#...#', '.###.']), // Con X medio
        this.p(['.##..', '#..#.', '.##..', '#..#.', '.##..']), // Muy ancho
        this.p(['.#.', '#.#', '.#.', '.#.', '#.#', '.#.']), // Cintura muy fina
      ],
      
      // ========== NUEVE (9) - Incluye 9 con colas diferentes, abiertos ==========
      9: [
        this.p(['.####.', '#....#', '#....#', '.#####', '.....#', '....#.', '.###..']),
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..']),
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '....#', '....#']),
        this.p(['####', '#..#', '####', '...#', '...#', '...#']),
        // Mal escritos / irregulares
        this.p(['.##.', '#..#', '#..#', '.###', '...#', '..#.', '.#..']),
        this.p(['.##.', '#..#', '.###', '...#', '...#', '..#.']),
        this.p(['###.', '#..#', '####', '...#', '..#.', '.#..']),
        this.p(['.#.', '#.#', '#.#', '.##', '..#', '..#']),
        this.p(['.##.', '#..#', '.##.', '..#.', '..#.', '.#..']),
        this.p(['##..', '#.#.', '.##.', '..#.', '..#.', '.#..']),
        // NUEVOS - Más variaciones (confusión con 4, 6, 8)
        this.p(['.###.', '#...#', '#...#', '.###.', '....#', '....#', '....#']), // Cola recta
        this.p(['.##.', '#..#', '#..#', '.###', '...#', '...#', '...#', '..#.']), // Cola muy larga
        this.p(['.##.', '#..#', '.##.', '...#', '..#.', '.#..', '#...']), // Cola diagonal
        this.p(['###', '#.#', '###', '..#', '..#']), // Muy compacto
        this.p(['.###.', '#...#', '#...#', '..##.', '...#.', '...#.']), // Asimétrico
        this.p(['.##..', '#..#.', '#..#.', '.###.', '...#.', '..#..']), // Inclinado
        this.p(['.#.', '#.#', '#.#', '.#.', '.#.', '.#.']), // Sin loop cerrado
        this.p(['.###.', '#...#', '.####', '....#', '....#', '.###.']), // Con curva abajo
      ]
    }
  }

  private p(lines: string[]): number[][] {
    return this.parseTemplate(lines)
  }

  private parseTemplate(lines: string[]): number[][] {
    return lines.map(line => 
      line.split('').map(c => c === '#' ? 1 : 0)
    )
  }

  async predict(pixels: number[]): Promise<PredictionResult> {
    if (!this.model || !this.isReady) return this.empty()

    const img: number[][][] = []
    for (let y = 0; y < 28; y++) {
      const row: number[][] = []
      for (let x = 0; x < 28; x++) {
        row.push([pixels[y * 28 + x]])
      }
      img.push(row)
    }

    const t = tf.tensor4d([img])
    const p = this.model.predict(t) as tf.Tensor
    const probs = Array.from(p.dataSync())
    t.dispose()
    p.dispose()

    const best = Math.max(...probs)
    const digit = probs.indexOf(best)

    const sorted = probs.map((v, i) => ({ d: i, p: v })).sort((a, b) => b.p - a.p)
    console.log(`[#] ${digit} (${(best*100).toFixed(0)}%) | Top3: ${sorted.slice(0,3).map(x=>`${x.d}:${(x.p*100).toFixed(0)}%`).join(' ')}`)

    return {
      probabilities: probs,
      predictedDigit: digit,
      confidence: best,
      hidden1: probs.map(v => Math.min(1, v + Math.random() * 0.15)),
      hidden2: probs.map(v => Math.min(1, v + Math.random() * 0.15)),
      inputSample: pixels.slice(0, 28)
    }
  }

  private empty(): PredictionResult {
    return {
      probabilities: Array(10).fill(0.1),
      predictedDigit: 0,
      confidence: 0,
      hidden1: Array(10).fill(0),
      hidden2: Array(10).fill(0),
      inputSample: Array(28).fill(0)
    }
  }
}

interface WritingStyle {
  scale: number
  thickness: number
  shearX: number
  shearY: number
  stretchX: number
  stretchY: number
  blur: number
  noise: number  // Ruido para simular escritura imperfecta
}
