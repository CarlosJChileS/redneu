import * as tf from '@tensorflow/tfjs'
import { PredictionResult } from '../types'

const MODEL_STORAGE_KEY = 'indexeddb://digit-recognition-model'
const MODEL_VERSION_KEY = 'digit-model-version'
const CURRENT_MODEL_VERSION = '2.0.0' // Versión mejorada para escritura irregular

// URL del servidor API
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000'

export class NeuralNetwork {
  private model: tf.Sequential | null = null
  public isReady = false
  public onProgress: ((progress: number, status: string) => void) | null = null

  private updateProgress(progress: number, status: string) {
    if (this.onProgress) {
      this.onProgress(progress, status)
    }
  }

  private async yieldToUI(): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, 0))
  }

  async initialize(): Promise<boolean> {
    try {
      this.updateProgress(5, 'Iniciando TensorFlow...')
      await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'))
      await tf.ready()
      
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

      this.updateProgress(100, '¡Listo!')
      this.isReady = true
      return true
    } catch (e) {
      console.error('❌', e)
      return false
    }
  }

  // Cargar modelo desde el servidor (MongoDB)
  private async tryLoadFromServer(): Promise<boolean> {
    try {
      const response = await fetch(`${API_URL}/api/model`)
      
      if (!response.ok) {
        console.log('📦 No hay modelo en el servidor')
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
      console.log(`✅ Modelo cargado desde servidor (v${data.model.version}, acc: ${(data.model.accuracy * 100).toFixed(1)}%)`)
      
      return true
    } catch (error) {
      console.log('📦 Error cargando del servidor:', error)
      return false
    }
  }

  // Subir modelo al servidor (MongoDB)
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
      
      // Convertir a base64
      const float32Array = new Float32Array(weightsData)
      const uint8Array = new Uint8Array(float32Array.buffer)
      const weightsBase64 = this.arrayBufferToBase64(uint8Array.buffer)

      // Enviar al servidor
      const response = await fetch(`${API_URL}/api/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelJson: modelJson,
          weightsBase64: weightsBase64,
          version: CURRENT_MODEL_VERSION,
          accuracy: 0.95 // Aproximado
        })
      })

      if (response.ok) {
        console.log('✅ Modelo subido al servidor')
        return true
      }
      return false
    } catch (error) {
      console.log('⚠️ No se pudo subir al servidor:', error)
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

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer)
    let binary = ''
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i])
    }
    return btoa(binary)
  }

  private async tryLoadSavedModel(): Promise<boolean> {
    try {
      const savedVersion = localStorage.getItem(MODEL_VERSION_KEY)
      if (savedVersion !== CURRENT_MODEL_VERSION) {
        console.log('📦 Versión de modelo diferente, re-entrenando...')
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
        console.log('✅ Modelo cargado desde IndexedDB')
        return true
      }
      return false
    } catch (error) {
      console.log('📦 No hay modelo local guardado')
      return false
    }
  }

  private async saveModel(): Promise<void> {
    if (!this.model) return
    
    try {
      await this.model.save(MODEL_STORAGE_KEY)
      localStorage.setItem(MODEL_VERSION_KEY, CURRENT_MODEL_VERSION)
      console.log('💾 Modelo guardado en IndexedDB')
    } catch (error) {
      console.error('Error guardando modelo:', error)
    }
  }

  async retrainModel(): Promise<boolean> {
    try {
      // Eliminar modelo local
      try {
        await tf.io.removeModel(MODEL_STORAGE_KEY)
        localStorage.removeItem(MODEL_VERSION_KEY)
      } catch {
        // Ignorar
      }

      // Eliminar modelo del servidor
      try {
        await fetch(`${API_URL}/api/model`, { method: 'DELETE' })
      } catch {
        // Ignorar si el servidor no está disponible
      }

      this.model = null
      this.isReady = false

      return await this.initialize()
    } catch (error) {
      console.error('Error re-entrenando:', error)
      return false
    }
  }

  private buildModel(): tf.Sequential {
    const m = tf.sequential()
    
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
    m.add(tf.layers.dropout({ rate: 0.5 }))
    m.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

    m.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    })
    
    return m
  }

  private async train(): Promise<void> {
    if (!this.model) return

    this.updateProgress(22, 'Generando estilos de escritura...')
    await this.yieldToUI()
    
    const { xs, ys } = await this.createDatasetAsync()
    
    const totalEpochs = 20
    
    await this.model.fit(xs, ys, {
      epochs: totalEpochs,
      batchSize: 64,
      shuffle: true,
      validationSplit: 0.15,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress = 25 + ((epoch + 1) / totalEpochs) * 65
          const acc = ((logs?.acc || 0) * 100).toFixed(0)
          this.updateProgress(progress, `Época ${epoch + 1}/${totalEpochs} (${acc}%)`)
          await this.yieldToUI()
        }
      }
    })

    xs.dispose()
    ys.dispose()
  }

  private async createDatasetAsync(): Promise<{ xs: tf.Tensor4D; ys: tf.Tensor2D }> {
    const X: number[][][][] = []
    const Y: number[][] = []
    
    const templates = this.getAllTemplates()
    const samplesPerDigit = 600 // Más muestras para mejor generalización
    
    for (let d = 0; d < 10; d++) {
      const digitTemplates = templates[d]
      
      for (let i = 0; i < samplesPerDigit; i++) {
        const t = digitTemplates[Math.floor(Math.random() * digitTemplates.length)]
        // Alternar entre estilos normales y "messy" (desordenados)
        const style = i % 3 === 0 ? this.getMessyStyle() : this.getRandomStyle()
        X.push(this.renderWithStyle(t, style))
        
        const label = new Array(10).fill(0)
        label[d] = 1
        Y.push(label)
      }
      
      this.updateProgress(22 + (d / 10) * 3, `Dígito ${d}...`)
      await this.yieldToUI()
    }

    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[X[i], X[j]] = [X[j], X[i]]
      ;[Y[i], Y[j]] = [Y[j], Y[i]]
    }

    return { xs: tf.tensor4d(X), ys: tf.tensor2d(Y) }
  }

  private getRandomStyle(): WritingStyle {
    const styles: WritingStyle[] = [
      { scale: 2.8, thickness: 0.9, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0, noise: 0 },
      { scale: 2.0, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0, noise: 0 },
      { scale: 3.8, thickness: 0.8, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0, noise: 0 },
      { scale: 2.8, thickness: 0.85, shearX: 0.25, shearY: 0, stretchX: 0.9, stretchY: 1, blur: 0, noise: 0 },
      { scale: 2.8, thickness: 0.85, shearX: -0.2, shearY: 0, stretchX: 0.9, stretchY: 1, blur: 0, noise: 0 },
      { scale: 2.5, thickness: 0.9, shearX: 0, shearY: 0, stretchX: 1.3, stretchY: 1, blur: 0, noise: 0 },
      { scale: 2.5, thickness: 0.85, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1.3, blur: 0, noise: 0 },
      { scale: 2.8, thickness: 0.55, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0, noise: 0 },
      { scale: 2.8, thickness: 1.2, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.5, noise: 0 },
      { scale: 3.0, thickness: 0.8, shearX: 0, shearY: 0, stretchX: 0.7, stretchY: 1, blur: 0, noise: 0 },
      { scale: 3.0, thickness: 0.85, shearX: 0, shearY: 0, stretchX: 1.4, stretchY: 0.9, blur: 0, noise: 0 },
      { scale: 2.6, thickness: 0.7, shearX: 0.15, shearY: 0.05, stretchX: 1.1, stretchY: 1, blur: 0.3, noise: 0 },
      { scale: 2.5, thickness: 1.1, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.2, noise: 0 },
      { scale: 4.2, thickness: 0.7, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0, noise: 0 },
    ]
    
    const base = styles[Math.floor(Math.random() * styles.length)]
    
    return {
      scale: base.scale + (Math.random() - 0.5) * 0.8,
      thickness: base.thickness + (Math.random() - 0.5) * 0.25,
      shearX: base.shearX + (Math.random() - 0.5) * 0.2,
      shearY: base.shearY + (Math.random() - 0.5) * 0.15,
      stretchX: base.stretchX + (Math.random() - 0.5) * 0.3,
      stretchY: base.stretchY + (Math.random() - 0.5) * 0.3,
      blur: base.blur + Math.random() * 0.3,
      noise: Math.random() * 0.1,
    }
  }

  // Estilo "desordenado" para escritura mal hecha
  private getMessyStyle(): WritingStyle {
    return {
      scale: 2.0 + Math.random() * 2.5,  // Tamaños muy variados
      thickness: 0.4 + Math.random() * 0.9,  // Grosor muy variable
      shearX: (Math.random() - 0.5) * 0.5,  // Mucha inclinación
      shearY: (Math.random() - 0.5) * 0.3,
      stretchX: 0.6 + Math.random() * 0.9,  // Muy estirado/comprimido
      stretchY: 0.6 + Math.random() * 0.9,
      blur: Math.random() * 0.6,  // Más borroso
      noise: 0.05 + Math.random() * 0.2,  // Ruido para imperfecciones
    }
  }

  private renderWithStyle(template: number[][], style: WritingStyle): number[][][] {
    const out: number[][][] = []
    const h = template.length
    const w = template[0].length
    
    // Más variación en posición
    const offsetX = (Math.random() - 0.5) * 10
    const offsetY = (Math.random() - 0.5) * 10
    const angle = (Math.random() - 0.5) * 0.5  // Más rotación
    
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
        this.p(['..##.', '.#..#', '#...#', '#...#', '.#..#', '..##.']), // Asimétrico
        this.p(['.###.', '#....', '#...#', '#...#', '.###.']), // Abierto arriba
        this.p(['.##..', '#..#.', '#...#', '#..#.', '.##..']), // Inclinado
        this.p(['..#..', '.#.#.', '#...#', '#...#', '.#.#.', '..#..']), // Ovalado raro
        this.p(['.###.', '#...#', '#....', '#...#', '.###.']), // Gap en medio
        this.p(['####', '#..#', '#..#', '####']), // Cuadrado
        this.p(['.#.#.', '#...#', '#...#', '#...#', '.#.#.']), // Abierto arriba/abajo
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
        this.p(['..#', '.#.', '.#.', '.#.', '#..', '#..']), // Muy inclinado
        this.p(['.#', '.#', '#.', '#.', '#.', '#.']), // Inclinado izquierda
        this.p(['#.', '.#', '.#', '.#', '.#', '#.']), // Curvo
        this.p(['##', '##', '.#', '.#', '.#', '##']), // Grueso arriba
        this.p(['.#.', '##.', '.#.', '.#.', '.##', '.#.']), // Irregular
        this.p(['...#', '..#.', '..#.', '.#..', '.#..', '#...']), // Diagonal
      ],
      
      // ========== DOS (2) - Incluye Z, curvas abiertas, angulares ==========
      2: [
        this.p(['.####.', '#....#', '.....#', '....#.', '..##..', '.#....', '######']),
        this.p(['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####']),
        this.p(['####.', '....#', '...#.', '..#..', '.#...', '#....', '#####']),
        this.p(['.##.', '#..#', '...#', '..#.', '.#..', '#...', '####']),
        this.p(['###', '..#', '.#.', '#..', '#..', '###']),
        // Mal escritos / irregulares
        this.p(['###.', '...#', '..#.', '.#..', '#...', '###.']), // Sin curva arriba
        this.p(['..##', '....#', '...#.', '..#..', '.#...', '####']), // Pequeño arriba
        this.p(['.##.', '...#', '..#.', '.#..', '#...', '##..']), // Sin base completa
        this.p(['###', '..#', '.#.', '.#.', '#..', '###']), // Recto
        this.p(['.#.', '#.#', '..#', '.#.', '#..', '###']), // Curva rara
        this.p(['##..', '..#.', '..#.', '.#..', '#...', '####']), // Angular
      ],
      
      // ========== TRES (3) - Incluye 3 abiertos, con curvas irregulares ==========
      3: [
        this.p(['.####.', '#....#', '.....#', '..###.', '.....#', '#....#', '.####.']),
        this.p(['####.', '....#', '....#', '.###.', '....#', '....#', '####.']),
        this.p(['.###.', '#...#', '....#', '..##.', '....#', '#...#', '.###.']),
        this.p(['###.', '...#', '...#', '.##.', '...#', '...#', '###.']),
        this.p(['###.', '...#', '###.', '...#', '...#', '###.']),
        // Mal escritos / irregulares
        this.p(['##..', '..#.', '..#.', '.#..', '..#.', '..#.', '##..']), // Muy curvo
        this.p(['###', '..#', '.#.', '..#', '..#', '###']), // Angular
        this.p(['.##.', '...#', '..#.', '...#', '...#', '.##.']), // Sin curva superior
        this.p(['###.', '...#', '.##.', '...#', '..#.', '.#..']), // Abierto abajo
        this.p(['.#..', '..#.', '.#..', '..#.', '..#.', '.#..']), // Muy ondulado
      ],
      
      // ========== CUATRO (4) - Incluye abiertos, cerrados, angulares ==========
      4: [
        this.p(['....#.', '...##.', '..#.#.', '.#..#.', '######', '....#.', '....#.']),
        this.p(['#...#', '#...#', '#...#', '#####', '....#', '....#', '....#']),
        this.p(['#..#', '#..#', '#..#', '####', '...#', '...#', '...#']),
        this.p(['#...#', '#...#', '#####', '....#', '....#', '....#']),
        this.p(['#.#', '#.#', '###', '..#', '..#']),
        // Mal escritos / irregulares
        this.p(['#..#', '#..#', '####', '...#', '...#']), // Corto
        this.p(['..#.', '.##.', '#.#.', '####', '..#.', '..#.']), // Con serif
        this.p(['#...#', '#..#.', '.###.', '...#.', '...#.']), // Cruzado diferente
        this.p(['.#.#', '#..#', '####', '...#', '...#', '..#.']), // Abierto
        this.p(['#..', '#.#', '###', '..#', '..#', '..#']), // Muy angular
      ],
      
      // ========== CINCO (5) - Incluye S invertidas, angulares ==========
      5: [
        this.p(['######', '#.....', '#.....', '.####.', '.....#', '#....#', '.####.']),
        this.p(['#####', '#....', '####.', '....#', '....#', '#...#', '.###.']),
        this.p(['#####', '#....', '#....', '####.', '....#', '....#', '####.']),
        this.p(['####', '#...', '###.', '...#', '...#', '###.']),
        // Mal escritos / irregulares
        this.p(['####', '#...', '##..', '..#.', '..#.', '##..']), // Curvo
        this.p(['###.', '#...', '###.', '...#', '..#.', '.#..']), // Abierto abajo
        this.p(['####', '#...', '#...', '###.', '...#', '###.']), // Más recto
        this.p(['.###', '.#..', '.##.', '...#', '...#', '.##.']), // Sin esquina
        this.p(['###', '#..', '##.', '..#', '..#', '#..']), // Muy curvo
      ],
      
      // ========== SEIS (6) - Incluye 6 abiertos, con loops irregulares ==========
      6: [
        this.p(['..###.', '.#....', '#.....', '#####.', '#....#', '#....#', '.####.']),
        this.p(['.###.', '#....', '#....', '####.', '#...#', '#...#', '.###.']),
        this.p(['.##.', '#...', '#...', '###.', '#..#', '#..#', '.##.']),
        this.p(['###.', '#...', '###.', '#..#', '#..#', '###.']),
        // Mal escritos / irregulares
        this.p(['..#.', '.#..', '#...', '###.', '#..#', '.##.']), // Muy curvo arriba
        this.p(['.##.', '#...', '##..', '#.#.', '#.#.', '.#..']), // Pequeño abajo
        this.p(['.#..', '#...', '###.', '#..#', '#..#', '.##.']), // Sin curva
        this.p(['..##', '.#..', '#...', '##..', '#.#.', '.#..']), // Irregular
        this.p(['.#.', '#..', '#..', '##.', '#.#', '.#.']), // Muy pequeño
      ],
      
      // ========== SIETE (7) - Incluye con/sin barra, muy inclinados ==========
      7: [
        this.p(['######', '.....#', '....#.', '...#..', '..#...', '..#...', '..#...']),
        this.p(['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...']),
        this.p(['####', '...#', '...#', '..#.', '..#.', '.#..', '.#..']),
        this.p(['###', '..#', '..#', '.#.', '.#.', '#..']),
        // Mal escritos / irregulares
        this.p(['####', '...#', '..#.', '..#.', '.#..', '.#..']), // Más vertical
        this.p(['###.', '..#.', '..#.', '.#..', '.#..', '#...']), // Muy inclinado
        this.p(['####', '...#', '..#.', '.#..', '#...', '#...']), // Diagonal fuerte
        this.p(['##', '.#', '.#', '#.', '#.']), // Muy pequeño
        this.p(['###', '..#', '.#.', '.#.', '#..', '#..']), // Curvo
        this.p(['####', '..#.', '..#.', '..#.', '.#..', '.#..']), // Casi vertical
      ],
      
      // ========== OCHO (8) - Incluye 8 desproporcionados, abiertos ==========
      8: [
        this.p(['.####.', '#....#', '#....#', '.####.', '#....#', '#....#', '.####.']),
        this.p(['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.']),
        this.p(['.##.', '#..#', '#..#', '.##.', '#..#', '#..#', '.##.']),
        this.p(['###.', '#..#', '###.', '#..#', '#..#', '###.']),
        // Mal escritos / irregulares
        this.p(['.##.', '#..#', '.##.', '#..#', '#..#', '.##.']), // Cintura estrecha
        this.p(['.#.', '#.#', '.#.', '#.#', '#.#', '.#.']), // Muy estrecho
        this.p(['.###.', '#...#', '.#.#.', '#...#', '#...#', '.###.']), // Cruz en medio
        this.p(['.##.', '#..#', '#..#', '.#..', '#..#', '.##.']), // Abierto
        this.p(['##..', '#.#.', '.##.', '#.#.', '#.#.', '.##.']), // Asimétrico
        this.p(['.#.', '#.#', '#.#', '.#.', '#.#', '.#.']), // Muy pequeño
      ],
      
      // ========== NUEVE (9) - Incluye 9 con colas diferentes, abiertos ==========
      9: [
        this.p(['.####.', '#....#', '#....#', '.#####', '.....#', '....#.', '.###..']),
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..']),
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '....#', '....#']),
        this.p(['####', '#..#', '####', '...#', '...#', '...#']),
        // Mal escritos / irregulares
        this.p(['.##.', '#..#', '#..#', '.###', '...#', '..#.', '.#..']), // Cola curva
        this.p(['.##.', '#..#', '.###', '...#', '...#', '..#.']), // Pequeño arriba
        this.p(['###.', '#..#', '####', '...#', '..#.', '.#..']), // Angular
        this.p(['.#.', '#.#', '#.#', '.##', '..#', '..#']), // Muy pequeño
        this.p(['.##.', '#..#', '.##.', '..#.', '..#.', '.#..']), // Sin conexión
        this.p(['##..', '#.#.', '.##.', '..#.', '..#.', '.#..']), // Asimétrico
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
    console.log(`🔢 ${digit} (${(best*100).toFixed(0)}%) | Top3: ${sorted.slice(0,3).map(x=>`${x.d}:${(x.p*100).toFixed(0)}%`).join(' ')}`)

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
