import * as tf from '@tensorflow/tfjs'
import { PredictionResult } from '../types'

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
      
      this.updateProgress(10, 'Creando modelo CNN...')
      await this.yieldToUI()
      this.model = this.buildModel()
      
      this.updateProgress(15, 'Generando datos...')
      await this.yieldToUI()
      await this.train()

      this.updateProgress(100, '¡Listo!')
      this.isReady = true
      return true
    } catch (e) {
      console.error('❌', e)
      return false
    }
  }

  private buildModel(): tf.Sequential {
    const m = tf.sequential()
    
    // CNN robusta para reconocimiento de dígitos
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

    this.updateProgress(18, 'Generando estilos de escritura...')
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
          const progress = 22 + ((epoch + 1) / totalEpochs) * 73
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
    const samplesPerDigit = 450
    
    for (let d = 0; d < 10; d++) {
      const digitTemplates = templates[d]
      
      for (let i = 0; i < samplesPerDigit; i++) {
        const t = digitTemplates[Math.floor(Math.random() * digitTemplates.length)]
        const style = this.getRandomStyle()
        X.push(this.renderWithStyle(t, style))
        
        const label = new Array(10).fill(0)
        label[d] = 1
        Y.push(label)
      }
      
      this.updateProgress(18 + (d / 10) * 4, `Dígito ${d}...`)
      await this.yieldToUI()
    }

    // Shuffle
    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[X[i], X[j]] = [X[j], X[i]]
      ;[Y[i], Y[j]] = [Y[j], Y[i]]
    }

    return { xs: tf.tensor4d(X), ys: tf.tensor2d(Y) }
  }

  private getRandomStyle(): WritingStyle {
    const styles: WritingStyle[] = [
      { scale: 2.8, thickness: 0.9, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      { scale: 2.0, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      { scale: 3.8, thickness: 0.8, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      { scale: 2.8, thickness: 0.85, shearX: 0.25, shearY: 0, stretchX: 0.9, stretchY: 1, blur: 0 },
      { scale: 2.8, thickness: 0.85, shearX: -0.2, shearY: 0, stretchX: 0.9, stretchY: 1, blur: 0 },
      { scale: 2.5, thickness: 0.9, shearX: 0, shearY: 0, stretchX: 1.3, stretchY: 1, blur: 0 },
      { scale: 2.5, thickness: 0.85, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1.3, blur: 0 },
      { scale: 2.8, thickness: 0.55, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      { scale: 2.8, thickness: 1.2, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.5 },
      { scale: 3.0, thickness: 0.8, shearX: 0, shearY: 0, stretchX: 0.7, stretchY: 1, blur: 0 },
      { scale: 3.0, thickness: 0.85, shearX: 0, shearY: 0, stretchX: 1.4, stretchY: 0.9, blur: 0 },
      { scale: 2.6, thickness: 0.7, shearX: 0.15, shearY: 0.05, stretchX: 1.1, stretchY: 1, blur: 0.3 },
      { scale: 2.5, thickness: 1.1, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.2 },
      { scale: 4.2, thickness: 0.7, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
    ]
    
    const base = styles[Math.floor(Math.random() * styles.length)]
    
    return {
      scale: base.scale + (Math.random() - 0.5) * 0.8,
      thickness: base.thickness + (Math.random() - 0.5) * 0.25,
      shearX: base.shearX + (Math.random() - 0.5) * 0.15,
      shearY: base.shearY + (Math.random() - 0.5) * 0.1,
      stretchX: base.stretchX + (Math.random() - 0.5) * 0.2,
      stretchY: base.stretchY + (Math.random() - 0.5) * 0.2,
      blur: base.blur + Math.random() * 0.2,
    }
  }

  private renderWithStyle(template: number[][], style: WritingStyle): number[][][] {
    const out: number[][][] = []
    const h = template.length
    const w = template[0].length
    
    const offsetX = (Math.random() - 0.5) * 6
    const offsetY = (Math.random() - 0.5) * 6
    const angle = (Math.random() - 0.5) * 0.3
    
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
        
        if (style.blur > 0 && x > 0 && x < 27 && y > 0 && y < 27) {
          const neighbors = (raw[y-1][x] + raw[y+1][x] + raw[y][x-1] + raw[y][x+1]) / 4
          val = val * (1 - style.blur * 0.5) + neighbors * style.blur * 0.5
        }
        
        if (val > 0.2) {
          val = val * (0.8 + Math.random() * 0.3)
        }
        
        if (val < 0.03 && Math.random() < 0.008) {
          val = Math.random() * 0.06
        }
        
        row.push([Math.max(0, Math.min(1, val))])
      }
      out.push(row)
    }
    
    return out
  }

  // ============================================
  // PLANTILLAS COMPLETAS PARA TODOS LOS DÍGITOS
  // ============================================
  private getAllTemplates(): Record<number, number[][][]> {
    return {
      // ========== CERO (0) ==========
      0: [
        // Círculo redondo clásico
        this.p(['..###..', '.#...#.', '#.....#', '#.....#', '#.....#', '.#...#.', '..###..']),
        // Óvalo vertical alto
        this.p(['..##..', '.#..#.', '#....#', '#....#', '#....#', '#....#', '.#..#.', '..##..']),
        // Óvalo horizontal ancho
        this.p(['.#####.', '#.....#', '#.....#', '#.....#', '.#####.']),
        // Cuadrado redondeado
        this.p(['.####.', '#....#', '#....#', '#....#', '#....#', '#....#', '.####.']),
        // Pequeño compacto
        this.p(['.##.', '#..#', '#..#', '#..#', '.##.']),
        // Muy redondo
        this.p(['..##..', '.#..#.', '.#..#.', '.#..#.', '.#..#.', '..##..']),
        // Con barra diagonal (cero europeo)
        this.p(['..###..', '.#...#.', '#....##', '#...#.#', '#..#..#', '.#...#.', '..###..']),
        // Ovalado estrecho
        this.p(['.##.', '#..#', '#..#', '#..#', '#..#', '#..#', '.##.']),
        // Grande y redondo
        this.p(['.####.', '#....#', '#....#', '#....#', '#....#', '.####.']),
        // Estilo manuscrito
        this.p(['..##.', '.#..#', '#...#', '#...#', '#...#', '.#..#', '..##.']),
        // Muy pequeño
        this.p(['##', '##']),
        // Circular perfecto
        this.p(['.###.', '#...#', '#...#', '#...#', '.###.']),
        // Ovalado inclinado
        this.p(['...##.', '..#..#', '.#...#', '#....#', '#...#.', '#..#..', '.##...']),
        // Rectangular
        this.p(['####', '#..#', '#..#', '#..#', '#..#', '####']),
        // Con abertura arriba
        this.p(['.#.#.', '#...#', '#...#', '#...#', '#...#', '.###.']),
      ],

      // ========== UNO (1) ==========
      1: [
        // Con serif y base
        this.p(['...#...', '..##...', '.#.#...', '...#...', '...#...', '...#...', '.#####.']),
        // Simple con base
        this.p(['..#.', '.##.', '..#.', '..#.', '..#.', '..#.', '.###']),
        // Solo línea vertical
        this.p(['#', '#', '#', '#', '#', '#', '#']),
        // Con gancho arriba
        this.p(['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '..#..']),
        // Inclinado
        this.p(['...#', '..##', '...#', '...#', '...#', '...#', '...#']),
        // Con gancho grande
        this.p(['....#.', '...##.', '..#.#.', '.#..#.', '....#.', '....#.', '.#####']),
        // Muy simple
        this.p(['.#.', '.#.', '.#.', '.#.', '.#.', '.#.', '.#.']),
        // Con serif completo
        this.p(['..##.', '.#.#.', '...#.', '...#.', '...#.', '...#.', '.####']),
        // Grueso
        this.p(['..##', '.###', '..##', '..##', '..##', '..##', '.####']),
        // Manuscrito rápido
        this.p(['.#', '##', '.#', '.#', '.#', '.#', '.#']),
        // Con base ancha
        this.p(['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '#####']),
        // Estilo romano
        this.p(['..#..', '.###.', '..#..', '..#..', '..#..', '..#..', '.###.']),
        // Delgado
        this.p(['#', '#', '#', '#', '#', '#', '#', '#']),
        // Con serif izquierdo
        this.p(['##.', '.#.', '.#.', '.#.', '.#.', '.#.', '###']),
      ],

      // ========== DOS (2) ==========
      2: [
        // Clásico con curva
        this.p(['.####.', '#....#', '.....#', '....#.', '..##..', '.#....', '######']),
        // Angular
        this.p(['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####']),
        // Muy curvo
        this.p(['####.', '....#', '...#.', '..#..', '.#...', '#....', '#####']),
        // Pequeño
        this.p(['.##.', '#..#', '...#', '..#.', '.#..', '#...', '####']),
        // Con curva pronunciada
        this.p(['..##..', '.#..#.', '.....#', '....#.', '...#..', '..#...', '.#####']),
        // Estilo Z
        this.p(['#####', '....#', '...#.', '..#..', '.#...', '#....', '#####']),
        // Redondeado arriba
        this.p(['.###.', '#...#', '....#', '..##.', '.#...', '#....', '#####']),
        // Manuscrito rápido
        this.p(['.##.', '...#', '..#.', '.#..', '#...', '#...', '###.']),
        // Con base extendida
        this.p(['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '######']),
        // Curva suave
        this.p(['..##.', '.#..#', '....#', '...#.', '..#..', '.#...', '####.']),
        // Muy angular
        this.p(['###', '..#', '.#.', '#..', '#..', '###']),
        // Con loop
        this.p(['.##.', '#..#', '...#', '..#.', '.#..', '#...', '####']),
        // Estilo digital
        this.p(['###.', '...#', '###.', '#...', '#...', '###.']),
      ],

      // ========== TRES (3) ==========
      3: [
        // Clásico con dos curvas
        this.p(['.####.', '#....#', '.....#', '..###.', '.....#', '#....#', '.####.']),
        // Angular
        this.p(['####.', '....#', '....#', '.###.', '....#', '....#', '####.']),
        // Redondeado
        this.p(['.###.', '#...#', '....#', '..##.', '....#', '#...#', '.###.']),
        // Pequeño
        this.p(['###.', '...#', '...#', '.##.', '...#', '...#', '###.']),
        // Manuscrito
        this.p(['.###', '...#', '..#.', '...#', '...#', '...#', '.###']),
        // Con curvas suaves
        this.p(['..##.', '.#..#', '....#', '..##.', '....#', '.#..#', '..##.']),
        // Estilo bloque
        this.p(['#####', '....#', '..###', '....#', '....#', '....#', '#####']),
        // Abierto
        this.p(['###.', '...#', '.##.', '...#', '...#', '...#', '###.']),
        // Con cintura estrecha
        this.p(['.###.', '#...#', '....#', '...#.', '....#', '#...#', '.###.']),
        // Muy redondeado
        this.p(['..##..', '.#..#.', '....#.', '..##..', '....#.', '.#..#.', '..##..']),
        // Digital
        this.p(['###.', '...#', '###.', '...#', '...#', '###.']),
        // Cursivo
        this.p(['.##.', '...#', '..#.', '...#', '...#', '.##.']),
      ],

      // ========== CUATRO (4) ==========
      4: [
        // Clásico cerrado
        this.p(['....#.', '...##.', '..#.#.', '.#..#.', '######', '....#.', '....#.']),
        // Abierto arriba
        this.p(['#...#', '#...#', '#...#', '#####', '....#', '....#', '....#']),
        // Con diagonal
        this.p(['..#.#', '.#..#', '#...#', '#####', '....#', '....#', '....#']),
        // Pequeño
        this.p(['#..#', '#..#', '#..#', '####', '...#', '...#', '...#']),
        // Estilo europeo
        this.p(['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.']),
        // Angular
        this.p(['#..#', '#..#', '####', '...#', '...#', '...#', '...#']),
        // Con serif
        this.p(['....##', '...#.#', '..#..#', '.#...#', '######', '.....#', '.....#']),
        // Manuscrito
        this.p(['.#.#', '#..#', '#..#', '####', '...#', '...#', '..#.']),
        // Muy abierto
        this.p(['#...#', '#...#', '#####', '....#', '....#', '....#']),
        // Con base
        this.p(['....#.', '...##.', '..#.#.', '.#..#.', '######', '....#.', '...###']),
        // Compacto
        this.p(['#.#', '#.#', '###', '..#', '..#']),
        // Digital
        this.p(['#..#', '#..#', '####', '...#', '...#']),
      ],

      // ========== CINCO (5) ==========
      5: [
        // Clásico
        this.p(['######', '#.....', '#.....', '.####.', '.....#', '#....#', '.####.']),
        // Con curva suave
        this.p(['#####', '#....', '####.', '....#', '....#', '#...#', '.###.']),
        // Angular
        this.p(['#####', '#....', '#....', '####.', '....#', '....#', '####.']),
        // Pequeño
        this.p(['####', '#...', '###.', '...#', '...#', '...#', '###.']),
        // Redondeado
        this.p(['#####', '#....', '#....', '.###.', '....#', '....#', '####.']),
        // Manuscrito
        this.p(['####.', '#....', '###..', '...#.', '...#.', '#..#.', '.##..']),
        // Estilo S invertida
        this.p(['.####', '.#...', '.###.', '....#', '....#', '.#..#', '..##.']),
        // Con base curva
        this.p(['#####', '#....', '####.', '....#', '....#', '...#.', '###..']),
        // Digital
        this.p(['####', '#...', '###.', '...#', '...#', '###.']),
        // Muy angular
        this.p(['#####', '#....', '####.', '....#', '#...#', '.###.']),
        // Con gancho
        this.p(['#####.', '#.....', '#####.', '.....#', '.....#', '#....#', '.####.']),
        // Cursivo
        this.p(['####', '#...', '##..', '..#.', '..#.', '##..']),
      ],

      // ========== SEIS (6) ==========
      6: [
        // Clásico
        this.p(['..###.', '.#....', '#.....', '#####.', '#....#', '#....#', '.####.']),
        // Redondeado
        this.p(['.###.', '#....', '#....', '####.', '#...#', '#...#', '.###.']),
        // Con curva arriba
        this.p(['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.']),
        // Pequeño
        this.p(['.##.', '#...', '#...', '###.', '#..#', '#..#', '.##.']),
        // Muy curvo arriba
        this.p(['..#.', '.#..', '#...', '###.', '#..#', '#..#', '.##.']),
        // Estilo bloque
        this.p(['.####', '#....', '#....', '#####', '#...#', '#...#', '.###.']),
        // Con loop completo
        this.p(['..##..', '.#..#.', '#.....', '#.##..', '#...#.', '.#..#.', '..##..']),
        // Manuscrito
        this.p(['..#.', '.#..', '#...', '##..', '#.#.', '#.#.', '.#..']),
        // Grande
        this.p(['.###.', '#....', '#....', '#....', '####.', '#...#', '.###.']),
        // Angular
        this.p(['.###', '#...', '#...', '####', '#..#', '#..#', '.##.']),
        // Digital
        this.p(['###.', '#...', '###.', '#..#', '#..#', '###.']),
        // Con espiral
        this.p(['..##.', '.#...', '#....', '#.##.', '#...#', '.###.']),
      ],

      // ========== SIETE (7) ==========
      7: [
        // Clásico
        this.p(['######', '.....#', '....#.', '...#..', '..#...', '..#...', '..#...']),
        // Con base
        this.p(['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...']),
        // Con serif arriba
        this.p(['######', '#....#', '....#.', '...#..', '..#...', '..#...', '..#...']),
        // Recto
        this.p(['####', '...#', '...#', '..#.', '..#.', '.#..', '.#..']),
        // Con barra horizontal
        this.p(['#####', '....#', '....#', '...#.', '..#..', '..#..', '.#...']),
        // Europeo con barra
        this.p(['######', '.....#', '....#.', '.####.', '..#...', '..#...', '..#...']),
        // Manuscrito curvo
        this.p(['#####', '....#', '...#.', '...#.', '..#..', '..#..', '.#...']),
        // Muy inclinado
        this.p(['####', '...#', '..#.', '..#.', '.#..', '.#..', '#...']),
        // Digital
        this.p(['###', '..#', '..#', '.#.', '.#.', '#..']),
        // Con gancho arriba
        this.p(['.#####', '....#.', '...#..', '...#..', '..#...', '..#...', '.#....']),
        // Muy recto
        this.p(['####', '...#', '...#', '...#', '..#.', '..#.', '.#..']),
        // Cursivo
        this.p(['####', '...#', '..#.', '.#..', '.#..', '#...']),
      ],

      // ========== OCHO (8) ==========
      8: [
        // Clásico simétrico
        this.p(['.####.', '#....#', '#....#', '.####.', '#....#', '#....#', '.####.']),
        // Redondeado compacto
        this.p(['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.']),
        // Pequeño cuadrado
        this.p(['.##.', '#..#', '#..#', '.##.', '#..#', '#..#', '.##.']),
        // Dos círculos (muñeco de nieve)
        this.p(['..##..', '.#..#.', '.#..#.', '..##..', '.#..#.', '.#..#.', '..##..']),
        // Estilo infinito/lazo
        this.p(['.###.', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '.###.']),
        // Círculo arriba más pequeño
        this.p(['..##..', '.#..#.', '..##..', '.#..#.', '#....#', '#....#', '.####.']),
        // Círculo abajo más pequeño
        this.p(['.####.', '#....#', '#....#', '.####.', '.#..#.', '.#..#.', '..##..']),
        // Muy redondeado continuo
        this.p(['..##..', '.#..#.', '#....#', '.#..#.', '#....#', '.#..#.', '..##..']),
        // Angular
        this.p(['####.', '#...#', '#...#', '.###.', '#...#', '#...#', '####.']),
        // Cintura muy estrecha
        this.p(['.###.', '#...#', '#...#', '..#..', '#...#', '#...#', '.###.']),
        // Digital
        this.p(['###.', '#..#', '###.', '#..#', '#..#', '###.']),
        // Manuscrito
        this.p(['.##.', '#..#', '.#..', '..#.', '.#..', '#..#', '.##.']),
        // Muy pequeño
        this.p(['.#.', '#.#', '.#.', '#.#', '.#.']),
        // Con centro marcado
        this.p(['.###.', '#...#', '#.#.#', '.###.', '#.#.#', '#...#', '.###.']),
        // Asimétrico
        this.p(['.##.', '#..#', '#..#', '.###', '#..#', '#..#', '.##.']),
      ],

      // ========== NUEVE (9) ==========
      9: [
        // Clásico con cola curva
        this.p(['.####.', '#....#', '#....#', '.#####', '.....#', '....#.', '.###..']),
        // Redondeado con cola corta
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..']),
        // Con cola recta vertical
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '....#', '....#']),
        // Pequeño compacto
        this.p(['.##.', '#..#', '#..#', '.###', '...#', '...#', '..#.']),
        // Cola muy larga
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '....#', '....#', '....#']),
        // Estilo q cursivo
        this.p(['.##.', '#..#', '#..#', '.###', '...#', '..#.', '.#..']),
        // Círculo arriba con línea
        this.p(['.##.', '#..#', '#..#', '.##.', '..#.', '..#.', '..#.']),
        // Angular cerrado
        this.p(['####.', '#...#', '#...#', '.####', '....#', '....#', '###..']),
        // Muy redondeado
        this.p(['..##..', '.#..#.', '#....#', '.#..##', '.....#', '....#.', '..##..']),
        // Con gancho abajo
        this.p(['.###.', '#...#', '#...#', '.####', '....#', '...#.', '..#..', '.#...']),
        // Digital
        this.p(['####', '#..#', '####', '...#', '...#', '...#']),
        // Círculo pequeño arriba
        this.p(['..#..', '.#.#.', '.#.#.', '..##.', '...#.', '...#.', '...#.']),
        // Ovalado con cola
        this.p(['.##.', '#..#', '#..#', '#..#', '.###', '...#', '...#']),
        // Muy cursivo
        this.p(['.##.', '#..#', '#.#.', '.#.#', '...#', '..#.', '.#..']),
        // Con círculo grande
        this.p(['.####.', '#....#', '#....#', '#....#', '.#####', '.....#', '.....#']),
      ]
    }
  }

  // Alias corto para parseTemplate
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
}
