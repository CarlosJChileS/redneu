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
    
    // CNN más robusta para diferentes estilos de escritura
    m.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    m.add(tf.layers.batchNormalization())
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }))
    
    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    m.add(tf.layers.batchNormalization())
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }))
    
    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    }))
    
    m.add(tf.layers.flatten())
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }))
    m.add(tf.layers.dropout({ rate: 0.4 }))
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
    
    const totalEpochs = 18
    
    await this.model.fit(xs, ys, {
      epochs: totalEpochs,
      batchSize: 64,
      shuffle: true,
      validationSplit: 0.12,
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
    const baseSamples = 350
    
    for (let d = 0; d < 10; d++) {
      const digitTemplates = templates[d]
      // Más muestras para dígitos difíciles (8 y 9)
      const samplesPerDigit = (d === 8 || d === 9) ? baseSamples + 150 : baseSamples
      
      for (let i = 0; i < samplesPerDigit; i++) {
        const t = digitTemplates[Math.floor(Math.random() * digitTemplates.length)]
        // Aplicar estilo aleatorio de escritura
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

  // Diferentes estilos de escritura
  private getRandomStyle(): WritingStyle {
    const styles: WritingStyle[] = [
      // Escritura normal
      { scale: 2.8, thickness: 0.9, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      // Letra grande
      { scale: 2.0, thickness: 1.0, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      // Letra pequeña
      { scale: 3.8, thickness: 0.8, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      // Cursiva derecha
      { scale: 2.8, thickness: 0.85, shearX: 0.25, shearY: 0, stretchX: 0.9, stretchY: 1, blur: 0 },
      // Cursiva izquierda
      { scale: 2.8, thickness: 0.85, shearX: -0.2, shearY: 0, stretchX: 0.9, stretchY: 1, blur: 0 },
      // Letra ancha
      { scale: 2.5, thickness: 0.9, shearX: 0, shearY: 0, stretchX: 1.3, stretchY: 1, blur: 0 },
      // Letra alta
      { scale: 2.5, thickness: 0.85, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1.3, blur: 0 },
      // Letra fina
      { scale: 2.8, thickness: 0.55, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
      // Letra gruesa
      { scale: 2.8, thickness: 1.2, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.5 },
      // Letra comprimida
      { scale: 3.0, thickness: 0.8, shearX: 0, shearY: 0, stretchX: 0.7, stretchY: 1, blur: 0 },
      // Letra expandida horizontal
      { scale: 3.0, thickness: 0.85, shearX: 0, shearY: 0, stretchX: 1.4, stretchY: 0.9, blur: 0 },
      // Estilo manuscrito rápido
      { scale: 2.6, thickness: 0.7, shearX: 0.15, shearY: 0.05, stretchX: 1.1, stretchY: 1, blur: 0.3 },
      // Estilo bloque
      { scale: 2.5, thickness: 1.1, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0.2 },
      // Micro escritura
      { scale: 4.2, thickness: 0.7, shearX: 0, shearY: 0, stretchX: 1, stretchY: 1, blur: 0 },
    ]
    
    const base = styles[Math.floor(Math.random() * styles.length)]
    
    // Añadir variación aleatoria al estilo base
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

    // Primera pasada: renderizar
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
    
    // Segunda pasada: aplicar blur si es necesario y añadir variación
    for (let y = 0; y < 28; y++) {
      const row: number[][] = []
      for (let x = 0; x < 28; x++) {
        let val = raw[y][x]
        
        // Blur simple (promedio con vecinos)
        if (style.blur > 0 && x > 0 && x < 27 && y > 0 && y < 27) {
          const neighbors = (raw[y-1][x] + raw[y+1][x] + raw[y][x-1] + raw[y][x+1]) / 4
          val = val * (1 - style.blur * 0.5) + neighbors * style.blur * 0.5
        }
        
        // Variación natural del trazo
        if (val > 0.2) {
          val = val * (0.8 + Math.random() * 0.3)
        }
        
        // Ruido mínimo de fondo
        if (val < 0.03 && Math.random() < 0.008) {
          val = Math.random() * 0.06
        }
        
        row.push([Math.max(0, Math.min(1, val))])
      }
      out.push(row)
    }
    
    return out
  }

  // Plantillas con múltiples estilos de escritura para cada dígito
  private getAllTemplates(): Record<number, number[][][]> {
    return {
      0: [
        // Estilo redondo clásico
        this.parseTemplate([
          '..###..',
          '.#...#.',
          '#.....#',
          '#.....#',
          '#.....#',
          '.#...#.',
          '..###..'
        ]),
        // Estilo ovalado vertical
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '#....#',
          '#....#',
          '#....#',
          '#....#',
          '.#..#.',
          '..##..'
        ]),
        // Estilo ovalado horizontal
        this.parseTemplate([
          '.####.',
          '#....#',
          '#....#',
          '#....#',
          '.####.'
        ]),
        // Estilo cuadrado
        this.parseTemplate([
          '.####.',
          '#....#',
          '#....#',
          '#....#',
          '#....#',
          '#....#',
          '.####.'
        ]),
        // Estilo pequeño
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '#..#',
          '.##.'
        ]),
        // Con línea diagonal (cero europeo)
        this.parseTemplate([
          '..###..',
          '.#...#.',
          '#....##',
          '#...#.#',
          '#..#..#',
          '.#...#.',
          '..###..'
        ]),
        // Muy redondo
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '.#..#.',
          '.#..#.',
          '.#..#.',
          '..##..'
        ]),
        // Angular
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '#...#',
          '#...#',
          '.###.'
        ])
      ],
      1: [
        // Con base y serif
        this.parseTemplate([
          '...#...',
          '..##...',
          '.#.#...',
          '...#...',
          '...#...',
          '...#...',
          '.#####.'
        ]),
        // Simple con base
        this.parseTemplate([
          '..#.',
          '.##.',
          '..#.',
          '..#.',
          '..#.',
          '..#.',
          '.###'
        ]),
        // Solo línea vertical
        this.parseTemplate([
          '.#.',
          '.#.',
          '.#.',
          '.#.',
          '.#.',
          '.#.',
          '.#.'
        ]),
        // Con serif arriba
        this.parseTemplate([
          '..#..',
          '.##..',
          '..#..',
          '..#..',
          '..#..',
          '..#..',
          '..#..'
        ]),
        // Estilo manuscrito inclinado
        this.parseTemplate([
          '...#',
          '..##',
          '...#',
          '...#',
          '...#',
          '...#',
          '...#'
        ]),
        // Con gancho grande
        this.parseTemplate([
          '....#.',
          '...##.',
          '..#.#.',
          '.#..#.',
          '....#.',
          '....#.',
          '.#####'
        ]),
        // Muy simple
        this.parseTemplate([
          '#',
          '#',
          '#',
          '#',
          '#',
          '#',
          '#'
        ]),
        // Con serif completo
        this.parseTemplate([
          '..##.',
          '.#.#.',
          '...#.',
          '...#.',
          '...#.',
          '...#.',
          '.####'
        ])
      ],
      2: [
        // Clásico con curva
        this.parseTemplate([
          '.####.',
          '#....#',
          '.....#',
          '....#.',
          '..##..',
          '.#....',
          '######'
        ]),
        // Más angular
        this.parseTemplate([
          '.###.',
          '#...#',
          '....#',
          '...#.',
          '..#..',
          '.#...',
          '#####'
        ]),
        // Muy curvo
        this.parseTemplate([
          '####.',
          '....#',
          '...#.',
          '..#..',
          '.#...',
          '#....',
          '#####'
        ]),
        // Pequeño
        this.parseTemplate([
          '.##.',
          '#..#',
          '...#',
          '..#.',
          '.#..',
          '#...',
          '####'
        ]),
        // Con curva pronunciada arriba
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '.....#',
          '....#.',
          '...#..',
          '..#...',
          '.#####'
        ]),
        // Estilo Z
        this.parseTemplate([
          '#####',
          '....#',
          '...#.',
          '..#..',
          '.#...',
          '#....',
          '#####'
        ]),
        // Redondeado
        this.parseTemplate([
          '.###.',
          '#...#',
          '....#',
          '..##.',
          '.#...',
          '#....',
          '#####'
        ]),
        // Manuscrito rápido
        this.parseTemplate([
          '.##.',
          '...#',
          '..#.',
          '.#..',
          '#...',
          '#...',
          '###.'
        ])
      ],
      3: [
        // Clásico con dos curvas
        this.parseTemplate([
          '.####.',
          '#....#',
          '.....#',
          '..###.',
          '.....#',
          '#....#',
          '.####.'
        ]),
        // Angular
        this.parseTemplate([
          '####.',
          '....#',
          '....#',
          '.###.',
          '....#',
          '....#',
          '####.'
        ]),
        // Redondeado
        this.parseTemplate([
          '.###.',
          '#...#',
          '....#',
          '..##.',
          '....#',
          '#...#',
          '.###.'
        ]),
        // Pequeño
        this.parseTemplate([
          '###.',
          '...#',
          '...#',
          '.##.',
          '...#',
          '...#',
          '###.'
        ]),
        // Manuscrito
        this.parseTemplate([
          '.###',
          '...#',
          '..#.',
          '...#',
          '...#',
          '...#',
          '.###'
        ]),
        // Con curvas suaves
        this.parseTemplate([
          '..##.',
          '.#..#',
          '....#',
          '..##.',
          '....#',
          '.#..#',
          '..##.'
        ]),
        // Estilo bloque
        this.parseTemplate([
          '#####',
          '....#',
          '..###',
          '....#',
          '....#',
          '....#',
          '#####'
        ]),
        // Abierto arriba
        this.parseTemplate([
          '###.',
          '...#',
          '.##.',
          '...#',
          '...#',
          '...#',
          '###.'
        ])
      ],
      4: [
        // Clásico cerrado
        this.parseTemplate([
          '....#.',
          '...##.',
          '..#.#.',
          '.#..#.',
          '######',
          '....#.',
          '....#.'
        ]),
        // Abierto arriba
        this.parseTemplate([
          '#...#',
          '#...#',
          '#...#',
          '#####',
          '....#',
          '....#',
          '....#'
        ]),
        // Con diagonal
        this.parseTemplate([
          '..#.#',
          '.#..#',
          '#...#',
          '#####',
          '....#',
          '....#',
          '....#'
        ]),
        // Pequeño
        this.parseTemplate([
          '#..#',
          '#..#',
          '#..#',
          '####',
          '...#',
          '...#',
          '...#'
        ]),
        // Estilo europeo
        this.parseTemplate([
          '...#.',
          '..##.',
          '.#.#.',
          '#..#.',
          '#####',
          '...#.',
          '...#.'
        ]),
        // Muy angular
        this.parseTemplate([
          '#..#',
          '#..#',
          '####',
          '...#',
          '...#',
          '...#',
          '...#'
        ]),
        // Con serif
        this.parseTemplate([
          '....##',
          '...#.#',
          '..#..#',
          '.#...#',
          '######',
          '.....#',
          '.....#'
        ]),
        // Manuscrito
        this.parseTemplate([
          '.#.#',
          '#..#',
          '#..#',
          '####',
          '...#',
          '...#',
          '..#.'
        ])
      ],
      5: [
        // Clásico
        this.parseTemplate([
          '######',
          '#.....',
          '#.....',
          '.####.',
          '.....#',
          '#....#',
          '.####.'
        ]),
        // Con curva suave
        this.parseTemplate([
          '#####',
          '#....',
          '####.',
          '....#',
          '....#',
          '#...#',
          '.###.'
        ]),
        // Angular
        this.parseTemplate([
          '#####',
          '#....',
          '#....',
          '####.',
          '....#',
          '....#',
          '####.'
        ]),
        // Pequeño
        this.parseTemplate([
          '####',
          '#...',
          '###.',
          '...#',
          '...#',
          '...#',
          '###.'
        ]),
        // Redondeado
        this.parseTemplate([
          '#####',
          '#....',
          '#....',
          '.###.',
          '....#',
          '....#',
          '####.'
        ]),
        // Manuscrito
        this.parseTemplate([
          '####.',
          '#....',
          '###..',
          '...#.',
          '...#.',
          '#..#.',
          '.##..'
        ]),
        // Estilo S invertida
        this.parseTemplate([
          '.####',
          '.#...',
          '.###.',
          '....#',
          '....#',
          '.#..#',
          '..##.'
        ]),
        // Con base curva
        this.parseTemplate([
          '#####',
          '#....',
          '####.',
          '....#',
          '....#',
          '...#.',
          '###..'
        ])
      ],
      6: [
        // Clásico
        this.parseTemplate([
          '..###.',
          '.#....',
          '#.....',
          '#####.',
          '#....#',
          '#....#',
          '.####.'
        ]),
        // Redondeado
        this.parseTemplate([
          '.###.',
          '#....',
          '#....',
          '####.',
          '#...#',
          '#...#',
          '.###.'
        ]),
        // Con curva arriba
        this.parseTemplate([
          '..##.',
          '.#...',
          '#....',
          '####.',
          '#...#',
          '#...#',
          '.###.'
        ]),
        // Pequeño
        this.parseTemplate([
          '.##.',
          '#...',
          '#...',
          '###.',
          '#..#',
          '#..#',
          '.##.'
        ]),
        // Muy curvo arriba
        this.parseTemplate([
          '..#.',
          '.#..',
          '#...',
          '###.',
          '#..#',
          '#..#',
          '.##.'
        ]),
        // Estilo bloque
        this.parseTemplate([
          '.####',
          '#....',
          '#....',
          '#####',
          '#...#',
          '#...#',
          '.###.'
        ]),
        // Con loop completo
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '#.....',
          '#.##..',
          '#...#.',
          '.#..#.',
          '..##..'
        ]),
        // Manuscrito
        this.parseTemplate([
          '..#.',
          '.#..',
          '#...',
          '##..',
          '#.#.',
          '#.#.',
          '.#..'
        ])
      ],
      7: [
        // Clásico
        this.parseTemplate([
          '######',
          '.....#',
          '....#.',
          '...#..',
          '..#...',
          '..#...',
          '..#...'
        ]),
        // Con base
        this.parseTemplate([
          '#####',
          '....#',
          '...#.',
          '..#..',
          '.#...',
          '.#...',
          '.#...'
        ]),
        // Con serif arriba
        this.parseTemplate([
          '######',
          '#....#',
          '....#.',
          '...#..',
          '..#...',
          '..#...',
          '..#...'
        ]),
        // Recto
        this.parseTemplate([
          '####',
          '...#',
          '...#',
          '..#.',
          '..#.',
          '.#..',
          '.#..'
        ]),
        // Con barra
        this.parseTemplate([
          '#####',
          '....#',
          '....#',
          '...#.',
          '..#..',
          '..#..',
          '.#...'
        ]),
        // Europeo con barra
        this.parseTemplate([
          '######',
          '.....#',
          '....#.',
          '.####.',
          '..#...',
          '..#...',
          '..#...'
        ]),
        // Manuscrito curvo
        this.parseTemplate([
          '#####',
          '....#',
          '...#.',
          '...#.',
          '..#..',
          '..#..',
          '.#...'
        ]),
        // Muy inclinado
        this.parseTemplate([
          '####',
          '...#',
          '..#.',
          '..#.',
          '.#..',
          '.#..',
          '#...'
        ])
      ],
      8: [
        // Clásico simétrico
        this.parseTemplate([
          '.####.',
          '#....#',
          '#....#',
          '.####.',
          '#....#',
          '#....#',
          '.####.'
        ]),
        // Redondeado compacto
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '.###.',
          '#...#',
          '#...#',
          '.###.'
        ]),
        // Pequeño cuadrado
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '.##.',
          '#..#',
          '#..#',
          '.##.'
        ]),
        // Dos círculos separados (muñeco de nieve)
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '.#..#.',
          '..##..',
          '.#..#.',
          '.#..#.',
          '..##..'
        ]),
        // Estilo infinito/lazo
        this.parseTemplate([
          '.###.',
          '#...#',
          '.#.#.',
          '..#..',
          '.#.#.',
          '#...#',
          '.###.'
        ]),
        // Círculo arriba más pequeño
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '..##..',
          '.#..#.',
          '#....#',
          '#....#',
          '.####.'
        ]),
        // Círculo abajo más pequeño
        this.parseTemplate([
          '.####.',
          '#....#',
          '#....#',
          '.####.',
          '.#..#.',
          '.#..#.',
          '..##..'
        ]),
        // Muy redondeado continuo
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '#....#',
          '.#..#.',
          '#....#',
          '.#..#.',
          '..##..'
        ]),
        // Estilo cursivo
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '.##.',
          '#..#',
          '#..#',
          '.##.'
        ]),
        // Angular
        this.parseTemplate([
          '####.',
          '#...#',
          '#...#',
          '.###.',
          '#...#',
          '#...#',
          '####.'
        ]),
        // Cintura muy estrecha
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '..#..',
          '#...#',
          '#...#',
          '.###.'
        ]),
        // Dos óvalos verticales
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '#..#',
          '#..#',
          '#..#',
          '.##.'
        ]),
        // Estilo manuscrito rápido
        this.parseTemplate([
          '.##.',
          '#..#',
          '.#..',
          '..#.',
          '.#..',
          '#..#',
          '.##.'
        ]),
        // Con centro marcado
        this.parseTemplate([
          '.###.',
          '#...#',
          '#.#.#',
          '.###.',
          '#.#.#',
          '#...#',
          '.###.'
        ]),
        // Muy pequeño
        this.parseTemplate([
          '.#.',
          '#.#',
          '.#.',
          '#.#',
          '.#.'
        ]),
        // Ovalado horizontal
        this.parseTemplate([
          '.####.',
          '#....#',
          '.####.',
          '#....#',
          '.####.'
        ])
      ],
      9: [
        // Clásico con cola curva
        this.parseTemplate([
          '.####.',
          '#....#',
          '#....#',
          '.#####',
          '.....#',
          '....#.',
          '.###..'
        ]),
        // Redondeado con cola corta
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '.####',
          '....#',
          '...#.',
          '.##..'
        ]),
        // Con cola recta vertical
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '.####',
          '....#',
          '....#',
          '....#'
        ]),
        // Pequeño compacto
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '.###',
          '...#',
          '...#',
          '..#.'
        ]),
        // Cola muy larga
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '.####',
          '....#',
          '....#',
          '....#',
          '....#'
        ]),
        // Estilo q cursivo
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '.###',
          '...#',
          '..#.',
          '.#..'
        ]),
        // Círculo arriba con línea
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '.##.',
          '..#.',
          '..#.',
          '..#.'
        ]),
        // Angular cerrado
        this.parseTemplate([
          '####.',
          '#...#',
          '#...#',
          '.####',
          '....#',
          '....#',
          '###..'
        ]),
        // Muy redondeado
        this.parseTemplate([
          '..##..',
          '.#..#.',
          '#....#',
          '.#..##',
          '.....#',
          '....#.',
          '..##..'
        ]),
        // Con gancho abajo
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '.####',
          '....#',
          '...#.',
          '..#..',
          '.#...'
        ]),
        // Estilo reloj (como un 9 en reloj digital simplificado)
        this.parseTemplate([
          '####',
          '#..#',
          '####',
          '...#',
          '...#',
          '...#'
        ]),
        // Círculo pequeño arriba
        this.parseTemplate([
          '..#..',
          '.#.#.',
          '.#.#.',
          '..##.',
          '...#.',
          '...#.',
          '...#.'
        ]),
        // Ovalado con cola
        this.parseTemplate([
          '.##.',
          '#..#',
          '#..#',
          '#..#',
          '.###',
          '...#',
          '...#'
        ]),
        // Muy cursivo
        this.parseTemplate([
          '.##.',
          '#..#',
          '#.#.',
          '.#.#',
          '...#',
          '..#.',
          '.#..'
        ]),
        // Con círculo grande
        this.parseTemplate([
          '.####.',
          '#....#',
          '#....#',
          '#....#',
          '.#####',
          '.....#',
          '.....#'
        ]),
        // Cola recta larga
        this.parseTemplate([
          '.###.',
          '#...#',
          '#...#',
          '.####',
          '....#',
          '....#',
          '....#',
          '...#.'
        ])
      ]
    }
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
