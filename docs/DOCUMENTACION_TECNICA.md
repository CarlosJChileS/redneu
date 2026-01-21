# Documentacion Tecnica - Red Neuronal para Reconocimiento de Digitos

## Indice

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Stack Tecnologico](#stack-tecnologico)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Red Neuronal Convolucional (CNN)](#red-neuronal-convolucional-cnn)
5. [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
6. [Generacion de Dataset Sintetico](#generacion-de-dataset-sintetico)
7. [API del Servidor](#api-del-servidor)
8. [Persistencia del Modelo](#persistencia-del-modelo)
9. [Optimizaciones Implementadas](#optimizaciones-implementadas)
10. [Flujo de Prediccion](#flujo-de-prediccion)

---

## Arquitectura del Sistema

```
+------------------+     HTTP/REST      +------------------+
|                  | <----------------> |                  |
|   Cliente        |                    |   Servidor       |
|   (React + Vite) |                    |   (Express)      |
|                  |                    |                  |
+------------------+                    +------------------+
        |                                       |
        v                                       v
+------------------+                    +------------------+
|   TensorFlow.js  |                    |   MongoDB        |
|   (WebGL/CPU)    |                    |   (Persistencia) |
+------------------+                    +------------------+
        |
        v
+------------------+
|   IndexedDB      |
|   (Cache Local)  |
+------------------+
```

### Patron de Comunicacion

- **Cliente-Servidor**: REST API para sincronizacion de modelos
- **Cliente-Local**: IndexedDB para cache de modelos entrenados
- **Tiempo Real**: Prediccion en cliente sin latencia de red

---

## Stack Tecnologico

### Frontend
| Tecnologia | Version | Proposito |
|------------|---------|-----------|
| React | 18.x | UI declarativa con componentes |
| TypeScript | 5.x | Tipado estatico |
| Vite | 5.4.x | Build tool y dev server |
| TensorFlow.js | 4.x | Inferencia y entrenamiento en browser |

### Backend
| Tecnologia | Version | Proposito |
|------------|---------|-----------|
| Node.js | 18+ | Runtime JavaScript |
| Express | 4.x | Framework HTTP |
| MongoDB | 6.x | Base de datos NoSQL |
| Mongoose | 8.x | ODM para MongoDB |
| tsx | - | Ejecucion TypeScript directa |

### Infraestructura
| Componente | Proposito |
|------------|-----------|
| concurrently | Ejecucion paralela de procesos |
| IndexedDB | Almacenamiento local del modelo |
| WebGL | Aceleracion GPU para TensorFlow |

---

## Estructura del Proyecto

```
fncionamiento-red/
├── package.json              # Monorepo config
├── start.ps1                 # Script inicio Windows
├── start.sh                  # Script inicio Unix
│
├── client/                   # Frontend React
│   ├── src/
│   │   ├── components/
│   │   │   ├── DrawingCanvas.tsx      # Canvas de dibujo 280x280
│   │   │   ├── NetworkVisualizer.tsx  # Visualizacion de capas
│   │   │   ├── PredictionPanel.tsx    # Panel de resultados
│   │   │   └── LoadingOverlay.tsx     # Overlay de progreso
│   │   ├── services/
│   │   │   └── NeuralNetwork.ts       # Clase principal CNN
│   │   ├── styles/                    # CSS modular
│   │   └── types/                     # Interfaces TypeScript
│   └── vite.config.ts
│
└── server/                   # Backend Express
    └── src/
        ├── index.ts          # Entry point
        ├── db/
        │   └── connection.ts # Conexion MongoDB
        ├── models/
        │   └── NeuralModel.ts # Schema Mongoose
        └── routes/
            ├── model.ts      # CRUD modelo
            └── groq.ts       # Integracion IA
```

---

## Red Neuronal Convolucional (CNN)

### Arquitectura del Modelo

```typescript
private buildModel(): tf.Sequential {
  const m = tf.sequential()
  
  // Bloque Convolucional 1
  m.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],  // Imagen 28x28 grayscale
    filters: 32,               // 32 filtros
    kernelSize: 3,             // Kernel 3x3
    activation: 'relu',
    padding: 'same'
  }))
  m.add(tf.layers.batchNormalization())
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }))
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }))  // 28x28 -> 14x14
  m.add(tf.layers.dropout({ rate: 0.25 }))
  
  // Bloque Convolucional 2
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }))
  m.add(tf.layers.batchNormalization())
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }))
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }))  // 14x14 -> 7x7
  m.add(tf.layers.dropout({ rate: 0.25 }))
  
  // Capas Densas
  m.add(tf.layers.flatten())                      // 7*7*64 = 3136 neuronas
  m.add(tf.layers.dense({ units: 256, activation: 'relu' }))
  m.add(tf.layers.batchNormalization())
  m.add(tf.layers.dropout({ rate: 0.4 }))
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }))
  m.add(tf.layers.dropout({ rate: 0.3 }))
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }))  // 10 clases

  m.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })
  
  return m
}
```

### Flujo de Datos por Capa

| Capa | Entrada | Salida | Parametros |
|------|---------|--------|------------|
| Conv2D (32) | 28x28x1 | 28x28x32 | 320 |
| BatchNorm | 28x28x32 | 28x28x32 | 128 |
| Conv2D (32) | 28x28x32 | 28x28x32 | 9,248 |
| MaxPool | 28x28x32 | 14x14x32 | 0 |
| Conv2D (64) | 14x14x32 | 14x14x64 | 18,496 |
| BatchNorm | 14x14x64 | 14x14x64 | 256 |
| Conv2D (64) | 14x14x64 | 14x14x64 | 36,928 |
| MaxPool | 14x14x64 | 7x7x64 | 0 |
| Flatten | 7x7x64 | 3136 | 0 |
| Dense (256) | 3136 | 256 | 803,072 |
| Dense (128) | 256 | 128 | 32,896 |
| Dense (10) | 128 | 10 | 1,290 |

**Total parametros**: ~902,634

---

## Pipeline de Entrenamiento

### Configuracion

```typescript
await this.model.fit(xs, ys, {
  epochs: 25,              // Iteraciones completas
  batchSize: 64,           // Muestras por actualizacion
  shuffle: true,           // Aleatorizar orden
  validationSplit: 0.15,   // 15% para validacion
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      // Actualizar UI con progreso
      this.updateProgress(progress, `Epoca ${epoch + 1}/25`)
      await this.yieldToUI()  // Evitar bloqueo de UI
    }
  }
})
```

### Dataset

- **Total muestras**: 3,500 (350 por digito x 10 digitos)
- **Split**: 85% entrenamiento / 15% validacion
- **Augmentation**: Aplicado en tiempo real

---

## Generacion de Dataset Sintetico

### Templates Base

Cada digito tiene multiples representaciones en ASCII-art:

```typescript
// Ejemplo para el digito "0"
0: [
  this.p(['..###..', '.#...#.', '#.....#', '#.....#', '.#...#.', '..###..']),
  this.p(['.####.', '#....#', '#....#', '#....#', '.####.']),
  // ... mas variaciones
]
```

### Estilos de Escritura (Data Augmentation)

```typescript
interface WritingStyle {
  scale: number      // 1.5 - 5.5 (tamano)
  thickness: number  // 0.8 - 1.6 (grosor de trazo)
  shearX: number     // -0.4 - 0.4 (inclinacion horizontal)
  shearY: number     // -0.1 - 0.1 (inclinacion vertical)
  stretchX: number   // 0.6 - 1.4 (estiramiento horizontal)
  stretchY: number   // 0.6 - 1.4 (estiramiento vertical)
  blur: number       // 0.1 - 0.7 (desenfoque)
  noise: number      // 0 - 0.25 (ruido)
}
```

### Transformacion Afin

```typescript
private renderWithStyle(template: number[][], style: WritingStyle): number[][][] {
  // Transformacion geometrica
  let dx = (x - 14 - offsetX) / style.stretchX
  let dy = (y - 14 - offsetY) / style.stretchY
  
  // Shear (inclinacion)
  dx += dy * style.shearX
  dy += dx * style.shearY
  
  // Rotacion
  const rx = dx * cos + dy * sin
  const ry = -dx * sin + dy * cos
  
  // Interpolacion bilineal
  val = v00 * (1-fx) * (1-fy) + 
        v10 * fx * (1-fy) + 
        v01 * (1-fx) * fy + 
        v11 * fx * fy
        
  // Grosor
  val *= style.thickness
  
  // Blur (promedio con vecinos)
  const neighbors = (raw[y-1][x] + raw[y+1][x] + raw[y][x-1] + raw[y][x+1]) / 4
  val = val * (1 - style.blur * 0.6) + neighbors * style.blur * 0.6
}
```

---

## API del Servidor

### Endpoints

#### GET /api/model
Recupera el modelo entrenado desde MongoDB.

**Response:**
```json
{
  "success": true,
  "model": {
    "modelJson": { ... },      // Arquitectura serializada
    "weightsBase64": "...",    // Pesos en Base64
    "version": "10.0.0",
    "accuracy": 0.97,
    "createdAt": "2026-01-21T..."
  }
}
```

#### POST /api/model
Guarda un modelo entrenado.

**Request Body:**
```json
{
  "modelJson": { ... },
  "weightsBase64": "...",
  "version": "10.0.0",
  "accuracy": 0.97
}
```

#### DELETE /api/model
Elimina el modelo almacenado.

### Schema MongoDB

```typescript
const neuralModelSchema = new Schema({
  modelJson: { type: Object, required: true },
  weightsBase64: { type: String, required: true },
  version: { type: String, required: true },
  accuracy: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now }
})
```

---

## Persistencia del Modelo

### Estrategia de Cache Multinivel

```
1. Servidor (MongoDB)     <- Fuente de verdad
      |
      v
2. Cliente (IndexedDB)    <- Cache local
      |
      v
3. Memoria (tf.Model)     <- Runtime
```

### Flujo de Inicializacion

```typescript
async initialize(): Promise<boolean> {
  // 1. Intentar cargar del servidor
  const loadedFromServer = await this.tryLoadFromServer()
  if (loadedFromServer) return true

  // 2. Intentar cargar de IndexedDB
  const loadedLocally = await this.tryLoadSavedModel()
  if (loadedLocally) return true

  // 3. Entrenar nuevo modelo
  this.model = this.buildModel()
  await this.train()
  
  // 4. Guardar en ambos lugares
  await this.saveModel()        // IndexedDB
  await this.uploadToServer()   // MongoDB
}
```

### Serializacion de Pesos (Fix Stack Overflow)

```typescript
// Conversion a Base64 usando chunks para evitar "Maximum call stack size exceeded"
private arrayBufferToBase64Chunked(uint8Array: Uint8Array): string {
  const CHUNK_SIZE = 0x8000  // 32KB chunks
  const chunks: string[] = []
  
  for (let i = 0; i < uint8Array.length; i += CHUNK_SIZE) {
    const chunk = uint8Array.subarray(i, Math.min(i + CHUNK_SIZE, uint8Array.length))
    chunks.push(String.fromCharCode.apply(null, Array.from(chunk)))
  }
  
  return btoa(chunks.join(''))
}
```

---

## Optimizaciones Implementadas

### 1. Singleton Pattern
```typescript
static getInstance(): NeuralNetwork {
  if (!instance) {
    instance = new NeuralNetwork()
  }
  return instance
}
```
Evita multiples instancias y re-entrenamientos.

### 2. Yield to UI
```typescript
private async yieldToUI(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 10))
}
```
Previene bloqueo del thread principal durante entrenamiento.

### 3. WebGL Backend
```typescript
try {
  await tf.setBackend('webgl')  // GPU
} catch (e) {
  await tf.setBackend('cpu')    // Fallback
}
```
Aceleracion por GPU cuando esta disponible.

### 4. Batch Normalization
Normaliza activaciones entre capas para:
- Entrenamiento mas estable
- Permite learning rates mas altos
- Actua como regularizacion

### 5. Dropout
```typescript
m.add(tf.layers.dropout({ rate: 0.25 }))  // Capas conv
m.add(tf.layers.dropout({ rate: 0.4 }))   // Capas densas
```
Previene overfitting desactivando neuronas aleatoriamente.

---

## Flujo de Prediccion

### Preprocesamiento de Entrada

```typescript
// Canvas 280x280 -> Imagen 28x28
const pixels = ctx.getImageData(0, 0, 280, 280)

// Downsampling 10:1
for (let y = 0; y < 28; y++) {
  for (let x = 0; x < 28; x++) {
    let sum = 0
    for (let dy = 0; dy < 10; dy++) {
      for (let dx = 0; dx < 10; dx++) {
        const idx = ((y * 10 + dy) * 280 + (x * 10 + dx)) * 4
        sum += pixels.data[idx + 3]  // Canal alpha
      }
    }
    input[y * 28 + x] = sum / (10 * 10 * 255)  // Normalizar [0,1]
  }
}
```

### Inferencia

```typescript
async predict(pixels: number[]): Promise<PredictionResult> {
  // Reshape a tensor 4D: [batch, height, width, channels]
  const t = tf.tensor4d([img])  // [1, 28, 28, 1]
  
  // Forward pass
  const p = this.model.predict(t) as tf.Tensor
  const probs = Array.from(p.dataSync())  // 10 probabilidades
  
  // Argmax
  const best = Math.max(...probs)
  const digit = probs.indexOf(best)
  
  // Cleanup
  t.dispose()
  p.dispose()
  
  return { predictedDigit: digit, confidence: best, probabilities: probs }
}
```

---

## Metricas de Rendimiento

| Metrica | Valor Tipico |
|---------|--------------|
| Tiempo de entrenamiento | 30-60 segundos |
| Tiempo de prediccion | < 10 ms |
| Precision en validacion | 95-98% |
| Tamano del modelo | ~3.5 MB |
| Muestras de entrenamiento | 3,500 |

---

## Referencias

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Convolutional Neural Networks (CS231n)](https://cs231n.github.io/convolutional-networks/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Dropout Paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
