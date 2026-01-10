# 🧠 Reconocimiento de Dígitos con Red Neuronal

Aplicación web para reconocimiento de dígitos manuscritos usando **React + TypeScript** en el frontend y **Node.js + Express** en el backend, con integración de **Groq API** para mejorar la precisión.

## 🛠️ Tecnologías

### Frontend (client/)
- **React 18** - Framework UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool rápido
- **TensorFlow.js** - Red neuronal en el navegador

### Backend (server/)
- **Node.js** - Runtime
- **Express** - Framework web
- **TypeScript** - Tipado estático
- **Groq API** - IA para mejorar predicciones

## 📁 Estructura del Proyecto

```
fncionamiento-red/
├── client/                    # Frontend React + TypeScript
│   ├── src/
│   │   ├── components/       # Componentes React
│   │   │   ├── DrawingCanvas.tsx
│   │   │   ├── PredictionPanel.tsx
│   │   │   ├── NetworkVisualizer.tsx
│   │   │   └── LoadingOverlay.tsx
│   │   ├── services/         # Servicios
│   │   │   ├── NeuralNetwork.ts
│   │   │   ├── GroqService.ts
│   │   │   └── DataGenerator.ts
│   │   ├── types/            # Tipos TypeScript
│   │   ├── styles/           # Estilos CSS
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── tsconfig.json
│
├── server/                    # Backend Node.js + Express
│   ├── src/
│   │   ├── routes/
│   │   │   └── groq.ts       # API de Groq
│   │   └── index.ts          # Entry point
│   ├── .env                  # Variables de entorno
│   ├── package.json
│   └── tsconfig.json
│
├── package.json              # Scripts del monorepo
└── README.md
```

## 🚀 Instalación

### 1. Clonar e instalar dependencias

```bash
# Instalar todas las dependencias
npm run install:all
```

### 2. Configurar variables de entorno

```bash
# Crear archivo .env en server/
cd server
cp .env.example .env

# Editar .env y agregar tu API key de Groq
GROQ_API_KEY=gsk_tu_api_key_aqui
```

### 3. Ejecutar en desarrollo

```bash
# Desde la raíz del proyecto
npm run dev
```

Esto inicia:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:4000

## 📦 Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `npm run dev` | Inicia frontend y backend en desarrollo |
| `npm run dev:client` | Solo frontend |
| `npm run dev:server` | Solo backend |
| `npm run build` | Construye para producción |
| `npm run install:all` | Instala todas las dependencias |

## 🎯 Características

### Red Neuronal CNN
- Arquitectura convolucional profunda
- 7 capas visualizadas en tiempo real
- Entrenamiento con datos sintéticos aumentados
- Precisión ~90-95%

### Integración Groq
- Modelo Llama 3.2 90B Vision
- API key segura en el servidor
- Combinación híbrida: 40% local + 60% Groq
- Mejora la precisión a ~95-98%

### UI/UX
- Diseño moderno con gradientes
- Canvas de dibujo preciso (32x32 grid)
- Pincel fino para mejor control
- Visualización de red neuronal en tiempo real
- Indicadores de confianza

## 🔒 Seguridad

La API key de Groq está protegida:
- Se almacena en `.env` (no se sube a git)
- El servidor actúa como proxy
- El frontend nunca ve la key

## 📊 API Endpoints

### `POST /api/groq/analyze`
Analiza una imagen de dígito.

**Request:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "digit": 7,
  "confidence": 0.85
}
```

### `GET /api/groq/status`
Verifica el estado de la API.

### `GET /api/health`
Health check del servidor.

## 🎨 Personalización

### Cambiar tamaño del canvas
```typescript
// client/src/App.tsx
<DrawingCanvas 
  gridSize={32}  // Cambiar aquí
  cellSize={12}  // Cambiar aquí
/>
```

### Cambiar modelo de Groq
```typescript
// server/src/routes/groq.ts
const GROQ_MODEL = 'llama-3.2-90b-vision-preview'
```

## 📝 Variables de Entorno

### server/.env
```env
PORT=4000
GROQ_API_KEY=gsk_tu_api_key
```

## 🤝 Contribuir

1. Fork el repositorio
2. Crear rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

MIT License

---

Desarrollado con ❤️ usando React, TypeScript y TensorFlow.js
