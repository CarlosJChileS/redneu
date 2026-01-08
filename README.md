# 🧠 Visualizador Interactivo de Redes Neuronales

Un proyecto interactivo para demostrar el funcionamiento de las redes neuronales, perfecto para ferias universitarias y presentaciones del club de IA.

## 🎯 Características

- ✨ **Visualización en tiempo real** de la arquitectura de la red neuronal
- 🎮 **Controles interactivos** para ajustar parámetros (capas, neuronas, tasa de aprendizaje)
- 📊 **Gráficos en vivo** de pérdida y precisión durante el entrenamiento
- 🎨 **Animaciones fluidas** mostrando activaciones de neuronas
- 💻 **100% en el navegador** - no requiere servidor ni instalaciones

## 🚀 Cómo usar

1. **Abre el proyecto**
   - Simplemente abre el archivo `index.html` en tu navegador web moderno (Chrome, Firefox, Edge, Safari)

2. **Ajusta los parámetros**
   - **Número de Capas Ocultas**: Controla la profundidad de la red (1-5 capas)
   - **Neuronas por Capa**: Ajusta el ancho de cada capa (2-20 neuronas)
   - **Tasa de Aprendizaje**: Velocidad de aprendizaje (0.001 - 0.5)
   - **Función de Activación**: Elige entre Sigmoid, ReLU o Tanh

3. **Entrena la red**
   - Haz clic en el botón "🚀 Entrenar Red" para comenzar el entrenamiento
   - Observa cómo cambian las activaciones de las neuronas en tiempo real
   - Mira los gráficos de pérdida y precisión actualizarse durante el entrenamiento

4. **Experimenta**
   - Prueba diferentes configuraciones para ver cómo afectan al rendimiento
   - Compara diferentes funciones de activación
   - Observa cómo la tasa de aprendizaje influye en la velocidad de convergencia

## 📁 Estructura del Proyecto

```
proyecto-red-neuronal/
│
├── index.html          # Estructura HTML principal
├── styles.css          # Estilos y diseño
├── network.js          # Lógica de la red neuronal (TensorFlow.js)
├── visualizer.js       # Visualización del canvas
├── app.js              # Lógica principal de la aplicación
└── README.md           # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **TensorFlow.js**: Para la implementación de la red neuronal
- **Chart.js**: Para los gráficos de pérdida y precisión
- **HTML5 Canvas**: Para la visualización de la red
- **Vanilla JavaScript**: Sin dependencias de frameworks

## 💡 Ideas para la Feria

- **Demostración en vivo**: Muestra cómo funciona el entrenamiento en tiempo real
- **Comparación de configuraciones**: Prepara ejemplos con diferentes parámetros
- **Explicación educativa**: Usa la visualización para explicar conceptos como:
  - Propagación hacia adelante (forward propagation)
  - Backpropagation
  - Funciones de activación
  - Overfitting vs Underfitting

## 📝 Notas

- El proyecto funciona completamente offline una vez cargadas las librerías desde CDN
- Para uso en producción, considera descargar las librerías localmente
- El entrenamiento se realiza con datos sintéticos (clasificación 2D)
- Puedes modificar la función `generateData()` en `network.js` para usar otros datos

## 🎓 Conceptos Demostrados

- Arquitectura de redes neuronales multicapa
- Entrenamiento mediante descenso de gradiente
- Funciones de activación y su impacto
- Visualización de activaciones neuronales
- Métricas de rendimiento (pérdida y precisión)

---

¡Disfruta experimentando con redes neuronales! 🚀

