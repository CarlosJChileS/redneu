// app.js - Lógica principal para reconocimiento de dígitos con TensorFlow.js
const CELL_SIZE = 14;
const GRID_SIZE = 20;

let isDrawing = false;
let network, visualizer, drawCanvas, drawCtx, thumbnailCanvas, thumbnailCtx;
let predictionTimeout = null;
let isPredicting = false;
let modelReady = false;

window.onload = async function() {
    // Mostrar overlay de carga
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Obtener elementos
    drawCanvas = document.getElementById('drawCanvas');
    thumbnailCanvas = document.getElementById('thumbnailCanvas');
    
    // Configurar canvas de dibujo
    drawCanvas.width = GRID_SIZE * CELL_SIZE;
    drawCanvas.height = GRID_SIZE * CELL_SIZE;
    drawCtx = drawCanvas.getContext('2d');
    
    // Configurar thumbnail
    thumbnailCtx = thumbnailCanvas.getContext('2d');
    
    // Inicializar visualizador
    visualizer = new NetworkVisualizer('networkCanvas');
    
    // Inicializar red neuronal con TensorFlow
    network = new SimpleNeuralNetwork();
    
    // Limpiar canvas inicial
    limpiarCanvas();
    
    // Inicializar modelo en segundo plano
    try {
        await network.initialize();
        modelReady = true;
        console.log('Modelo TensorFlow listo');
    } catch (error) {
        console.error('Error inicializando modelo:', error);
        modelReady = true; // Usar fallback
    }
    
    // Ocultar overlay
    if (loadingOverlay) {
        loadingOverlay.classList.add('hidden');
    }
    limpiarCanvas();
    
    // === EVENTOS DE MOUSE ===
    drawCanvas.onmousedown = function(e) {
        isDrawing = true;
        dibujar(e);
    };
    
    drawCanvas.onmousemove = function(e) {
        if (isDrawing) {
            dibujar(e);
        }
    };
    
    drawCanvas.onmouseup = function() {
        isDrawing = false;
        // Hacer predicción final al soltar
        if (predictionTimeout) {
            clearTimeout(predictionTimeout);
        }
        hacerPrediccion();
    };
    
    drawCanvas.onmouseleave = function() {
        isDrawing = false;
    };
    
    // === EVENTOS TOUCH ===
    drawCanvas.ontouchstart = function(e) {
        e.preventDefault();
        isDrawing = true;
        dibujarTouch(e);
    };
    
    drawCanvas.ontouchmove = function(e) {
        e.preventDefault();
        if (isDrawing) {
            dibujarTouch(e);
        }
    };
    
    drawCanvas.ontouchend = function(e) {
        e.preventDefault();
        isDrawing = false;
        if (predictionTimeout) {
            clearTimeout(predictionTimeout);
        }
        hacerPrediccion();
    };
    
    // Botón limpiar
    document.getElementById('clearBtn').onclick = function() {
        limpiarCanvas();
    };
    
    console.log('Red neuronal simple lista!');
};

function dibujar(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    pintarCelda(x, y);
}

function dibujarTouch(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    pintarCelda(x, y);
}

function pintarCelda(x, y) {
    const cellX = Math.floor(x / CELL_SIZE);
    const cellY = Math.floor(y / CELL_SIZE);
    
    if (cellX >= 0 && cellX < GRID_SIZE && cellY >= 0 && cellY < GRID_SIZE) {
        // Pintar celda principal
        drawCtx.fillStyle = 'black';
        drawCtx.fillRect(cellX * CELL_SIZE, cellY * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        
        // Pintar vecinos inmediatos (trazo grueso)
        drawCtx.fillStyle = 'rgba(0,0,0,0.7)';
        if (cellX > 0) drawCtx.fillRect((cellX-1) * CELL_SIZE, cellY * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (cellX < GRID_SIZE-1) drawCtx.fillRect((cellX+1) * CELL_SIZE, cellY * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (cellY > 0) drawCtx.fillRect(cellX * CELL_SIZE, (cellY-1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (cellY < GRID_SIZE-1) drawCtx.fillRect(cellX * CELL_SIZE, (cellY+1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        
        // Pintar diagonales para trazo más suave
        drawCtx.fillStyle = 'rgba(0,0,0,0.35)';
        if (cellX > 0 && cellY > 0) drawCtx.fillRect((cellX-1) * CELL_SIZE, (cellY-1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (cellX < GRID_SIZE-1 && cellY > 0) drawCtx.fillRect((cellX+1) * CELL_SIZE, (cellY-1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (cellX > 0 && cellY < GRID_SIZE-1) drawCtx.fillRect((cellX-1) * CELL_SIZE, (cellY+1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (cellX < GRID_SIZE-1 && cellY < GRID_SIZE-1) drawCtx.fillRect((cellX+1) * CELL_SIZE, (cellY+1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        
        // Redibujar cuadrícula
        dibujarCuadricula();
        
        // Hacer predicción en tiempo real mientras dibujas (debounce muy corto)
        if (predictionTimeout) {
            clearTimeout(predictionTimeout);
        }
        // Ejecutar predicción después de un breve delay para no bloquear el dibujo
        predictionTimeout = setTimeout(() => {
            hacerPrediccion();
        }, 30); // Reducido a 30ms para respuesta casi instantánea mientras dibujas
    }
}

function dibujarCuadricula() {
    drawCtx.strokeStyle = '#ccc';
    drawCtx.lineWidth = 0.5;
    
    for (let i = 0; i <= GRID_SIZE; i++) {
        drawCtx.beginPath();
        drawCtx.moveTo(i * CELL_SIZE, 0);
        drawCtx.lineTo(i * CELL_SIZE, GRID_SIZE * CELL_SIZE);
        drawCtx.stroke();
        
        drawCtx.beginPath();
        drawCtx.moveTo(0, i * CELL_SIZE);
        drawCtx.lineTo(GRID_SIZE * CELL_SIZE, i * CELL_SIZE);
        drawCtx.stroke();
    }
}

function limpiarCanvas() {
    drawCtx.fillStyle = 'white';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    dibujarCuadricula();
    
    thumbnailCtx.fillStyle = 'white';
    thumbnailCtx.fillRect(0, 0, 28, 28);
    
    // Resetear salidas
    for (let i = 0; i < 10; i++) {
        const box = document.getElementById('output-' + i);
        box.style.backgroundColor = 'white';
        box.style.borderColor = '#333';
        box.classList.remove('highlight');
    }
    
    visualizer.simulateActivations([], [], [], []);
}

function hacerPrediccion() {
    if (!network || !network.isLoaded || isPredicting) {
        return;
    }
    
    isPredicting = true;
    
    try {
        // Redimensionar a 28x28
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Fondo blanco
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, 28, 28);
        
        // Dibujar imagen escalada
        tempCtx.drawImage(drawCanvas, 0, 0, 28, 28);
        
        // Actualizar thumbnail
        thumbnailCtx.fillStyle = 'white';
        thumbnailCtx.fillRect(0, 0, 28, 28);
        thumbnailCtx.drawImage(tempCanvas, 0, 0);
        
        // Obtener píxeles (invertir: negro = 1, blanco = 0)
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const pixels = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            // Convertir RGB a escala de grises y normalizar
            const gray = 1 - (imageData.data[i] / 255); // Invertir: negro = 1
            pixels.push(gray);
        }
        
        // Predicción (síncrona, sin await)
        const result = network.predict(pixels);
        
        // Actualizar cajas de salida con probabilidades
        const maxProb = Math.max(...result.probabilities);
        const maxIndex = result.probabilities.indexOf(maxProb);
        
        for (let i = 0; i < 10; i++) {
            const box = document.getElementById('output-' + i);
            const prob = result.probabilities[i];
            
            // Si es el más probable y tiene probabilidad significativa, ponerlo negro
            if (i === maxIndex && prob > 0.1) {
                box.style.backgroundColor = '#333'; // Negro
                box.style.borderColor = '#000'; // Borde más oscuro
                box.classList.add('highlight');
            } else {
                // Para los demás, escala de gris basada en probabilidad
                const intensity = Math.floor(prob * 255);
                const grayValue = 255 - intensity;
                box.style.backgroundColor = `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
                box.style.borderColor = '#333';
                box.classList.remove('highlight');
            }
        }
        
        // Actualizar visualizador con activaciones reales
        // Normalizar activaciones para visualización
        const normalizeActivation = (arr) => {
            if (!arr || arr.length === 0) return [];
            const max = Math.max(...arr);
            if (max === 0) return arr.map(() => 0);
            return arr.map(v => Math.min(1, Math.max(0, v / max)));
        };
        
        visualizer.simulateActivations(
            normalizeActivation(result.inputSample || []),
            normalizeActivation(result.hidden1 || []), // Ya viene con 32 elementos
            normalizeActivation(result.hidden2 || []), // Ya viene con 32 elementos
            normalizeActivation(result.probabilities)
        );
        
    } catch (error) {
        console.error('Error en predicción:', error);
    } finally {
        isPredicting = false;
    }
}
