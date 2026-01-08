// visualizer.js - Visualización de la red neuronal estilo MNIST
class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.updateCanvasSize();
        
        this.activations = [];
        
        // Estructura de la red: entrada (28 neuronas visibles), 3 capas ocultas, salida (10)
        this.layers = [
            { units: 28, name: 'Entrada (784→28)' },
            { units: 32, name: 'Oculta 1 (128→32)' },
            { units: 32, name: 'Oculta 2 (64→32)' },
            { units: 10, name: 'Salida' }
        ];
        
        window.addEventListener('resize', () => {
            this.updateCanvasSize();
            this.draw();
        });
        
        this.draw();
    }

    updateCanvasSize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    simulateActivations(inputSample, hidden1, hidden2, output) {
        // Normalizar y mapear activaciones a las dimensiones visibles
        this.activations = [
            (inputSample && inputSample.length > 0) ? this.normalizeArray(inputSample, 28) : new Array(28).fill(0),
            (hidden1 && hidden1.length > 0) ? this.normalizeArray(hidden1, 32) : new Array(32).fill(0),
            (hidden2 && hidden2.length > 0) ? this.normalizeArray(hidden2, 32) : new Array(32).fill(0),
            (output && output.length > 0) ? this.normalizeArray(output, 10) : new Array(10).fill(0.1)
        ];
        this.draw();
    }
    
    normalizeArray(arr, targetLength) {
        if (arr.length === targetLength) {
            return arr.map(v => Math.max(0, Math.min(1, v)));
        }
        
        // Si es más largo, muestrear
        if (arr.length > targetLength) {
            const result = [];
            const step = arr.length / targetLength;
            for (let i = 0; i < targetLength; i++) {
                const idx = Math.floor(i * step);
                result.push(Math.max(0, Math.min(1, arr[idx] || 0)));
            }
            return result;
        }
        
        // Si es más corto, repetir valores
        const result = [];
        for (let i = 0; i < targetLength; i++) {
            const idx = i % arr.length;
            result.push(Math.max(0, Math.min(1, arr[idx] || 0)));
        }
        return result;
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const padding = 40;
        const layerCount = this.layers.length;
        const availableWidth = this.canvas.width - 2 * padding;
        const layerSpacing = availableWidth / (layerCount - 1);
        const availableHeight = this.canvas.height - 20;

        // Primero dibujar todas las conexiones
        for (let i = 1; i < layerCount; i++) {
            this.drawLayerConnections(i, layerSpacing, padding, availableHeight);
        }

        // Luego dibujar todas las neuronas
        for (let i = 0; i < layerCount; i++) {
            this.drawLayer(i, layerSpacing, padding, availableHeight);
        }
    }

    drawLayerConnections(layerIndex, layerSpacing, padding, availableHeight) {
        const currentLayer = this.layers[layerIndex];
        const prevLayer = this.layers[layerIndex - 1];
        
        const currentX = padding + layerIndex * layerSpacing;
        const prevX = padding + (layerIndex - 1) * layerSpacing;
        
        const currentUnits = currentLayer.units;
        const prevUnits = prevLayer.units;
        
        const currentSpacing = Math.min(12, (availableHeight - 20) / Math.max(currentUnits, 1));
        const prevSpacing = Math.min(12, (availableHeight - 20) / Math.max(prevUnits, 1));
        
        const currentStartY = 10 + availableHeight / 2 - (currentUnits - 1) * currentSpacing / 2;
        const prevStartY = 10 + availableHeight / 2 - (prevUnits - 1) * prevSpacing / 2;

        // Dibujar conexiones (muestreadas para no saturar)
        const step = Math.max(1, Math.floor(prevUnits / 8));
        
        for (let j = 0; j < currentUnits; j += Math.max(1, Math.floor(currentUnits / 16))) {
            const currentY = currentStartY + j * currentSpacing;
            const currentActivation = this.getActivation(layerIndex, j);
            
            for (let k = 0; k < prevUnits; k += step) {
                const prevY = prevStartY + k * prevSpacing;
                const prevActivation = this.getActivation(layerIndex - 1, k);
                
                this.drawConnection(prevX, prevY, currentX, currentY, prevActivation, currentActivation);
            }
        }
    }

    drawLayer(layerIndex, layerSpacing, padding, availableHeight) {
        const layer = this.layers[layerIndex];
        const x = padding + layerIndex * layerSpacing;
        const units = layer.units;
        const neuronSpacing = Math.min(12, (availableHeight - 20) / Math.max(units, 1));
        const startY = 10 + availableHeight / 2 - (units - 1) * neuronSpacing / 2;

        for (let j = 0; j < units; j++) {
            const y = startY + j * neuronSpacing;
            const activation = this.getActivation(layerIndex, j);
            this.drawNeuron(x, y, activation);
        }
    }

    getActivation(layerIndex, neuronIndex) {
        if (this.activations[layerIndex] && this.activations[layerIndex][neuronIndex] !== undefined) {
            return Math.max(0, Math.min(1, this.activations[layerIndex][neuronIndex]));
        }
        return 0;
    }

    drawNeuron(x, y, activation) {
        const radius = 6;
        
        // Fondo de la neurona
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = 'white';
        this.ctx.fill();
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();

        // Relleno basado en activación
        if (activation > 0.05) {
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius - 1, 0, Math.PI * 2);
            const intensity = Math.floor(activation * 200);
            this.ctx.fillStyle = `rgb(${255 - intensity}, ${255 - intensity}, ${255 - intensity})`;
            this.ctx.fill();
        }
    }

    drawConnection(x1, y1, x2, y2, fromActivation, toActivation) {
        const strength = (fromActivation + toActivation) / 2;
        
        // Solo mostrar conexiones que tengan activación significativa
        if (strength < 0.15) {
            // No dibujar conexiones muy débiles (más mágico - solo ver lo importante)
            return;
        }
        
        let color;
        if (strength > 0.3) {
            // Verde brillante para conexiones muy activas (la magia!)
            const intensity = Math.min(1, (strength - 0.3) / 0.7);
            const greenValue = Math.floor(100 + intensity * 155); // 100-255
            color = `rgba(0, ${greenValue}, 0, ${0.5 + intensity * 0.5})`;
            this.ctx.lineWidth = 1 + intensity * 1; // Líneas más gruesas para conexiones fuertes
        } else if (strength > 0.2) {
            // Verde medio para conexiones moderadamente activas
            const intensity = (strength - 0.2) / 0.1;
            color = `rgba(0, ${Math.floor(150 * intensity)}, 0, ${0.3 + intensity * 0.3})`;
            this.ctx.lineWidth = 0.8;
        } else {
            // Gris muy tenue para conexiones débiles pero visibles
            color = `rgba(150, 150, 150, 0.15)`;
            this.ctx.lineWidth = 0.3;
        }

        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.strokeStyle = color;
        this.ctx.stroke();
    }
}

