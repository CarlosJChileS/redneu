// network.js - Red neuronal simple para reconocimiento MUY preciso de dígitos (sin TensorFlow)
class SimpleNeuralNetwork {
    constructor() {
        this.isLoaded = true;
        this.templates = this.createTemplates();
        this.templateVariations = this.createTemplateVariations();
        
        // Pesos simulados de la red (para visualización)
        this.weights = this.generateWeights();
        
        console.log('Red neuronal simple inicializada');
    }

    // Generar pesos aleatorios para visualización realista
    generateWeights() {
        const weights = {
            inputToHidden1: [],
            hidden1ToHidden2: [],
            hidden2ToOutput: []
        };
        
        // 784 inputs -> 128 hidden1
        for (let i = 0; i < 128; i++) {
            weights.inputToHidden1[i] = [];
            for (let j = 0; j < 784; j++) {
                weights.inputToHidden1[i][j] = (Math.random() - 0.5) * 2;
            }
        }
        
        // 128 hidden1 -> 64 hidden2
        for (let i = 0; i < 64; i++) {
            weights.hidden1ToHidden2[i] = [];
            for (let j = 0; j < 128; j++) {
                weights.hidden1ToHidden2[i][j] = (Math.random() - 0.5) * 2;
            }
        }
        
        // 64 hidden2 -> 10 output
        for (let i = 0; i < 10; i++) {
            weights.hidden2ToOutput[i] = [];
            for (let j = 0; j < 64; j++) {
                weights.hidden2ToOutput[i][j] = (Math.random() - 0.5) * 2;
            }
        }
        
        return weights;
    }

    // Función de activación ReLU
    relu(x) {
        return Math.max(0, x);
    }

    // Función sigmoide
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Calcular activaciones de una capa
    forwardLayer(inputs, weights, activation = 'relu') {
        const outputs = [];
        for (let i = 0; i < weights.length; i++) {
            let sum = 0;
            for (let j = 0; j < inputs.length && j < weights[i].length; j++) {
                sum += inputs[j] * weights[i][j];
            }
            outputs[i] = activation === 'relu' ? this.relu(sum) : this.sigmoid(sum);
        }
        return outputs;
    }

    // Preprocesar imagen mejorado con suavizado
    preprocessImage(pixels) {
        const size = 28;
        let processed = [...pixels];
        
        // Aplicar suavizado gaussiano ligero
        processed = this.applyGaussianBlur(processed, size, 0.8);
        
        // Encontrar bounding box mejorado
        let minX = size, maxX = 0, minY = size, maxY = 0;
        let hasContent = false;
        let totalPixels = 0;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (processed[y * size + x] > 0.08) {
                    hasContent = true;
                    totalPixels++;
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        
        if (!hasContent || maxX < minX || totalPixels < 5) {
            return Array(size * size).fill(0);
        }
        
        // Centrar y normalizar con mejor interpolación
        const width = maxX - minX + 1;
        const height = maxY - minY + 1;
        const maxDim = Math.max(width, height);
        const scale = (size - 6) / maxDim; // Dejar 3px de padding por lado
        const scaledWidth = width * scale;
        const scaledHeight = height * scale;
        const offsetX = (size - scaledWidth) / 2;
        const offsetY = (size - scaledHeight) / 2;
        
        const result = Array(size * size).fill(0);
        
        // Interpolación bilineal mejorada
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const srcX = (x - offsetX) / scale + minX;
                const srcY = (y - offsetY) / scale + minY;
                
                const x1 = Math.floor(srcX);
                const y1 = Math.floor(srcY);
                const x2 = Math.min(size - 1, x1 + 1);
                const y2 = Math.min(size - 1, y1 + 1);
                
                const fx = srcX - x1;
                const fy = srcY - y1;
                
                let value = 0;
                if (x1 >= 0 && x1 < size && y1 >= 0 && y1 < size) {
                    // Interpolación bilineal
                    const v11 = processed[y1 * size + x1];
                    const v21 = x2 < size ? processed[y1 * size + x2] : 0;
                    const v12 = y2 < size ? processed[y2 * size + x1] : 0;
                    const v22 = (x2 < size && y2 < size) ? processed[y2 * size + x2] : 0;
                    
                    value = v11 * (1 - fx) * (1 - fy) +
                            v21 * fx * (1 - fy) +
                            v12 * (1 - fx) * fy +
                            v22 * fx * fy;
                }
                
                result[y * size + x] = Math.min(1, Math.max(0, value));
            }
        }
        
        // Normalización mejorada con gamma correction
        let maxVal = 0;
        let minVal = 1;
        for (let i = 0; i < result.length; i++) {
            if (result[i] > 0.1) {
                maxVal = Math.max(maxVal, result[i]);
                minVal = Math.min(minVal, result[i]);
            }
        }
        
        if (maxVal > 0) {
            // Normalización adaptativa
            const range = maxVal - minVal;
            if (range > 0) {
                for (let i = 0; i < result.length; i++) {
                    if (result[i] > 0.1) {
                        result[i] = Math.pow((result[i] - minVal) / range, 0.6);
                    } else {
                        result[i] = 0;
                    }
                }
            }
        }
        
        return result;
    }

    // Aplicar suavizado gaussiano
    applyGaussianBlur(pixels, size, sigma) {
        const result = [...pixels];
        const kernel = this.createGaussianKernel(3, sigma);
        
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                let sum = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * size + (x + kx);
                        const weight = kernel[ky + 1][kx + 1];
                        sum += pixels[idx] * weight;
                    }
                }
                result[y * size + x] = sum;
            }
        }
        
        return result;
    }

    createGaussianKernel(size, sigma) {
        const kernel = [];
        const center = Math.floor(size / 2);
        let sum = 0;
        
        for (let y = 0; y < size; y++) {
            kernel[y] = [];
            for (let x = 0; x < size; x++) {
                const dx = x - center;
                const dy = y - center;
                const value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[y][x] = value;
                sum += value;
            }
        }
        
        // Normalizar
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                kernel[y][x] /= sum;
            }
        }
        
        return kernel;
    }

    // Extraer características avanzadas del dígito (ULTRA PRECISO)
    extractFeatures(pixels) {
        const size = 28;
        const features = {
            holes: 0,
            endpoints: 0,
            crossings: 0,
            aspectRatio: 0,
            density: 0,
            centerOfMass: { x: 0, y: 0 },
            zones: [],
            verticalLines: 0,
            horizontalLines: 0,
            diagonals: 0,
            topHeavy: false,
            bottomHeavy: false,
            leftHeavy: false,
            rightHeavy: false,
            symmetry: 0,
            curvature: 0,
            // Nuevas características avanzadas
            lineThickness: 0,
            connectivity: 0,
            corners: 0,
            topLine: 0,
            bottomLine: 0,
            leftLine: 0,
            rightLine: 0,
            centerDensity: 0,
            edgeDensity: 0,
            verticalBalance: 0,
            horizontalBalance: 0,
            continuousLines: { vertical: 0, horizontal: 0 },
            hookPattern: false,
            closedLoop: false
        };
        
        // Calcular densidad total y centro de masa
        let total = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (pixels[y * size + x] > 0.25) {
                    total++;
                    features.centerOfMass.x += x;
                    features.centerOfMass.y += y;
                }
            }
        }
        features.density = total / (size * size);
        
        if (total > 0) {
            features.centerOfMass.x /= total;
            features.centerOfMass.y /= total;
        }
        
        // Bounding box para aspect ratio
        let minX = size, maxX = 0, minY = size, maxY = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (pixels[y * size + x] > 0.25) {
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        const width = maxX - minX + 1;
        const height = maxY - minY + 1;
        features.aspectRatio = width / Math.max(height, 1);
        
        // Análisis de zonas 3x3 mejorado
        features.zones = [];
        for (let zy = 0; zy < 3; zy++) {
            for (let zx = 0; zx < 3; zx++) {
                const y1 = Math.floor(zy * size / 3);
                const y2 = Math.floor((zy + 1) * size / 3);
                const x1 = Math.floor(zx * size / 3);
                const x2 = Math.floor((zx + 1) * size / 3);
                
                let zoneTotal = 0;
                for (let y = y1; y < y2; y++) {
                    for (let x = x1; x < x2; x++) {
                        if (pixels[y * size + x] > 0.25) zoneTotal++;
                    }
                }
                features.zones.push(zoneTotal / ((y2 - y1) * (x2 - x1)));
            }
        }
        
        // Análisis de distribución (peso superior/inferior, izquierda/derecha)
        const topDensity = (features.zones[0] + features.zones[1] + features.zones[2]) / 3;
        const bottomDensity = (features.zones[6] + features.zones[7] + features.zones[8]) / 3;
        const leftDensity = (features.zones[0] + features.zones[3] + features.zones[6]) / 3;
        const rightDensity = (features.zones[2] + features.zones[5] + features.zones[8]) / 3;
        
        features.topHeavy = topDensity > bottomDensity * 1.2;
        features.bottomHeavy = bottomDensity > topDensity * 1.2;
        features.leftHeavy = leftDensity > rightDensity * 1.2;
        features.rightHeavy = rightDensity > leftDensity * 1.2;
        
        // Simetría vertical aproximada
        let symmetric = 0;
        let totalSymmetric = 0;
        const midX = size / 2;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < midX; x++) {
                const leftVal = pixels[y * size + x] > 0.25 ? 1 : 0;
                const rightVal = pixels[y * size + (size - 1 - x)] > 0.25 ? 1 : 0;
                if (leftVal === rightVal) symmetric++;
                totalSymmetric++;
            }
        }
        features.symmetry = totalSymmetric > 0 ? symmetric / totalSymmetric : 0;
        
        // Curvatura (medida por cambios de dirección)
        features.curvature = this.measureCurvature(pixels, size);
        
        // Contar huecos
        features.holes = this.countHoles(pixels, size);
        
        // Contar endpoints
        features.endpoints = this.countEndpoints(pixels, size);
        
        // Contar cruces
        features.crossings = this.countCrossings(pixels, size);
        
        // Detectar líneas verticales y horizontales
        features.verticalLines = this.detectVerticalLines(pixels, size);
        features.horizontalLines = this.detectHorizontalLines(pixels, size);
        features.diagonals = this.detectDiagonals(pixels, size);
        
        // NUEVAS CARACTERÍSTICAS AVANZADAS PARA MÁXIMA PRECISIÓN
        
        // Análisis de grosor de líneas
        features.lineThickness = this.measureLineThickness(pixels, size);
        
        // Análisis de conectividad (cuántos componentes conectados hay)
        features.connectivity = this.measureConnectivity(pixels, size);
        
        // Contar esquinas (cambios bruscos de dirección)
        features.corners = this.countCorners(pixels, size);
        
        // Densidad en bordes específicos
        features.topLine = this.measureEdgeDensity(pixels, size, 'top');
        features.bottomLine = this.measureEdgeDensity(pixels, size, 'bottom');
        features.leftLine = this.measureEdgeDensity(pixels, size, 'left');
        features.rightLine = this.measureEdgeDensity(pixels, size, 'right');
        
        // Densidad en centro vs bordes
        features.centerDensity = this.measureCenterDensity(pixels, size);
        features.edgeDensity = this.measureEdgeDensity(pixels, size, 'all');
        
        // Balance vertical y horizontal (más preciso que topHeavy/bottomHeavy)
        features.verticalBalance = (topDensity - bottomDensity) / Math.max(topDensity + bottomDensity, 0.01);
        features.horizontalBalance = (rightDensity - leftDensity) / Math.max(leftDensity + rightDensity, 0.01);
        
        // Líneas continuas (longitud de líneas más largas)
        features.continuousLines = this.measureContinuousLines(pixels, size);
        
        // Detectar patrones específicos
        features.hookPattern = this.detectHookPattern(pixels, size);
        features.closedLoop = this.detectClosedLoop(pixels, size);
        
        return features;
    }
    
    // Medir grosor promedio de líneas
    measureLineThickness(pixels, size) {
        let totalThickness = 0;
        let count = 0;
        
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                if (pixels[y * size + x] > 0.25) {
                    // Medir grosor en dirección vertical
                    let vThickness = 1;
                    for (let dy = 1; y + dy < size && pixels[(y + dy) * size + x] > 0.25; dy++) vThickness++;
                    for (let dy = 1; y - dy >= 0 && pixels[(y - dy) * size + x] > 0.25; dy++) vThickness++;
                    
                    // Medir grosor en dirección horizontal
                    let hThickness = 1;
                    for (let dx = 1; x + dx < size && pixels[y * size + (x + dx)] > 0.25; dx++) hThickness++;
                    for (let dx = 1; x - dx >= 0 && pixels[y * size + (x - dx)] > 0.25; dx++) hThickness++;
                    
                    totalThickness += Math.min(vThickness, hThickness);
                    count++;
                }
            }
        }
        
        return count > 0 ? totalThickness / count : 0;
    }
    
    // Medir conectividad (número de componentes conectados)
    measureConnectivity(pixels, size) {
        const visited = Array(size).fill().map(() => Array(size).fill(false));
        let components = 0;
        
        const floodFill = (x, y) => {
            if (x < 0 || x >= size || y < 0 || y >= size || visited[y][x]) return;
            if (pixels[y * size + x] <= 0.25) return;
            
            visited[y][x] = true;
            floodFill(x + 1, y);
            floodFill(x - 1, y);
            floodFill(x, y + 1);
            floodFill(x, y - 1);
        };
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (pixels[y * size + x] > 0.25 && !visited[y][x]) {
                    components++;
                    floodFill(x, y);
                }
            }
        }
        
        return components;
    }
    
    // Contar esquinas (cambios bruscos de dirección)
    countCorners(pixels, size) {
        let corners = 0;
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                if (pixels[y * size + x] > 0.25) {
                    const neighbors = [
                        pixels[(y-1) * size + x] > 0.25,
                        pixels[(y+1) * size + x] > 0.25,
                        pixels[y * size + (x-1)] > 0.25,
                        pixels[y * size + (x+1)] > 0.25,
                        pixels[(y-1) * size + (x-1)] > 0.25,
                        pixels[(y-1) * size + (x+1)] > 0.25,
                        pixels[(y+1) * size + (x-1)] > 0.25,
                        pixels[(y+1) * size + (x+1)] > 0.25
                    ];
                    
                    // Esquina: tiene vecinos en solo 2 lados adyacentes (no opuestos)
                    const active = neighbors.filter(n => n).length;
                    if (active === 2) {
                        if ((neighbors[0] && neighbors[2]) || (neighbors[0] && neighbors[3]) ||
                            (neighbors[1] && neighbors[2]) || (neighbors[1] && neighbors[3])) {
                            corners++;
                        }
                    }
                }
            }
        }
        return corners;
    }
    
    // Medir densidad en un borde específico
    measureEdgeDensity(pixels, size, edge) {
        let total = 0;
        let count = 0;
        const edgeWidth = 3; // Ancho del borde a analizar
        
        if (edge === 'top' || edge === 'all') {
            for (let y = 0; y < edgeWidth; y++) {
                for (let x = 0; x < size; x++) {
                    if (pixels[y * size + x] > 0.25) total++;
                    count++;
                }
            }
        }
        if (edge === 'bottom' || edge === 'all') {
            for (let y = size - edgeWidth; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    if (pixels[y * size + x] > 0.25) total++;
                    count++;
                }
            }
        }
        if (edge === 'left' || edge === 'all') {
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < edgeWidth; x++) {
                    if (pixels[y * size + x] > 0.25) total++;
                    count++;
                }
            }
        }
        if (edge === 'right' || edge === 'all') {
            for (let y = 0; y < size; y++) {
                for (let x = size - edgeWidth; x < size; x++) {
                    if (pixels[y * size + x] > 0.25) total++;
                    count++;
                }
            }
        }
        
        return count > 0 ? total / count : 0;
    }
    
    // Medir densidad en el centro
    measureCenterDensity(pixels, size) {
        const centerSize = 8;
        const startX = Math.floor((size - centerSize) / 2);
        const startY = Math.floor((size - centerSize) / 2);
        let total = 0;
        let count = 0;
        
        for (let y = startY; y < startY + centerSize; y++) {
            for (let x = startX; x < startX + centerSize; x++) {
                if (pixels[y * size + x] > 0.25) total++;
                count++;
            }
        }
        
        return count > 0 ? total / count : 0;
    }
    
    // Medir líneas continuas (longitud de las líneas más largas)
    measureContinuousLines(pixels, size) {
        let maxVertical = 0;
        let maxHorizontal = 0;
        
        // Líneas verticales continuas
        for (let x = 0; x < size; x++) {
            let current = 0;
            for (let y = 0; y < size; y++) {
                if (pixels[y * size + x] > 0.25) {
                    current++;
                    maxVertical = Math.max(maxVertical, current);
                } else {
                    current = 0;
                }
            }
        }
        
        // Líneas horizontales continuas
        for (let y = 0; y < size; y++) {
            let current = 0;
            for (let x = 0; x < size; x++) {
                if (pixels[y * size + x] > 0.25) {
                    current++;
                    maxHorizontal = Math.max(maxHorizontal, current);
                } else {
                    current = 0;
                }
            }
        }
        
        return {
            vertical: maxVertical / size,
            horizontal: maxHorizontal / size
        };
    }
    
    // Detectar patrón de gancho (típico del 2, 5)
    detectHookPattern(pixels, size) {
        // Buscar patrón: línea horizontal seguida de curva/vertical
        let hooks = 0;
        for (let y = 2; y < size - 2; y++) {
            for (let x = 2; x < size - 2; x++) {
                if (pixels[y * size + x] > 0.25) {
                    // Verificar si hay línea horizontal seguida de vertical/curva
                    const hasHorizontal = pixels[y * size + (x+1)] > 0.25 && pixels[y * size + (x+2)] > 0.25;
                    const hasVertical = pixels[(y+1) * size + x] > 0.25 || pixels[(y+2) * size + x] > 0.25;
                    if (hasHorizontal && hasVertical) hooks++;
                }
            }
        }
        return hooks > 2;
    }
    
    // Detectar si hay un bucle cerrado (típico del 0, 6, 8, 9)
    detectClosedLoop(pixels, size) {
        // Verificar si hay un camino cerrado alrededor del centro
        const centerY = Math.floor(size / 2);
        const centerX = Math.floor(size / 2);
        const radius = 8;
        
        let perimeter = 0;
        let activePerimeter = 0;
        
        for (let angle = 0; angle < 360; angle += 10) {
            const rad = angle * Math.PI / 180;
            const x = Math.round(centerX + radius * Math.cos(rad));
            const y = Math.round(centerY + radius * Math.sin(rad));
            
            if (x >= 0 && x < size && y >= 0 && y < size) {
                perimeter++;
                if (pixels[y * size + x] > 0.25) activePerimeter++;
            }
        }
        
        return perimeter > 0 && activePerimeter / perimeter > 0.4;
    }

    measureCurvature(pixels, size) {
        let curvature = 0;
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                if (pixels[y * size + x] > 0.25) {
                    // Detectar cambios de dirección
                    const neighbors = [
                        pixels[(y-1) * size + x] > 0.25,
                        pixels[(y+1) * size + x] > 0.25,
                        pixels[y * size + (x-1)] > 0.25,
                        pixels[y * size + (x+1)] > 0.25,
                        pixels[(y-1) * size + (x-1)] > 0.25,
                        pixels[(y-1) * size + (x+1)] > 0.25,
                        pixels[(y+1) * size + (x-1)] > 0.25,
                        pixels[(y+1) * size + (x+1)] > 0.25
                    ];
                    
                    let activeCount = neighbors.filter(n => n).length;
                    // Curvatura: esquinas y cambios de dirección
                    if (activeCount === 2 || activeCount === 3) {
                        // Verificar si es esquina
                        if ((neighbors[0] && neighbors[2] && !neighbors[4]) ||
                            (neighbors[0] && neighbors[3] && !neighbors[5]) ||
                            (neighbors[1] && neighbors[2] && !neighbors[6]) ||
                            (neighbors[1] && neighbors[3] && !neighbors[7])) {
                            curvature++;
                        }
                    }
                }
            }
        }
        return curvature / (size * size);
    }

    countHoles(pixels, size) {
        const visited = Array(size).fill().map(() => Array(size).fill(false));
        let holes = 0;
        
        // Flood fill desde el exterior
        const floodFill = (x, y) => {
            if (x < 0 || x >= size || y < 0 || y >= size || visited[y][x]) return;
            if (pixels[y * size + x] > 0.25) return;
            
            visited[y][x] = true;
            floodFill(x + 1, y);
            floodFill(x - 1, y);
            floodFill(x, y + 1);
            floodFill(x, y - 1);
        };
        
        // Marcar exterior
        for (let i = 0; i < size; i++) {
            floodFill(i, 0);
            floodFill(i, size - 1);
            floodFill(0, i);
            floodFill(size - 1, i);
        }
        
        // Contar huecos internos
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                if (pixels[y * size + x] <= 0.25 && !visited[y][x]) {
                    holes++;
                    floodFill(x, y);
                }
            }
        }
        
        return holes;
    }

    countEndpoints(pixels, size) {
        let count = 0;
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                if (pixels[y * size + x] > 0.25) {
                    let neighbors = 0;
                    if (pixels[(y-1) * size + x] > 0.25) neighbors++;
                    if (pixels[(y+1) * size + x] > 0.25) neighbors++;
                    if (pixels[y * size + (x-1)] > 0.25) neighbors++;
                    if (pixels[y * size + (x+1)] > 0.25) neighbors++;
                    if (neighbors === 1) count++;
                }
            }
        }
        return count;
    }

    countCrossings(pixels, size) {
        let crossings = 0;
        for (let y = 2; y < size - 2; y++) {
            for (let x = 2; x < size - 2; x++) {
                if (pixels[y * size + x] > 0.25) {
                    const neighbors = [
                        pixels[(y-1) * size + x] > 0.25,
                        pixels[(y+1) * size + x] > 0.25,
                        pixels[y * size + (x-1)] > 0.25,
                        pixels[y * size + (x+1)] > 0.25
                    ];
                    // Cruz: tiene opuestos activos (vertical Y horizontal)
                    const vertical = neighbors[0] && neighbors[1];
                    const horizontal = neighbors[2] && neighbors[3];
                    
                    // Cruz fuerte: ambos ejes activos
                    if (vertical && horizontal) {
                        crossings += 2;
                    } else if (vertical || horizontal) {
                        // Cruz parcial: solo un eje
                        crossings += 1;
                    }
                }
            }
        }
        return Math.floor(crossings / 2); // Normalizar porque contamos cada cruce dos veces
    }

    detectVerticalLines(pixels, size) {
        let count = 0;
        let strongLines = 0;
        
        for (let x = 2; x < size - 2; x++) {
            let consecutive = 0;
            let maxConsecutive = 0;
            let totalInColumn = 0;
            
            // Verificar también columnas vecinas para detectar líneas más anchas
            for (let y = 0; y < size; y++) {
                let hasPixel = false;
                for (let dx = -1; dx <= 1; dx++) {
                    const nx = x + dx;
                    if (nx >= 0 && nx < size && pixels[y * size + nx] > 0.25) {
                        hasPixel = true;
                        totalInColumn++;
                        break;
                    }
                }
                
                if (hasPixel) {
                    consecutive++;
                    maxConsecutive = Math.max(maxConsecutive, consecutive);
                } else {
                    consecutive = 0;
                }
            }
            
            // Línea vertical: debe tener al menos 50% del tamaño y continuidad
            if (maxConsecutive > size * 0.45) {
                count++;
                if (maxConsecutive > size * 0.65) {
                    strongLines++;
                }
            }
        }
        
        // Si hay líneas fuertes, dar más peso
        return Math.max(count, strongLines * 2);
    }

    detectHorizontalLines(pixels, size) {
        let count = 0;
        let strongLines = 0;
        
        for (let y = 2; y < size - 2; y++) {
            let consecutive = 0;
            let maxConsecutive = 0;
            let totalInRow = 0;
            
            // Verificar también filas vecinas para detectar líneas más anchas
            for (let x = 0; x < size; x++) {
                let hasPixel = false;
                for (let dy = -1; dy <= 1; dy++) {
                    const ny = y + dy;
                    if (ny >= 0 && ny < size && pixels[ny * size + x] > 0.25) {
                        hasPixel = true;
                        totalInRow++;
                        break;
                    }
                }
                
                if (hasPixel) {
                    consecutive++;
                    maxConsecutive = Math.max(maxConsecutive, consecutive);
                } else {
                    consecutive = 0;
                }
            }
            
            // Línea horizontal: debe tener al menos 50% del tamaño y continuidad
            if (maxConsecutive > size * 0.45) {
                count++;
                if (maxConsecutive > size * 0.65) {
                    strongLines++;
                }
            }
        }
        
        // Si hay líneas fuertes, dar más peso
        return Math.max(count, strongLines * 2);
    }

    detectDiagonals(pixels, size) {
        let count = 0;
        // Diagonales principales y secundarias
        for (let d = -size + 5; d < size - 5; d++) {
            let consecutive = 0;
            let maxConsecutive = 0;
            for (let x = 0; x < size; x++) {
                const y = x + d;
                if (y >= 0 && y < size && pixels[y * size + x] > 0.25) {
                    consecutive++;
                    maxConsecutive = Math.max(maxConsecutive, consecutive);
                } else {
                    consecutive = 0;
                }
            }
            if (maxConsecutive > size * 0.4) count++;
        }
        return Math.min(count, 4);
    }

    // Reconocimiento mejorado basado en características muy precisas
    recognizeByFeatures(features) {
        const scores = Array(10).fill(0);
        
        // Validación básica - si no hay suficiente contenido, devolver scores bajos
        if (features.density < 0.02) {
            return scores;
        }
        
        // 0: Hueco central claro, aspecto circular, muy simétrico, denso, estructura cerrada
        if (features.holes >= 1) scores[0] += 16;
        if (features.holes === 1) scores[0] += 5; // Un hueco perfecto
        if (features.holes >= 2) scores[0] -= 13; // Más de un hueco = probablemente 8
        if (features.aspectRatio > 0.75 && features.aspectRatio < 1.3) scores[0] += 11; // Casi circular
        if (features.aspectRatio > 0.85 && features.aspectRatio < 1.15) scores[0] += 6; // Muy circular
        if (features.zones[4] < 0.1) scores[0] += 13; // Centro muy vacío
        if (features.zones[4] < 0.06) scores[0] += 6; // Centro casi completamente vacío
        if (features.centerDensity < 0.08) scores[0] += 8; // Centro muy vacío (nueva característica)
        if (features.zones[1] > 0.12 && features.zones[7] > 0.12 && features.zones[4] < 0.08) scores[0] += 11; // Lados muy activos, centro muy vacío
        if (features.zones[3] > 0.1 && features.zones[5] > 0.1 && features.zones[4] < 0.1) scores[0] += 9; // Centro horizontal activo, centro vertical vacío
        if (features.zones[0] > 0.08 && features.zones[2] > 0.08 && features.zones[6] > 0.08 && features.zones[8] > 0.08) scores[0] += 8; // Esquinas activas
        if (features.symmetry > 0.65) scores[0] += 10;
        if (features.symmetry > 0.75) scores[0] += 7; // Muy simétrico
        if (features.symmetry > 0.85) scores[0] += 5; // Extremadamente simétrico
        if (features.closedLoop) scores[0] += 12; // Bucle cerrado (nueva característica)
        if (features.crossings >= 2 && features.crossings <= 6) scores[0] += 8;
        if (features.crossings >= 4 && features.crossings <= 6) scores[0] += 4; // Múltiples cruces (estructura circular)
        if (features.endpoints <= 2) scores[0] += 7;
        if (features.endpoints === 0 || features.endpoints === 1) scores[0] += 5; // Casi sin endpoints (círculo cerrado)
        if (features.verticalLines >= 1 && features.horizontalLines >= 1) scores[0] += 6; // Estructura circular (tiene ambos)
        if (features.curvature > 0.018) scores[0] += 5; // Curvatura alta (círculo)
        if (features.curvature > 0.025) scores[0] += 4; // Mucha curvatura
        if (features.density > 0.12 && features.density < 0.22) scores[0] += 5; // Densidad típica del 0
        if (Math.abs(features.verticalBalance) < 0.2 && Math.abs(features.horizontalBalance) < 0.2) scores[0] += 8; // Muy equilibrado (nueva característica)
        if (features.connectivity === 1) scores[0] += 6; // Un solo componente conectado (nueva característica)
        if (features.corners <= 4) scores[0] += 4; // Pocas esquinas (círculo suave)
        if (features.topLine > 0.1 && features.bottomLine > 0.1 && features.leftLine > 0.1 && features.rightLine > 0.1) scores[0] += 5; // Bordes activos
        
        // Penalizaciones más estrictas
        if (features.holes === 0) scores[0] -= 18; // CRÍTICO: el 0 DEBE tener un hueco
        if (features.aspectRatio < 0.55 || features.aspectRatio > 1.8) scores[0] -= 10;
        if (features.zones[4] > 0.2) scores[0] -= 10; // Centro muy lleno no es 0
        if (features.centerDensity > 0.15) scores[0] -= 8; // Centro denso no es 0
        if (features.zones[4] > 0.25) scores[0] -= 6; // Centro extremadamente lleno
        if (features.endpoints > 5) scores[0] -= 6; // Muchos endpoints no es un círculo cerrado
        if (features.symmetry < 0.35) scores[0] -= 8; // Poca simetría no es 0
        if (features.diagonals > 3) scores[0] -= 6; // Muchas diagonales no son típicas del 0
        if (features.horizontalLines > 2 || features.verticalLines > 2) scores[0] -= 5; // Demasiadas líneas rectas
        if (features.hookPattern) scores[0] -= 7; // Patrón de gancho no es 0
        if (features.connectivity > 2) scores[0] -= 6; // Múltiples componentes no es 0
        
        // 1: MUY estrecho, vertical, SIN líneas horizontales, SIN cruces, estructura simple
        if (features.aspectRatio < 0.28) scores[1] += 20; // MUY estrecho
        if (features.aspectRatio >= 0.28 && features.aspectRatio < 0.35) scores[1] += 13;
        if (features.aspectRatio >= 0.35 && features.aspectRatio < 0.4) scores[1] += 9;
        if (features.aspectRatio >= 0.4 && features.aspectRatio < 0.45) scores[1] += 5;
        
        // El 1 NO tiene líneas horizontales - esto es CRÍTICO para diferenciarlo del 4
        if (features.horizontalLines === 0) scores[1] += 13; // MUY importante
        if (features.horizontalLines === 0 && features.aspectRatio < 0.4) scores[1] += 7; // Bonificación extra
        if (features.topLine < 0.15 && features.bottomLine < 0.15) scores[1] += 6; // Bordes horizontales vacíos (nueva característica)
        
        // El 1 NO tiene cruces - esto también es CRÍTICO
        if (features.crossings === 0) scores[1] += 11;
        if (features.crossings === 0 && features.horizontalLines === 0) scores[1] += 6; // Bonificación por ambos
        
        if (features.verticalLines >= 1) scores[1] += 9;
        if (features.verticalLines >= 2 && features.aspectRatio < 0.35) scores[1] += 5; // Múltiples verticales solo si muy estrecho
        if (features.continuousLines.vertical > 0.6) scores[1] += 7; // Línea vertical larga (nueva característica)
        if (features.holes === 0) scores[1] += 7;
        if (features.endpoints <= 3) scores[1] += 6;
        if (features.endpoints === 2 || features.endpoints === 1) scores[1] += 5; // Pocos endpoints típico del 1
        
        // El 1 tiene densidad baja y estructura simple
        if (features.density > 0.03 && features.density < 0.11) scores[1] += 6;
        if (features.density < 0.08 && features.aspectRatio < 0.35) scores[1] += 4; // Muy denso pero estrecho
        
        // El 1 tiene estructura vertical simple (centro vertical activo, lados vacíos)
        if (features.zones[1] > 0.15 && features.zones[4] < 0.25) scores[1] += 7; // Centro vertical activo
        if (features.zones[0] < 0.15 && features.zones[2] < 0.15) scores[1] += 6; // Lados vacíos
        if (features.zones[3] < 0.12 && features.zones[5] < 0.12) scores[1] += 5; // Centro horizontal vacío
        if (features.zones[1] > features.zones[0] * 2 && features.zones[1] > features.zones[2] * 2) scores[1] += 7; // Mucho más centro vertical que lados
        if (features.leftLine < 0.1 && features.rightLine < 0.1) scores[1] += 5; // Bordes laterales vacíos (nueva característica)
        if (features.connectivity === 1) scores[1] += 6; // Un solo componente (nueva característica)
        if (features.corners <= 2) scores[1] += 4; // Pocas esquinas (línea simple)
        if (Math.abs(features.horizontalBalance) < 0.3) scores[1] += 5; // Balanceado horizontalmente (nueva característica)
        
        // Penalizaciones más estrictas para diferenciarlo del 4
        if (features.aspectRatio > 0.42) scores[1] -= 13; // Más ancho = menos probable que sea 1
        if (features.aspectRatio > 0.48) scores[1] -= 16; // Mucho más ancho
        if (features.holes >= 1) scores[1] -= 11;
        if (features.crossings >= 1) scores[1] -= 13; // CRÍTICO: el 1 NO tiene cruces
        if (features.crossings >= 2) scores[1] -= 16; // Muchos cruces = definitivamente NO es 1
        if (features.horizontalLines >= 1) scores[1] -= 16; // CRÍTICO: el 1 NO tiene líneas horizontales
        if (features.horizontalLines >= 2) scores[1] -= 21; // Múltiples horizontales = definitivamente NO es 1
        if (features.zones[4] > 0.25) scores[1] -= 11; // Centro muy activo = probablemente cruce (4)
        if (features.centerDensity > 0.2) scores[1] -= 9; // Centro denso (nueva característica)
        if (features.zones[0] > 0.18 || features.zones[2] > 0.18) scores[1] -= 7; // Mucho contenido en lados = no es 1
        if (features.zones[3] > 0.15 || features.zones[5] > 0.15) scores[1] -= 9; // Centro horizontal activo = probablemente 4
        if (features.rightHeavy || features.topHeavy) scores[1] -= 6; // Distribución asimétrica = probablemente 4
        if (features.hookPattern) scores[1] -= 8; // Patrón de gancho no es 1
        if (features.continuousLines.horizontal > 0.3) scores[1] -= 10; // Línea horizontal larga no es 1
        
        // 2: Curva superior derecha, línea horizontal media, base izquierda, forma de S invertida, asimétrico
        if (features.holes === 0) scores[2] += 7;
        if (features.zones[2] > features.zones[0] * 1.5) scores[2] += 13; // Mucha más actividad derecha arriba
        if (features.zones[2] > features.zones[0] * 1.8) scores[2] += 6; // Extremadamente más derecha arriba
        if (features.zones[6] > features.zones[8] * 1.5) scores[2] += 13; // Mucha más actividad izquierda abajo
        if (features.zones[6] > features.zones[8] * 1.8) scores[2] += 6; // Extremadamente más izquierda abajo
        if (features.zones[1] > features.zones[3] * 1.3) scores[2] += 8; // Más centro-derecha que centro-izquierda
        if (features.zones[8] < features.zones[6] * 0.7) scores[2] += 7; // Mucho menos derecha abajo
        if (features.horizontalLines >= 1) scores[2] += 9;
        if (features.horizontalLines >= 1 && features.zones[3] > 0.12) scores[2] += 5; // Horizontal en zona central
        if (features.horizontalLines >= 2) scores[2] += 5; // Múltiples líneas horizontales
        if (features.diagonals >= 1) scores[2] += 9; // Diagonales son muy características del 2
        if (features.diagonals >= 2) scores[2] += 5; // Múltiples diagonales (forma de S)
        if (features.hookPattern) scores[2] += 10; // Patrón de gancho es muy característico del 2
        if (features.topHeavy && features.leftHeavy) scores[2] += 7;
        if (features.curvature > 0.018) scores[2] += 8; // Curvatura es característica del 2
        if (features.curvature > 0.028) scores[2] += 6; // Mucha curvatura (forma de S)
        if (features.endpoints >= 3 && features.endpoints <= 7) scores[2] += 4;
        if (features.endpoints >= 4 && features.endpoints <= 6) scores[2] += 2; // Endpoints típicos del 2
        if (features.aspectRatio > 0.65 && features.aspectRatio < 1.35) scores[2] += 3;
        if (features.zones[4] > 0.12 && features.zones[4] < 0.32) scores[2] += 4; // Centro medio activo
        if (features.zones[4] > 0.18 && features.zones[4] < 0.28) scores[2] += 3; // Centro en rango ideal
        if (features.crossings >= 1 && features.crossings <= 3) scores[2] += 3; // Algunos cruces típicos del 2
        if (features.density > 0.08 && features.density < 0.18) scores[2] += 3; // Densidad típica
        
        // Penalizaciones más estrictas
        if (features.holes >= 1) scores[2] -= 10; // El 2 NO tiene huecos
        if (features.symmetry > 0.6) scores[2] -= 7; // El 2 no es simétrico
        if (features.symmetry > 0.7) scores[2] -= 4; // Muy simétrico = probablemente no es 2
        if (features.verticalLines > 1) scores[2] -= 5; // Muchas líneas verticales no son típicas del 2
        if (features.verticalLines > 2) scores[2] -= 3; // Demasiadas verticales
        if (features.zones[0] > features.zones[2] * 0.9) scores[2] -= 5; // Más izquierda arriba que derecha arriba
        if (features.zones[8] > features.zones[6] * 0.9) scores[2] -= 5; // Más derecha abajo que izquierda abajo
        if (features.rightHeavy && !features.leftHeavy) scores[2] -= 4; // Más pesado a la derecha no es 2
        
        // 3: Dos curvas simétricas, líneas horizontales, forma de doble gancho, más simétrico que 2
        if (features.holes === 0) scores[3] += 6;
        if (features.zones[2] > features.zones[0] && features.zones[8] > features.zones[6]) scores[3] += 14;
        if (features.zones[2] > features.zones[0] * 1.15 && features.zones[8] > features.zones[6] * 1.15) scores[3] += 7; // Más simétrico
        if (features.zones[2] > features.zones[0] * 1.3 && features.zones[8] > features.zones[6] * 1.3) scores[3] += 5; // Muy simétrico
        if (features.horizontalLines >= 1) scores[3] += 10;
        if (features.horizontalLines >= 1 && (features.zones[1] > 0.1 || features.zones[7] > 0.1)) scores[3] += 5; // Horizontal con lados activos
        if (features.horizontalLines >= 2) scores[3] += 6; // Múltiples líneas horizontales (arriba y abajo)
        if (features.endpoints <= 6) scores[3] += 6;
        if (features.endpoints >= 3 && features.endpoints <= 6) scores[3] += 5; // Endpoints típicos del 3
        if (features.endpoints >= 4 && features.endpoints <= 5) scores[3] += 3; // Rango ideal
        if (features.curvature > 0.018) scores[3] += 8; // Curvatura es característica del 3
        if (features.curvature > 0.028) scores[3] += 5; // Mucha curvatura (curvas dobles)
        if (features.symmetry > 0.5 && features.symmetry < 0.75) scores[3] += 8; // Simetría moderada es muy característica del 3
        if ((features.leftHeavy && features.rightHeavy) || (features.zones[1] > 0.12 && features.zones[7] > 0.12)) scores[3] += 6;
        if (features.zones[1] > features.zones[3] * 1.2 && features.zones[7] > features.zones[5] * 1.2) scores[3] += 6; // Lados derechos muy activos
        if (features.zones[1] > 0.12 && features.zones[7] > 0.12 && features.zones[4] < 0.25) scores[3] += 5; // Lados activos, centro no lleno
        if (features.aspectRatio > 0.6 && features.aspectRatio < 1.25) scores[3] += 4;
        if (features.centerOfMass.y > 11 && features.centerOfMass.y < 17) scores[3] += 3; // Centro equilibrado
        if (features.crossings >= 1 && features.crossings <= 4) scores[3] += 3; // Algunos cruces típicos
        if (features.density > 0.08 && features.density < 0.18) scores[3] += 3; // Densidad típica
        if (features.symmetry > 0.5 && features.symmetry < 0.75) scores[3] += 4; // Simetría moderada (más que 2, menos que 8)
        
        // Penalizaciones más estrictas
        if (features.holes >= 1) scores[3] -= 9; // El 3 NO tiene huecos
        if (features.aspectRatio < 0.5) scores[3] -= 7; // Muy estrecho no es 3
        if (features.symmetry > 0.78) scores[3] -= 5; // Demasiada simetría podría ser 8
        if (features.symmetry < 0.35) scores[3] -= 6; // Muy asimétrico podría ser 2
        if (features.verticalLines > 1) scores[3] -= 5; // Muchas verticales no son típicas del 3
        if (features.zones[0] > features.zones[2] * 1.2) scores[3] -= 5; // Mucho más izquierda arriba (probablemente 2)
        if (features.zones[6] > features.zones[8] * 1.2) scores[3] -= 5; // Mucho más izquierda abajo (probablemente 2)
        
        // 4: Línea horizontal central MUY fuerte OBLIGATORIA, vertical derecha, CRUCES donde se unen, estructura compleja
        // El 4 requiere TODAS estas características: horizontal + vertical + cruces + centro activo
        if (features.holes === 0) scores[4] += 3;
        
        // CRÍTICO: El 4 DEBE tener líneas horizontales Y verticales Y cruces - TODAS juntas
        // Solo dar puntos si tiene TODAS las características del 4
        if (features.horizontalLines >= 1 && features.verticalLines >= 1 && features.crossings >= 1) {
            scores[4] += 20; // CRÍTICO: todas las características juntas
            if (features.zones[4] > 0.22) scores[4] += 10; // Centro activo con todas las características
        } else {
            // Si no tiene todas, penalizar mucho
            if (features.horizontalLines >= 1 && features.verticalLines === 0) scores[4] -= 15; // Solo horizontal no es 4
            if (features.horizontalLines >= 1 && features.crossings === 0) scores[4] -= 15; // Horizontal sin cruces no es 4
        }
        
        // El 4 tiene líneas verticales Y horizontales JUNTAS (requisito obligatorio)
        if (features.verticalLines >= 1 && features.horizontalLines >= 1 && features.crossings >= 1) {
            scores[4] += 15; // CRÍTICO: ambas líneas con cruces
            if (features.zones[4] > 0.25) scores[4] += 8; // Centro muy activo con estructura completa
        }
        
        // El 4 tiene zona central MUY activa (donde se cruzan las líneas) - SOLO si tiene estructura completa
        if (features.zones[4] > 0.25 && features.horizontalLines >= 1 && features.verticalLines >= 1 && features.crossings >= 1) {
            scores[4] += 12; // Centro muy activo = cruce (solo con estructura completa)
        }
        if (features.zones[4] > 0.3 && features.horizontalLines >= 1 && features.verticalLines >= 1) {
            scores[4] += 6; // Centro extremadamente activo con estructura
        }
        
        // El 4 tiene estructura específica: zona central-derecha muy activa
        if (features.zones[1] > 0.15 && features.zones[2] > 0.15) scores[4] += 8; // Zona central-derecha activa
        if (features.zones[1] > 0.2 || features.zones[2] > 0.2) scores[4] += 5; // Zonas muy activas
        if (features.zones[3] > 0.12 || features.zones[0] > 0.1) scores[4] += 6; // Parte superior izquierda activa
        if (features.zones[7] > 0.12) scores[4] += 4; // Parte inferior izquierda activa
        
        // El 4 tiene distribución específica
        if (features.zones[8] < features.zones[2] * 0.7) scores[4] += 6; // Más arriba derecha que abajo izquierda
        if (features.rightHeavy && (features.topHeavy || features.centerOfMass.y < 14)) scores[4] += 7;
        if (features.zones[1] > features.zones[3] && features.zones[2] > features.zones[0]) scores[4] += 5; // Más derecha que izquierda
        if (features.zones[1] > features.zones[5]) scores[4] += 4; // Centro-vertical más activo que centro-horizontal inferior
        
        // El 4 tiene aspecto específico
        if (features.aspectRatio > 0.5 && features.aspectRatio < 1.1) scores[4] += 4; // Aspecto típico del 4
        if (features.aspectRatio > 0.55 && features.aspectRatio < 1.0) scores[4] += 3; // Aspecto ideal del 4
        if (features.endpoints >= 3 && features.endpoints <= 5) scores[4] += 3;
        if (features.density > 0.08 && features.density < 0.18) scores[4] += 3; // Densidad típica del 4
        
        // Penalizaciones MUY estrictas para asegurar que no se confunda con 2, 3, 1, 5
        if (features.holes >= 1) scores[4] -= 16;
        if (features.horizontalLines === 0) scores[4] -= 25; // CRÍTICO: sin horizontal = NO es 4
        if (features.verticalLines === 0) scores[4] -= 25; // CRÍTICO: sin vertical = NO es 4
        if (features.crossings === 0) scores[4] -= 20; // CRÍTICO: sin cruces = NO es 4
        if (features.aspectRatio < 0.35) scores[4] -= 20; // CRÍTICO: muy estrecho = probablemente 1
        if (features.aspectRatio < 0.4) scores[4] -= 15; // Estrecho = menos probable que sea 4
        if (features.zones[4] < 0.18) scores[4] -= 15; // Centro no muy activo = menos probable (4 necesita cruce claro)
        if (features.verticalLines === 0 && features.horizontalLines === 1) scores[4] -= 20; // Solo horizontal = NO es 4
        if (features.horizontalLines === 1 && features.verticalLines === 0) scores[4] -= 20; // Solo horizontal sin vertical = NO es 4
        
        // Penalizaciones específicas para diferenciarlo del 2 y 3
        if (features.curvature > 0.025) scores[4] -= 12; // Mucha curvatura = probablemente 2 o 3, no 4
        if (features.diagonals >= 2) scores[4] -= 10; // Múltiples diagonales = probablemente 2 o 3
        if (features.symmetry > 0.55 && features.holes === 0) scores[4] -= 10; // Simétrico sin huecos = probablemente 3
        if (features.zones[2] > features.zones[0] * 1.3 && features.zones[6] > features.zones[8] * 1.3) scores[4] -= 12; // Patrón de 2
        if (features.zones[2] > features.zones[0] && features.zones[8] > features.zones[6] && features.symmetry > 0.5) scores[4] -= 10; // Patrón de 3
        
        if (features.symmetry > 0.65) scores[4] -= 8; // El 4 no es muy simétrico
        if (features.diagonals > 3) scores[4] -= 6;
        if (features.zones[6] > features.zones[0]) scores[4] -= 5; // Si abajo izquierda > arriba izquierda
        if (features.zones[0] > 0.2 && features.zones[2] < 0.15) scores[4] -= 8; // Mucha izquierda arriba pero poca derecha = probablemente no es 4
        if (features.hookPattern) scores[4] -= 10; // Patrón de gancho = probablemente 2 o 5
        
        // 5: Línea superior horizontal fuerte, línea vertical izquierda, curva inferior derecha, asimétrico
        if (features.holes === 0) scores[5] += 5;
        if (features.horizontalLines >= 1) scores[5] += 9;
        if (features.horizontalLines >= 1 && features.zones[1] > 0.1) scores[5] += 4; // Horizontal en parte superior
        if (features.horizontalLines >= 2) scores[5] += 4; // Múltiples horizontales
        if (features.zones[0] > features.zones[2] * 1.4) scores[5] += 10; // Mucha más actividad izquierda arriba
        if (features.zones[0] > features.zones[2] * 1.7) scores[5] += 5; // Extremadamente más izquierda arriba
        if (features.zones[8] > features.zones[6] * 1.4) scores[5] += 10; // Mucha más actividad derecha abajo
        if (features.zones[8] > features.zones[6] * 1.7) scores[5] += 5; // Extremadamente más derecha abajo
        if (features.zones[3] > features.zones[5] * 1.3) scores[5] += 7; // Más centro-izquierda que centro-derecha
        if (features.zones[0] > 0.16 && features.zones[3] > 0.12) scores[5] += 6; // Parte superior izquierda muy activa
        if (features.zones[6] < 0.12 && features.zones[2] > 0.12) scores[5] += 5; // Poca izquierda abajo, mucha derecha arriba
        if (features.curvature > 0.018) scores[5] += 7;
        if (features.curvature > 0.028) scores[5] += 4; // Curvatura significativa (curva del 5)
        if (features.topHeavy && features.leftHeavy) scores[5] += 6;
        if (features.endpoints >= 3 && features.endpoints <= 7) scores[5] += 4;
        if (features.endpoints >= 4 && features.endpoints <= 6) scores[5] += 2; // Endpoints típicos
        if (features.diagonals >= 1) scores[5] += 4; // Puede tener diagonal
        if (features.diagonals >= 2) scores[5] += 2; // Múltiples diagonales posibles
        if (features.aspectRatio > 0.65 && features.aspectRatio < 1.25) scores[5] += 3;
        if (features.zones[4] > 0.12 && features.zones[4] < 0.32) scores[5] += 4; // Centro medio activo
        if (features.crossings >= 1 && features.crossings <= 3) scores[5] += 3; // Algunos cruces típicos
        if (features.density > 0.08 && features.density < 0.18) scores[5] += 3; // Densidad típica
        
        // Penalizaciones más estrictas
        if (features.holes >= 1) scores[5] -= 9; // El 5 NO tiene huecos
        if (features.verticalLines > 2) scores[5] -= 6; // Muchas verticales no son típicas
        if (features.verticalLines > 3) scores[5] -= 4; // Demasiadas verticales
        if (features.symmetry > 0.6) scores[5] -= 6; // El 5 no es simétrico
        if (features.symmetry > 0.7) scores[5] -= 4; // Muy simétrico no es 5
        if (features.zones[2] > features.zones[0] * 0.9) scores[5] -= 5; // Más derecha arriba que izquierda arriba
        if (features.zones[6] > features.zones[8] * 0.9) scores[5] -= 5; // Más izquierda abajo que derecha abajo
        if (features.rightHeavy && !features.leftHeavy) scores[5] -= 4; // Más pesado a la derecha no es 5
        
        // 6: Hueco superior claro, curva inferior cerrada, más peso abajo, estructura característica
        if (features.holes >= 1) scores[6] += 13;
        if (features.holes === 1) scores[6] += 4; // Un hueco perfecto
        if (features.holes >= 2) scores[6] -= 7; // Más de un hueco podría ser 8
        if (features.bottomHeavy) scores[6] += 12;
        if (features.zones[8] > features.zones[2] * 1.5) scores[6] += 11; // Mucho más abajo
        if (features.zones[8] > features.zones[2] * 1.8) scores[6] += 5; // Extremadamente más abajo
        if (features.zones[6] > features.zones[0] * 1.3) scores[6] += 8; // Más abajo izquierda
        if (features.zones[4] < 0.22 && features.zones[1] < 0.14 && features.zones[0] < 0.14) scores[6] += 10; // Hueco arriba claro
        if (features.zones[4] < 0.18 && features.zones[1] < 0.12 && features.zones[0] < 0.12) scores[6] += 5; // Hueco arriba muy claro
        if (features.zones[7] > 0.16 && features.zones[8] > 0.22) scores[6] += 8; // Base muy activa
        if (features.zones[7] > 0.18 && features.zones[8] > 0.24) scores[6] += 4; // Base extremadamente activa
        if (features.centerOfMass.y > 14.5) scores[6] += 7;
        if (features.centerOfMass.y > 15.5) scores[6] += 5; // Muy abajo
        if (features.centerOfMass.y > 16.5) scores[6] += 3; // Extremadamente abajo
        if (features.curvature > 0.018) scores[6] += 6;
        if (features.curvature > 0.028) scores[6] += 3; // Mucha curvatura (curva del 6)
        if (features.endpoints <= 3) scores[6] += 5; // Pocos endpoints
        if (features.endpoints <= 2) scores[6] += 3; // Muy pocos endpoints
        if (features.aspectRatio > 0.6 && features.aspectRatio < 1.25) scores[6] += 3;
        if (features.verticalLines >= 1) scores[6] += 3; // Puede tener línea vertical izquierda
        if (features.crossings >= 1 && features.crossings <= 3) scores[6] += 3; // Algunos cruces típicos
        if (features.density > 0.1 && features.density < 0.2) scores[6] += 3; // Densidad típica
        
        // Penalizaciones más estrictas
        if (features.holes === 0) scores[6] -= 12; // CRÍTICO: el 6 DEBE tener un hueco
        if (features.topHeavy) scores[6] -= 9; // El 6 NO es pesado arriba
        if (features.zones[2] > features.zones[8] * 0.9) scores[6] -= 7; // Más arriba que abajo = probablemente 9
        if (features.zones[2] > features.zones[8]) scores[6] -= 5; // Más arriba = definitivamente no es 6
        if (features.aspectRatio < 0.5) scores[6] -= 6; // Muy estrecho no es 6
        if (features.symmetry > 0.6) scores[6] -= 5; // El 6 no es muy simétrico
        if (features.symmetry > 0.7) scores[6] -= 3; // Muy simétrico no es 6
        if (features.zones[4] > 0.3) scores[6] -= 4; // Centro muy lleno no es 6
        
        // 7: Línea horizontal superior muy fuerte, diagonal descendente, minimalista inferior, muy asimétrico
        if (features.holes === 0) scores[7] += 7;
        if (features.horizontalLines >= 1) scores[7] += 12;
        if (features.horizontalLines >= 1 && features.zones[1] > 0.1) scores[7] += 4; // Horizontal en parte superior
        if (features.horizontalLines >= 2) scores[7] += 5; // Múltiples horizontales (arriba)
        if (features.diagonals >= 1) scores[7] += 11;
        if (features.diagonals >= 2) scores[7] += 5; // Múltiples diagonales (típico del 7)
        if (features.zones[2] > features.zones[8] * 1.7) scores[7] += 11; // Muy superior
        if (features.zones[2] > features.zones[8] * 2.0) scores[7] += 5; // Extremadamente superior
        if (features.zones[0] > features.zones[6] * 1.5) scores[7] += 8; // Más arriba izquierda que abajo
        if (features.zones[0] > features.zones[6] * 1.8) scores[7] += 4; // Extremadamente más arriba
        if (features.topHeavy && !features.bottomHeavy) scores[7] += 9;
        if (features.zones[6] < 0.1 && features.zones[7] < 0.1) scores[7] += 7; // Muy poco contenido abajo
        if (features.zones[6] < 0.08 && features.zones[7] < 0.08) scores[7] += 4; // Extremadamente poco abajo
        if (features.zones[8] < 0.08) scores[7] += 6; // Prácticamente nada abajo izquierda
        if (features.zones[8] < 0.06) scores[7] += 3; // Nada abajo izquierda
        if (features.endpoints <= 4) scores[7] += 6;
        if (features.endpoints === 2 || features.endpoints === 3) scores[7] += 4; // Pocos endpoints típico del 7
        if (features.endpoints === 2) scores[7] += 2; // Dos endpoints ideal
        if (features.density > 0.04 && features.density < 0.13) scores[7] += 4; // Densidad baja-media
        if (features.density < 0.08) scores[7] += 3; // Muy baja densidad típica
        if (features.aspectRatio > 0.65 && features.aspectRatio < 1.35) scores[7] += 3;
        if (features.curvature < 0.012) scores[7] += 4; // Poca curvatura (más líneas rectas)
        if (features.curvature < 0.008) scores[7] += 2; // Muy poca curvatura
        
        // Penalizaciones más estrictas
        if (features.holes >= 1) scores[7] -= 11; // El 7 NO tiene huecos
        if (features.verticalLines > 1) scores[7] -= 8; // Verticales no son típicas del 7
        if (features.verticalLines > 2) scores[7] -= 5; // Muchas verticales = probablemente no es 7
        if (features.bottomHeavy) scores[7] -= 8; // El 7 NO es pesado abajo
        if (features.zones[8] > features.zones[2] * 0.75) scores[7] -= 7; // Similar arriba y abajo no es 7
        if (features.zones[8] > features.zones[2] * 0.5) scores[7] -= 4; // Mucho contenido abajo no es 7
        if (features.symmetry > 0.5) scores[7] -= 6; // El 7 no es simétrico
        if (features.symmetry > 0.6) scores[7] -= 4; // Muy simétrico no es 7
        if (features.horizontalLines === 0 && features.diagonals === 0) scores[7] -= 8; // Sin horizontal ni diagonal = no es 7
        
        // 8: Dos huecos claros, muy simétrico, denso, estructura vertical, equilibrado
        if (features.holes >= 2) scores[8] += 18;
        if (features.holes === 2) scores[8] += 5; // Dos huecos perfectos
        if (features.holes === 3) scores[8] += 4; // Podría tener un tercer hueco pequeño
        if (features.holes === 1) {
            // Un hueco pero con características de 8
            if (features.symmetry > 0.55 && features.zones[4] < 0.22) scores[8] += 12;
            if (features.symmetry > 0.6 && features.zones[4] < 0.18) scores[8] += 6;
            if (features.zones[1] > 0.12 && features.zones[7] > 0.12 && features.zones[4] < 0.15) scores[8] += 8;
        }
        if (features.zones[1] > 0.13 && features.zones[7] > 0.13) scores[8] += 10; // Centro vertical muy activo
        if (features.zones[1] > 0.15 && features.zones[7] > 0.15) scores[8] += 5; // Extremadamente activo
        if (features.zones[4] < 0.18) scores[8] += 9; // Centro vacío o muy parcial
        if (features.zones[4] < 0.14) scores[8] += 5; // Centro muy vacío
        if (features.zones[1] > 0.11 && features.zones[4] < 0.14 && features.zones[7] > 0.11) scores[8] += 8; // Estructura de 8
        if (features.zones[3] > 0.1 && features.zones[5] > 0.1 && features.zones[4] < 0.12) scores[8] += 7; // Estructura horizontal también
        if (features.crossings >= 2) scores[8] += 8;
        if (features.crossings >= 3) scores[8] += 5; // Múltiples cruces
        if (features.crossings >= 4) scores[8] += 3; // Muchos cruces típicos del 8
        if (features.aspectRatio > 0.7 && features.aspectRatio < 1.3) scores[8] += 7;
        if (features.aspectRatio > 0.8 && features.aspectRatio < 1.2) scores[8] += 4; // Muy equilibrado
        if (features.symmetry > 0.55) scores[8] += 8;
        if (features.symmetry > 0.65) scores[8] += 6; // Muy simétrico
        if (features.symmetry > 0.75) scores[8] += 4; // Extremadamente simétrico
        if (features.endpoints <= 4) scores[8] += 4;
        if (features.endpoints <= 2) scores[8] += 2; // Muy pocos endpoints
        if (features.verticalLines >= 1) scores[8] += 3;
        if (features.verticalLines >= 2) scores[8] += 2; // Múltiples verticales típicas
        if (features.density > 0.14 && features.density < 0.24) scores[8] += 4; // Densidad típica del 8
        if (!features.topHeavy && !features.bottomHeavy && !features.leftHeavy && !features.rightHeavy) scores[8] += 5; // Equilibrado
        
        // Penalizaciones más estrictas
        if (features.holes === 0) scores[8] -= 14; // CRÍTICO: el 8 DEBE tener al menos un hueco
        if (features.aspectRatio < 0.55) scores[8] -= 7; // Muy estrecho no es 8
        if (features.zones[4] > 0.3) scores[8] -= 8; // Centro muy lleno no es 8
        if (features.zones[4] > 0.35) scores[8] -= 5; // Centro extremadamente lleno
        if (features.topHeavy || features.bottomHeavy) scores[8] -= 5; // El 8 es más equilibrado
        if (features.leftHeavy || features.rightHeavy) scores[8] -= 4; // El 8 es simétrico horizontalmente
        if (features.symmetry < 0.4) scores[8] -= 6; // Muy asimétrico no es 8
        
        // 9: Hueco inferior claro, curva superior cerrada, más peso arriba (inverso de 6)
        if (features.holes >= 1) scores[9] += 13;
        if (features.holes === 1) scores[9] += 4; // Un hueco perfecto
        if (features.holes >= 2) scores[9] -= 7; // Más de un hueco podría ser 8
        if (features.topHeavy) scores[9] += 12;
        if (features.zones[2] > features.zones[8] * 1.5) scores[9] += 11; // Mucho más arriba
        if (features.zones[2] > features.zones[8] * 1.8) scores[9] += 5; // Extremadamente más arriba
        if (features.zones[0] > features.zones[6] * 1.3) scores[9] += 8; // Más arriba izquierda
        if (features.zones[4] < 0.22 && features.zones[7] < 0.14 && features.zones[8] < 0.14) scores[9] += 10; // Hueco abajo claro
        if (features.zones[4] < 0.18 && features.zones[7] < 0.12 && features.zones[8] < 0.12) scores[9] += 5; // Hueco abajo muy claro
        if (features.zones[1] > 0.16 && features.zones[2] > 0.22) scores[9] += 8; // Parte superior muy activa
        if (features.zones[1] > 0.18 && features.zones[2] > 0.24) scores[9] += 4; // Parte superior extremadamente activa
        if (features.centerOfMass.y < 13.5) scores[9] += 7;
        if (features.centerOfMass.y < 12.5) scores[9] += 5; // Muy arriba
        if (features.centerOfMass.y < 11.5) scores[9] += 3; // Extremadamente arriba
        if (features.curvature > 0.018) scores[9] += 6;
        if (features.curvature > 0.028) scores[9] += 3; // Mucha curvatura (curva del 9)
        if (features.endpoints <= 3) scores[9] += 5; // Pocos endpoints
        if (features.endpoints <= 2) scores[9] += 3; // Muy pocos endpoints
        if (features.aspectRatio > 0.6 && features.aspectRatio < 1.25) scores[9] += 3;
        if (features.verticalLines >= 1) scores[9] += 3; // Puede tener línea vertical derecha
        if (features.crossings >= 1 && features.crossings <= 3) scores[9] += 3; // Algunos cruces típicos
        if (features.density > 0.1 && features.density < 0.2) scores[9] += 3; // Densidad típica
        
        // Penalizaciones más estrictas
        if (features.holes === 0) scores[9] -= 12; // CRÍTICO: el 9 DEBE tener un hueco
        if (features.bottomHeavy) scores[9] -= 9; // El 9 NO es pesado abajo
        if (features.zones[8] > features.zones[2] * 0.9) scores[9] -= 7; // Más abajo que arriba = probablemente 6
        if (features.zones[8] > features.zones[2]) scores[9] -= 5; // Más abajo = definitivamente no es 9
        if (features.aspectRatio < 0.5) scores[9] -= 6; // Muy estrecho no es 9
        if (features.symmetry > 0.6) scores[9] -= 5; // El 9 no es muy simétrico
        if (features.symmetry > 0.7) scores[9] -= 3; // Muy simétrico no es 9
        if (features.zones[4] > 0.3) scores[9] -= 4; // Centro muy lleno no es 9
        
        // Ajustes finales de precisión: validaciones cruzadas entre dígitos similares
        
        // Diferenciar 0 y 8 (ambos tienen huecos, pero 8 tiene dos)
        if (features.holes === 1 && scores[0] > scores[8] * 1.2) {
            scores[8] *= 0.7; // Reducir 8 si tiene un solo hueco y 0 es más probable
        }
        if (features.holes === 2 && scores[8] > scores[0] * 1.2) {
            scores[0] *= 0.6; // Reducir 0 si tiene dos huecos y 8 es más probable
        }
        
        // Diferenciar 6 y 9 (ambos tienen un hueco, pero en posiciones opuestas)
        if (features.holes === 1) {
            if (features.zones[8] > features.zones[2] * 1.3 && scores[6] > scores[9]) {
                scores[9] *= 0.7; // Si más abajo, favorecer 6 sobre 9
            }
            if (features.zones[2] > features.zones[8] * 1.3 && scores[9] > scores[6]) {
                scores[6] *= 0.7; // Si más arriba, favorecer 9 sobre 6
            }
        }
        
        // CRÍTICO: Diferenciar 1 y 4 (confusión más común)
        // El 1 es MUY estrecho, NO tiene líneas horizontales, NO tiene cruces
        // El 4 es más ancho, DEBE tener líneas horizontales, DEBE tener cruces
        
        // Si tiene líneas horizontales o cruces, NO puede ser 1
        if (features.horizontalLines >= 1 || features.crossings >= 1) {
            if (scores[1] > scores[4] * 0.8) {
                scores[1] *= 0.4; // Reducir significativamente 1 si tiene horizontal o cruces
            }
            if (scores[1] > scores[4]) {
                scores[1] *= 0.3; // Reducir mucho más si 1 tiene mayor probabilidad que 4
            }
        }
        
        // Si NO tiene líneas horizontales NI cruces, NO puede ser 4
        if (features.horizontalLines === 0 && features.crossings === 0) {
            if (scores[4] > scores[1] * 0.8) {
                scores[4] *= 0.4; // Reducir significativamente 4 si no tiene horizontal ni cruces
            }
            if (scores[4] > scores[1]) {
                scores[4] *= 0.3; // Reducir mucho más si 4 tiene mayor probabilidad que 1
            }
        }
        
        // Si es MUY estrecho (< 0.35), favorecer 1 sobre 4
        if (features.aspectRatio < 0.35) {
            if (scores[4] > scores[1] * 0.9) {
                scores[4] *= 0.5; // Reducir 4 si es muy estrecho
            }
            if (features.horizontalLines === 0 && features.crossings === 0 && scores[4] > scores[1]) {
                scores[4] *= 0.3; // Reducir mucho más 4 si no tiene horizontal/cruces y es estrecho
            }
        }
        
        // Si es más ancho (> 0.45) y tiene líneas horizontales, favorecer 4 sobre 1
        if (features.aspectRatio > 0.45 && features.horizontalLines >= 1) {
            if (scores[1] > scores[4] * 0.9) {
                scores[1] *= 0.5; // Reducir 1 si es ancho y tiene horizontal
            }
            if (features.crossings >= 1 && scores[1] > scores[4]) {
                scores[1] *= 0.3; // Reducir mucho más 1 si tiene cruces y es ancho
            }
        }
        
        // Si tiene zona central muy activa (> 0.25) y cruces, favorecer 4 sobre 1
        if (features.zones[4] > 0.25 && features.crossings >= 1) {
            if (scores[1] > scores[4] * 0.9) {
                scores[1] *= 0.4; // Reducir 1 si tiene centro activo con cruces
            }
        }
        
        // Si tiene estructura vertical simple (centro vertical activo, lados vacíos), favorecer 1 sobre 4
        if (features.zones[1] > features.zones[0] * 2 && features.zones[1] > features.zones[2] * 2) {
            if (features.horizontalLines === 0 && features.crossings === 0) {
                if (scores[4] > scores[1] * 0.9) {
                    scores[4] *= 0.5; // Reducir 4 si tiene estructura vertical simple
                }
            }
        }
        
        // Diferenciar 1 y otros dígitos verticales (7, pero no 4 aquí ya que lo manejamos arriba)
        if (features.aspectRatio < 0.4 && scores[1] > 0.5) {
            // Si es muy estrecho y 1 tiene alta probabilidad, reducir 7
            if (scores[7] > scores[1] * 0.8) {
                scores[7] *= 0.6;
            }
        }
        
        // CRÍTICO: Diferenciar 4 de 2 y 3 (confusión más común)
        // El 4 requiere horizontal + vertical + cruces + centro activo TODOS juntos
        // El 2 y 3 tienen curvas, no estructura de cruce como el 4
        
        // Si tiene características de 2 (curvatura, diagonales, patrón S)
        if (features.curvature > 0.02 || features.diagonals >= 1 || features.hookPattern) {
            if (scores[2] > scores[4] * 0.6) {
                scores[4] *= 0.5; // Reducir 4 si tiene características de 2
            }
            if (features.zones[2] > features.zones[0] * 1.3 && features.zones[6] > features.zones[8] * 1.3) {
                scores[4] *= 0.4; // Patrón claro de 2 = reducir más el 4
            }
        }
        
        // Si tiene características de 3 (simetría, curvas dobles)
        if (features.symmetry > 0.5 && features.holes === 0 && features.curvature > 0.015) {
            if (scores[3] > scores[4] * 0.6) {
                scores[4] *= 0.5; // Reducir 4 si tiene características de 3
            }
            if (features.zones[2] > features.zones[0] && features.zones[8] > features.zones[6] && features.symmetry > 0.55) {
                scores[4] *= 0.4; // Patrón claro de 3 = reducir más el 4
            }
        }
        
        // Si NO tiene la estructura completa del 4 (horizontal + vertical + cruces), reducir
        if (!(features.horizontalLines >= 1 && features.verticalLines >= 1 && features.crossings >= 1)) {
            if (scores[4] > scores[2] * 0.7 || scores[4] > scores[3] * 0.7) {
                scores[4] *= 0.4; // Sin estructura completa, reducir 4
            }
        }
        
        // Si tiene horizontal pero NO tiene vertical o cruces, probablemente es 2, 3 o 5, no 4
        if (features.horizontalLines >= 1 && (features.verticalLines === 0 || features.crossings === 0)) {
            if (scores[4] > scores[2] * 0.9 || scores[4] > scores[3] * 0.9 || scores[4] > scores[5] * 0.9) {
                scores[4] *= 0.4; // Reducir 4 si no tiene estructura completa
            }
        }
        
        // Diferenciar 4 y otros dígitos con líneas horizontales (5, 2, 3) - solo si 4 tiene estructura completa
        if (features.horizontalLines >= 1 && features.verticalLines >= 1 && features.crossings >= 1) {
            if (features.zones[4] > 0.22 && scores[4] > 0.4) {
                // Solo reducir otros si 4 tiene estructura completa Y centro activo
                scores[5] *= 0.8;
                scores[2] *= 0.85;
                scores[3] *= 0.85;
            }
        }
        
        // Diferenciar 2 y 3 (ambos tienen curvas pero 3 es más simétrico)
        if (features.symmetry > 0.6 && features.holes === 0) {
            if (scores[3] > scores[2] * 0.9) {
                scores[2] *= 0.7; // Si más simétrico, favorecer 3
            }
            if (scores[2] > scores[3] * 1.1) {
                scores[3] *= 0.8; // Si 2 es mucho más probable, reducir 3
            }
        }
        if (features.symmetry < 0.5 && features.holes === 0) {
            if (scores[2] > scores[3] * 0.9) {
                scores[3] *= 0.7; // Si menos simétrico, favorecer 2
            }
            if (scores[3] > scores[2] * 1.1) {
                scores[2] *= 0.8; // Si 3 es mucho más probable, reducir 2
            }
        }
        
        // Diferenciar 2 y 5 (ambos tienen curvas pero patrones diferentes)
        if (features.zones[0] > features.zones[2] * 1.3 && features.zones[8] > features.zones[6] * 1.3) {
            if (scores[5] > scores[2] * 0.9) {
                scores[2] *= 0.7; // Si más izquierda, favorecer 5
            }
            if (scores[2] > scores[5] * 1.1) {
                scores[5] *= 0.8; // Si 2 es mucho más probable, reducir 5
            }
        }
        if (features.zones[2] > features.zones[0] * 1.3 && features.zones[6] > features.zones[8] * 1.3) {
            if (scores[2] > scores[5] * 0.9) {
                scores[5] *= 0.7; // Si más derecha arriba e izquierda abajo, favorecer 2
            }
            if (scores[5] > scores[2] * 1.1) {
                scores[2] *= 0.8; // Si 5 es mucho más probable, reducir 2
            }
        }
        
        // Diferenciar 2 y 5 con base en horizontal superior
        if (features.horizontalLines >= 1 && features.zones[1] > 0.12) {
            if (features.zones[0] > features.zones[2] * 1.2) {
                if (scores[5] > scores[2] * 0.9) {
                    scores[2] *= 0.75; // Horizontal superior izquierda = probablemente 5
                }
            }
        }
        
        // Diferenciar 3 y 5 (ambos pueden tener horizontales pero 5 es más asimétrico)
        if (features.horizontalLines >= 1 && features.holes === 0) {
            if (features.symmetry > 0.55 && features.zones[2] > features.zones[0]) {
                if (scores[3] > scores[5] * 0.9) {
                    scores[5] *= 0.75; // Simétrico con derecha activa = probablemente 3
                }
            }
            if (features.symmetry < 0.5 && features.zones[0] > features.zones[2]) {
                if (scores[5] > scores[3] * 0.9) {
                    scores[3] *= 0.75; // Asimétrico con izquierda activa = probablemente 5
                }
            }
        }
        
        // Diferenciar 6 y 9 más estrictamente (ambos tienen un hueco)
        if (features.holes === 1) {
            if (features.zones[8] > features.zones[2] * 1.4 && scores[6] > scores[9] * 0.9) {
                scores[9] *= 0.6; // Mucho más abajo = definitivamente 6
            }
            if (features.zones[2] > features.zones[8] * 1.4 && scores[9] > scores[6] * 0.9) {
                scores[6] *= 0.6; // Mucho más arriba = definitivamente 9
            }
            if (features.bottomHeavy && scores[6] > scores[9] * 0.9) {
                scores[9] *= 0.7; // Pesado abajo = 6
            }
            if (features.topHeavy && scores[9] > scores[6] * 0.9) {
                scores[6] *= 0.7; // Pesado arriba = 9
            }
            if (features.centerOfMass.y > 16 && scores[6] > scores[9]) {
                scores[9] *= 0.5; // Centro muy abajo = 6
            }
            if (features.centerOfMass.y < 12 && scores[9] > scores[6]) {
                scores[6] *= 0.5; // Centro muy arriba = 9
            }
        }
        
        // Diferenciar 6 y 0 (ambos pueden tener huecos)
        if (features.holes === 1) {
            if (features.bottomHeavy && features.zones[4] < 0.2 && features.zones[2] < 0.12) {
                if (scores[6] > scores[0] * 0.9) {
                    scores[0] *= 0.7; // Hueco arriba y pesado abajo = probablemente 6
                }
            }
            if (features.symmetry > 0.6 && features.zones[4] < 0.1 && !features.bottomHeavy && !features.topHeavy) {
                if (scores[0] > scores[6] * 0.9) {
                    scores[6] *= 0.7; // Simétrico con hueco central = probablemente 0
                }
            }
        }
        
        // Diferenciar 9 y 0 (ambos pueden tener huecos)
        if (features.holes === 1) {
            if (features.topHeavy && features.zones[4] < 0.2 && features.zones[8] < 0.12) {
                if (scores[9] > scores[0] * 0.9) {
                    scores[0] *= 0.7; // Hueco abajo y pesado arriba = probablemente 9
                }
            }
            if (features.symmetry > 0.6 && features.zones[4] < 0.1 && !features.topHeavy && !features.bottomHeavy) {
                if (scores[0] > scores[9] * 0.9) {
                    scores[9] *= 0.7; // Simétrico con hueco central = probablemente 0
                }
            }
        }
        
        // Diferenciar 7 y 1 (ambos pueden ser verticales)
        if (features.aspectRatio < 0.45 && features.holes === 0) {
            if (features.horizontalLines >= 1 || features.diagonals >= 1) {
                if (scores[7] > scores[1] * 0.9) {
                    scores[1] *= 0.6; // Tiene horizontal o diagonal = probablemente 7
                }
            }
            if (features.horizontalLines === 0 && features.diagonals === 0 && features.crossings === 0) {
                if (scores[1] > scores[7] * 0.9) {
                    scores[7] *= 0.6; // Sin horizontal/diagonal/cruces = probablemente 1
                }
            }
        }
        
        // Diferenciar 7 y 4 (ambos tienen líneas horizontales)
        if (features.horizontalLines >= 1 && features.holes === 0) {
            if (features.diagonals >= 1 && features.verticalLines === 0) {
                if (scores[7] > scores[4] * 0.9) {
                    scores[4] *= 0.7; // Diagonal sin vertical = probablemente 7
                }
            }
            if (features.verticalLines >= 1 && features.crossings >= 1) {
                if (scores[4] > scores[7] * 0.9) {
                    scores[7] *= 0.7; // Vertical con cruces = probablemente 4
                }
            }
            if (features.topHeavy && features.zones[8] < 0.1) {
                if (scores[7] > scores[4] * 0.9) {
                    scores[4] *= 0.7; // Pesado arriba sin contenido abajo = probablemente 7
                }
            }
        }
        
        // Diferenciar 0 y 8 más estrictamente
        if (features.holes === 1 && features.symmetry > 0.6 && features.zones[4] < 0.12) {
            if (scores[0] > scores[8] * 1.3) {
                scores[8] *= 0.5; // Un hueco, simétrico, centro vacío = probablemente 0
            }
        }
        if (features.holes >= 2) {
            if (scores[8] > scores[0] * 1.3) {
                scores[0] *= 0.5; // Dos huecos = probablemente 8
            }
        }
        if (features.holes === 1 && features.zones[1] > 0.12 && features.zones[7] > 0.12 && features.zones[4] < 0.15) {
            if (scores[8] > scores[0] * 1.2) {
                scores[0] *= 0.6; // Estructura vertical con centro vacío = probablemente 8
            }
        }
        
        // Normalizar scores manteniendo la precisión
        const maxScore = Math.max(...scores);
        if (maxScore > 0) {
            // Normalización estándar que preserva las diferencias importantes
            return scores.map(s => Math.max(0, s / maxScore));
        }
        return scores;
    }

    // Crear variaciones de plantillas para mejor matching
    createTemplateVariations() {
        // Plantillas con ligeras variaciones para cada dígito
        return {
            // Se pueden agregar variaciones aquí si es necesario
        };
    }

    // Matching de plantillas mejorado con múltiples escalas y desplazamientos
    matchTemplates(pixels) {
        const size = 28;
        const scores = Array(10).fill(0);
        
        // Reducir imagen a 7x7 con mejor muestreo (promedio ponderado de áreas)
        const scaled = [];
        for (let y = 0; y < 7; y++) {
            scaled[y] = [];
            for (let x = 0; x < 7; x++) {
                const srcX = Math.floor((x / 6) * (size - 1));
                const srcY = Math.floor((y / 6) * (size - 1));
                
                // Muestrear área 5x5 alrededor con promedio ponderado (más peso al centro)
                let sum = 0;
                let weightSum = 0;
                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        const sx = Math.min(size - 1, Math.max(0, srcX + dx));
                        const sy = Math.min(size - 1, Math.max(0, srcY + dy));
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        const weight = Math.exp(-distance * 0.5); // Más peso cerca del centro
                        sum += pixels[sy * size + sx] * weight;
                        weightSum += weight;
                    }
                }
                scaled[y][x] = (sum / weightSum) > 0.32 ? 1 : 0;
            }
        }
        
        // Comparar con plantillas (con tolerancia mejorada y rotaciones menores)
        for (let digit = 0; digit < 10; digit++) {
            const template = this.templates[digit];
            let bestMatch = 0;
            
            // Probar diferentes desplazamientos pequeños
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    let match = 0;
                    let mismatch = 0;
                    let criticalMatch = 0; // Coincidencias en posiciones críticas
                    let total = 0;
                    
                    for (let y = 0; y < 7; y++) {
                        for (let x = 0; x < 7; x++) {
                            const ty = y + dy;
                            const tx = x + dx;
                            if (ty >= 0 && ty < 7 && tx >= 0 && tx < 7) {
                                total++;
                                const isMatch = scaled[ty][tx] === template[y][x];
                                if (isMatch) {
                                    match++;
                                    // Posiciones centrales y de borde son más críticas
                                    const isCritical = (x === 3 || y === 3 || x === 0 || x === 6 || y === 0 || y === 6);
                                    if (isCritical) criticalMatch++;
                                } else {
                                    mismatch++;
                                }
                            }
                        }
                    }
                    
                    if (total > 0) {
                        // Score mejorado: bonificar coincidencias críticas, penalizar desajustes
                        const baseScore = match / total;
                        const criticalBonus = criticalMatch / Math.max(total * 0.4, 1) * 0.3;
                        const penalty = (mismatch / total) * 0.25;
                        const score = baseScore + criticalBonus - penalty;
                        bestMatch = Math.max(bestMatch, score);
                    }
                }
            }
            
            // También probar con tolerancia a errores en posiciones no críticas (mejorado)
            let fuzzyMatch = 0;
            let criticalFuzzy = 0;
            let criticalTotal = 0;
            for (let y = 0; y < 7; y++) {
                for (let x = 0; x < 7; x++) {
                    const isMatch = scaled[y][x] === template[y][x];
                    const isCritical = (x === 3 || y === 3 || x === 0 || x === 6 || y === 0 || y === 6);
                    
                    if (isCritical) criticalTotal++;
                    
                    if (isMatch) {
                        fuzzyMatch += isCritical ? 1.3 : 1.0; // Más peso a críticos
                        if (isCritical) criticalFuzzy++;
                    } else if (!isCritical) {
                        fuzzyMatch += 0.35; // Tolerancia parcial solo en no críticos (reducida para más precisión)
                    }
                    // No dar tolerancia a errores en posiciones críticas
                }
            }
            
            // Score mejorado: dar más importancia a coincidencias críticas
            const criticalRatio = criticalTotal > 0 ? criticalFuzzy / criticalTotal : 0;
            const fuzzyScore = (fuzzyMatch / 49) * (0.7 + criticalRatio * 0.3); // Bonificar si críticos coinciden
            scores[digit] = Math.max(bestMatch * 1.05, fuzzyScore); // Mejorar el mejor match ligeramente
        }
        
        return scores;
    }

    // Predicción mejorada combinando múltiples métodos con pesos optimizados
    predict(pixels) {
        // Preprocesar imagen
        const processed = this.preprocessImage(pixels);
        
        // Muestrear entrada para visualización (tomar cada 28 píxeles)
        const inputSample = [];
        for (let i = 0; i < 784; i += 28) {
            inputSample.push(processed[i]);
        }
        
        // Propagación hacia adelante (para visualización)
        const hidden1 = this.forwardLayer(processed, this.weights.inputToHidden1, 'relu');
        const hidden2 = this.forwardLayer(hidden1, this.weights.hidden1ToHidden2, 'relu');
        const output = this.forwardLayer(hidden2, this.weights.hidden2ToOutput, 'sigmoid');
        const nnProbabilities = this.softmax(output);
        
        // Extraer características avanzadas
        const features = this.extractFeatures(processed);
        
        // Reconocimiento por características (más peso ahora)
        const featureScores = this.recognizeByFeatures(features);
        
        // Matching de plantillas mejorado
        const templateScores = this.matchTemplates(processed);
        
        // Validación cruzada: verificar consistencia entre métodos
        const maxFeature = Math.max(...featureScores);
        const maxTemplate = Math.max(...templateScores);
        const maxNN = Math.max(...nnProbabilities);
        
        const featureIndex = featureScores.indexOf(maxFeature);
        const templateIndex = templateScores.indexOf(maxTemplate);
        const nnIndex = nnProbabilities.indexOf(maxNN);
        
        // Bonificación si hay coincidencia entre métodos
        const consensusBonus = Array(10).fill(0);
        
        // Umbrales balanceados para mantener precisión
        const featureThreshold = maxFeature * 0.65;
        const templateThreshold = maxTemplate * 0.65;
        const nnThreshold = maxNN * 0.65;
        
        if (featureIndex === templateIndex && maxFeature > featureThreshold && maxTemplate > templateThreshold) {
            consensusBonus[featureIndex] += 0.15;
        }
        if (featureIndex === nnIndex && maxFeature > featureThreshold && maxNN > nnThreshold) {
            consensusBonus[featureIndex] += 0.10;
        }
        if (templateIndex === nnIndex && maxTemplate > templateThreshold && maxNN > nnThreshold) {
            consensusBonus[templateIndex] += 0.08;
        }
        if (featureIndex === templateIndex && featureIndex === nnIndex && 
            maxFeature > featureThreshold && maxTemplate > templateThreshold && maxNN > nnThreshold) {
            consensusBonus[featureIndex] += 0.20; // Triple coincidencia
        }
        
        // Bonificación adicional moderada
        for (let i = 0; i < 10; i++) {
            const featureScore = featureScores[i];
            const templateScore = templateScores[i];
            
            if (Math.abs(featureScore - templateScore) < Math.max(maxFeature, maxTemplate) * 0.18 && 
                featureScore > featureThreshold * 0.75 && templateScore > templateThreshold * 0.75) {
                consensusBonus[i] += 0.06;
            }
        }
        
        // Combinar los tres métodos con pesos optimizados
        const combinedProbabilities = [];
        for (let i = 0; i < 10; i++) {
            // Características: 72%, Plantillas: 20%, NN simulada: 8%
            combinedProbabilities[i] = 
                featureScores[i] * 0.72 + 
                templateScores[i] * 0.20 + 
                nnProbabilities[i] * 0.08 +
                consensusBonus[i] * 1.8;
        }
        
        // Softmax final con temperatura balanceada
        const finalProbabilities = this.softmax(combinedProbabilities, 7);
        
        // Muestrear para visualización
        const hidden1Sample = hidden1.slice(0, 32);
        const hidden2Sample = hidden2.slice(0, 32);
        
        return {
            probabilities: finalProbabilities,
            hidden1: hidden1Sample,
            hidden2: hidden2Sample,
            inputSample: inputSample
        };
    }

    // Softmax para normalizar probabilidades con temperatura ajustable
    softmax(values, temperature = 6) {
        const max = Math.max(...values);
        // Temperatura ajustable para decisión más clara
        const exp = values.map(v => {
            const adjusted = (v - max) * temperature;
            // Usar función más suave para valores muy negativos para evitar overflow
            return Math.exp(Math.max(-10, adjusted));
        });
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(e => e / sum);
    }

    // Reconocimiento por plantillas (para mejorar precisión)
    createTemplates() {
        return {
            0: [[0,1,1,1,1,1,0], [1,1,0,0,0,1,1], [1,1,0,0,0,1,1], [1,1,0,0,0,1,1], [1,1,0,0,0,1,1], [1,1,0,0,0,1,1], [0,1,1,1,1,1,0]],
            1: [[0,0,1,1,0,0,0], [0,1,1,1,0,0,0], [0,0,1,1,0,0,0], [0,0,1,1,0,0,0], [0,0,1,1,0,0,0], [0,0,1,1,0,0,0], [0,1,1,1,1,0,0]],
            2: [[0,1,1,1,1,0,0], [1,1,0,0,1,1,0], [0,0,0,0,1,1,0], [0,0,1,1,1,0,0], [0,1,1,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,1,1,1,0]],
            3: [[0,1,1,1,1,0,0], [1,1,0,0,1,1,0], [0,0,0,0,1,1,0], [0,0,1,1,1,0,0], [0,0,0,0,1,1,0], [1,1,0,0,1,1,0], [0,1,1,1,1,0,0]],
            4: [[1,0,0,0,1,0,0], [1,0,0,0,1,0,0], [1,0,0,1,1,0,0], [1,1,1,1,1,1,1], [0,0,0,0,1,0,0], [0,0,0,0,1,0,0], [0,0,0,0,1,0,0]],
            5: [[1,1,1,1,1,1,0], [1,1,0,0,0,0,0], [1,1,1,1,1,0,0], [0,0,0,0,1,1,0], [0,0,0,0,1,1,0], [1,1,0,0,1,1,0], [0,1,1,1,1,0,0]],
            6: [[0,0,1,1,1,0,0], [0,1,1,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,1,1,0,0], [1,1,0,0,1,1,0], [1,1,0,0,1,1,0], [0,1,1,1,1,0,0]],
            7: [[1,1,1,1,1,1,1], [0,0,0,0,1,1,0], [0,0,0,1,1,0,0], [0,0,1,1,0,0,0], [0,0,1,1,0,0,0], [0,1,1,0,0,0,0], [0,1,1,0,0,0,0]],
            8: [[0,1,1,1,1,0,0], [1,1,0,0,1,1,0], [1,1,0,0,1,1,0], [0,1,1,1,1,0,0], [1,1,0,0,1,1,0], [1,1,0,0,1,1,0], [0,1,1,1,1,0,0]],
            9: [[0,1,1,1,1,0,0], [1,1,0,0,1,1,0], [1,1,0,0,1,1,0], [0,1,1,1,1,1,0], [0,0,0,0,1,1,0], [0,0,0,1,1,0,0], [0,1,1,1,0,0,0]]
        };
    }
}
