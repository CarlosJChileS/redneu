// network.js - Red neuronal con TensorFlow.js para MNIST
class SimpleNeuralNetwork {
    constructor() {
        this.model = null;
        this.isLoaded = false;
        this.isReady = false;
        this.templates = this.createTemplates();
        console.log('Inicializando TensorFlow.js...');
    }

    async initialize() {
        try {
            await tf.ready();
            console.log('TensorFlow listo, backend:', tf.getBackend());
            
            // Crear y entrenar modelo CNN
            this.model = this.createCNNModel();
            await this.trainModel();
            
            this.isReady = true;
            this.isLoaded = true;
            console.log('Modelo CNN listo para predicciones');
            return true;
        } catch (error) {
            console.error('Error inicializando TensorFlow:', error);
            this.isLoaded = true; // Usar fallback
            return false;
        }
    }

    createCNNModel() {
        const model = tf.sequential();
        
        // Conv2D + MaxPool 1
        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        
        // Conv2D + MaxPool 2
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        
        // Flatten + Dense
        model.add(tf.layers.flatten());
        model.add(tf.layers.dropout({ rate: 0.25 }));
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Modelo CNN creado');
        return model;
    }

    async trainModel() {
        console.log('Generando datos de entrenamiento...');
        const { xs, ys } = this.generateTrainingData(1500);
        
        console.log('Entrenando modelo CNN (esto tomará unos segundos)...');
        await this.model.fit(xs, ys, {
            epochs: 12,
            batchSize: 32,
            validationSplit: 0.1,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Época ${epoch + 1}/12: accuracy=${(logs.acc * 100).toFixed(1)}%`);
                }
            }
        });
        
        xs.dispose();
        ys.dispose();
        console.log('Entrenamiento completado');
    }

    generateTrainingData(samplesPerDigit) {
        const images = [];
        const labels = [];
        
        for (let digit = 0; digit < 10; digit++) {
            const template = this.templates[digit];
            for (let s = 0; s < samplesPerDigit; s++) {
                const img = this.generateVariation(template);
                images.push(img);
                const label = new Array(10).fill(0);
                label[digit] = 1;
                labels.push(label);
            }
        }
        
        const xs = tf.tensor4d(images, [images.length, 28, 28, 1]);
        const ys = tf.tensor2d(labels, [labels.length, 10]);
        return { xs, ys };
    }

    generateVariation(template) {
        const img = Array(28).fill().map(() => Array(28).fill(0));
        
        const offsetX = Math.floor(Math.random() * 6) - 3;
        const offsetY = Math.floor(Math.random() * 6) - 3;
        const scale = 0.85 + Math.random() * 0.3;
        const thickness = 2 + Math.floor(Math.random() * 2);
        
        const baseSize = Math.floor(20 * scale);
        const startX = Math.floor((28 - baseSize) / 2) + offsetX;
        const startY = Math.floor((28 - baseSize) / 2) + offsetY;
        
        for (let ty = 0; ty < 7; ty++) {
            for (let tx = 0; tx < 7; tx++) {
                if (template[ty][tx] === 1) {
                    const px = startX + Math.floor(tx * baseSize / 7);
                    const py = startY + Math.floor(ty * baseSize / 7);
                    
                    for (let dy = -thickness; dy <= thickness; dy++) {
                        for (let dx = -thickness; dx <= thickness; dx++) {
                            const nx = px + dx;
                            const ny = py + dy;
                            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                                const dist = Math.sqrt(dx*dx + dy*dy);
                                const intensity = Math.max(0, 1 - dist / (thickness + 1));
                                img[ny][nx] = Math.max(img[ny][nx], intensity * (0.7 + Math.random() * 0.3));
                            }
                        }
                    }
                }
            }
        }
        
        // Ruido
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                if (img[y][x] > 0) {
                    img[y][x] = Math.min(1, img[y][x] + (Math.random() - 0.5) * 0.15);
                }
            }
        }
        
        return img;
    }

    createTemplates() {
        return {
            0: [[0,1,1,1,1,1,0],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,0]],
            1: [[0,0,1,1,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,1,1,1,1,0,0]],
            2: [[0,1,1,1,1,1,0],[1,1,0,0,0,1,1],[0,0,0,0,1,1,0],[0,0,1,1,1,0,0],[0,1,1,0,0,0,0],[1,1,0,0,0,0,0],[1,1,1,1,1,1,1]],
            3: [[0,1,1,1,1,1,0],[1,1,0,0,0,1,1],[0,0,0,0,0,1,1],[0,0,1,1,1,1,0],[0,0,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,0]],
            4: [[0,0,0,1,1,1,0],[0,0,1,1,1,1,0],[0,1,1,0,1,1,0],[1,1,0,0,1,1,0],[1,1,1,1,1,1,1],[0,0,0,0,1,1,0],[0,0,0,0,1,1,0]],
            5: [[1,1,1,1,1,1,1],[1,1,0,0,0,0,0],[1,1,1,1,1,1,0],[0,0,0,0,0,1,1],[0,0,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,0]],
            6: [[0,1,1,1,1,1,0],[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,1,1,1,1,1,0],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,0]],
            7: [[1,1,1,1,1,1,1],[0,0,0,0,0,1,1],[0,0,0,0,1,1,0],[0,0,0,1,1,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0]],
            8: [[0,1,1,1,1,1,0],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,0],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,0]],
            9: [[0,1,1,1,1,1,0],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[0,1,1,1,1,1,1],[0,0,0,0,0,1,1],[0,0,0,0,1,1,0],[0,1,1,1,1,0,0]]
        };
    }

    predict(pixels) {
        if (this.isReady && this.model) {
            return this.predictWithTensorFlow(pixels);
        }
        return this.predictFallback(pixels);
    }

    predictWithTensorFlow(pixels) {
        const size = 28;
        
        // Crear imagen 28x28 desde los pixeles
        const img = [];
        for (let y = 0; y < size; y++) {
            const row = [];
            for (let x = 0; x < size; x++) {
                row.push(pixels[y * size + x]);
            }
            img.push(row);
        }
        
        // Tensor para predicción
        const tensor = tf.tensor4d([img.map(r => r.map(v => [v]))], [1, 28, 28, 1]);
        const prediction = this.model.predict(tensor);
        const probs = prediction.dataSync();
        
        tensor.dispose();
        prediction.dispose();
        
        const probabilities = Array.from(probs);
        
        // Para visualización
        const hidden1 = this.generateActivations(probabilities, 32);
        const hidden2 = this.generateActivations(probabilities, 32);
        
        return { probabilities, hidden1, hidden2, inputSample: pixels.slice(0, 28) };
    }

    predictFallback(pixels) {
        // Método de respaldo basado en reglas
        const size = 28;
        const grid = [];
        for (let y = 0; y < size; y++) {
            grid[y] = [];
            for (let x = 0; x < size; x++) {
                grid[y][x] = pixels[y * size + x] > 0.15 ? 1 : 0;
            }
        }
        
        const ruleScores = this.applyRules(grid, size);
        const templateScores = this.matchTemplates(grid, size);
        
        const combined = [];
        for (let i = 0; i < 10; i++) {
            combined[i] = ruleScores[i] * 0.6 + templateScores[i] * 0.4;
        }
        
        const probabilities = this.softmax(combined, 6);
        const hidden1 = this.generateActivations(combined, 32);
        const hidden2 = this.generateActivations(probabilities, 32);
        
        return { probabilities, hidden1, hidden2, inputSample: pixels.slice(0, 28) };
    }

    applyRules(grid, size) {
        const scores = Array(10).fill(0);
        
        // Características básicas
        let total = 0, cx = 0, cy = 0;
        let minX = size, maxX = 0, minY = size, maxY = 0;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (grid[y][x]) {
                    total++;
                    cx += x; cy += y;
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        
        if (total < 5) return scores;
        
        const width = maxX - minX + 1;
        const height = maxY - minY + 1;
        const ratio = width / height;
        
        // Densidades por zona
        const zones = Array(9).fill(0);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (grid[y][x]) {
                    const zx = Math.min(2, Math.floor(x * 3 / size));
                    const zy = Math.min(2, Math.floor(y * 3 / size));
                    zones[zy * 3 + zx]++;
                }
            }
        }
        const maxZ = Math.max(...zones, 1);
        const z = zones.map(v => v / maxZ);
        
        // Huecos
        const holes = this.countHoles(grid, size);
        
        // Líneas
        let hLines = 0, vLines = 0;
        for (let y = 0; y < size; y++) {
            let len = 0;
            for (let x = 0; x < size; x++) {
                if (grid[y][x]) len++;
                else len = 0;
            }
            if (len > size * 0.5) hLines++;
        }
        for (let x = 0; x < size; x++) {
            let len = 0;
            for (let y = 0; y < size; y++) {
                if (grid[y][x]) len++;
                else len = 0;
            }
            if (len > size * 0.5) vLines++;
        }
        
        // Reglas por dígito
        // 0: Oval con hueco
        if (holes >= 1) scores[0] += 0.4;
        if (ratio > 0.7 && ratio < 1.3) scores[0] += 0.2;
        if (z[4] < 0.3) scores[0] += 0.2;
        
        // 1: Delgado vertical
        if (ratio < 0.4) scores[1] += 0.5;
        if (vLines >= 1 && hLines === 0) scores[1] += 0.3;
        if (holes === 0) scores[1] += 0.1;
        
        // 2: Diagonal arriba-derecha a abajo-izquierda
        if (holes === 0 && z[2] > 0.3 && z[6] > 0.3) scores[2] += 0.4;
        if (hLines >= 1) scores[2] += 0.2;
        
        // 3: Curvas a la derecha
        if (holes === 0 && z[2] > z[0] && z[8] > z[6]) scores[3] += 0.4;
        
        // 4: Cruz
        if (hLines >= 1 && vLines >= 1) scores[4] += 0.5;
        if (holes === 0) scores[4] += 0.2;
        
        // 5: Arriba izq, abajo derecha
        if (holes === 0 && z[0] > 0.4 && z[8] > 0.4) scores[5] += 0.4;
        
        // 6: Hueco abajo
        if (holes >= 1 && z[7] > z[1]) scores[6] += 0.5;
        
        // 7: Línea arriba, diagonal
        if (holes === 0 && z[0] + z[1] + z[2] > 1.5) scores[7] += 0.4;
        
        // 8: Dos huecos
        if (holes >= 2) scores[8] += 0.6;
        else if (holes >= 1 && z[4] < 0.2) scores[8] += 0.3;
        
        // 9: Hueco arriba
        if (holes >= 1 && z[1] > z[7]) scores[9] += 0.5;
        
        return scores;
    }

    countHoles(grid, size) {
        const visited = Array(size).fill().map(() => Array(size).fill(false));
        const stack = [];
        
        for (let i = 0; i < size; i++) {
            if (!grid[0][i]) stack.push([0, i]);
            if (!grid[size-1][i]) stack.push([size-1, i]);
            if (!grid[i][0]) stack.push([i, 0]);
            if (!grid[i][size-1]) stack.push([i, size-1]);
        }
        
        while (stack.length) {
            const [y, x] = stack.pop();
            if (y < 0 || y >= size || x < 0 || x >= size || visited[y][x] || grid[y][x]) continue;
            visited[y][x] = true;
            stack.push([y-1,x], [y+1,x], [y,x-1], [y,x+1]);
        }
        
        let holes = 0;
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                if (!grid[y][x] && !visited[y][x]) {
                    holes++;
                    const s2 = [[y, x]];
                    while (s2.length) {
                        const [py, px] = s2.pop();
                        if (py < 0 || py >= size || px < 0 || px >= size || visited[py][px] || grid[py][px]) continue;
                        visited[py][px] = true;
                        s2.push([py-1,px], [py+1,px], [py,px-1], [py,px+1]);
                    }
                }
            }
        }
        return holes;
    }

    matchTemplates(grid, size) {
        const scores = Array(10).fill(0);
        const scaled = [];
        
        for (let y = 0; y < 7; y++) {
            scaled[y] = [];
            for (let x = 0; x < 7; x++) {
                let sum = 0, cnt = 0;
                const y1 = Math.floor(y * size / 7);
                const y2 = Math.floor((y + 1) * size / 7);
                const x1 = Math.floor(x * size / 7);
                const x2 = Math.floor((x + 1) * size / 7);
                for (let py = y1; py < y2; py++) {
                    for (let px = x1; px < x2; px++) {
                        sum += grid[py][px];
                        cnt++;
                    }
                }
                scaled[y][x] = cnt > 0 && sum / cnt > 0.3 ? 1 : 0;
            }
        }
        
        for (let digit = 0; digit < 10; digit++) {
            const template = this.templates[digit];
            let match = 0;
            for (let y = 0; y < 7; y++) {
                for (let x = 0; x < 7; x++) {
                    if (scaled[y][x] === template[y][x]) match++;
                }
            }
            scores[digit] = match / 49;
        }
        
        return scores;
    }

    softmax(values, temperature = 6) {
        const max = Math.max(...values);
        const exp = values.map(v => Math.exp((v - max) * temperature));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(e => e / sum);
    }

    generateActivations(data, count) {
        const values = Array.isArray(data) ? data : Object.values(data);
        const act = [];
        for (let i = 0; i < count; i++) {
            const v = values[i % values.length] || 0;
            act.push(Math.min(1, Math.max(0, Math.abs(v) + Math.random() * 0.2)));
        }
        return act;
    }
}
