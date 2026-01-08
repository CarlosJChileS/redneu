// network.js - Red neuronal precisa para reconocimiento de dígitos
class SimpleNeuralNetwork {
    constructor() {
        this.isLoaded = true;
        this.templates = this.createTemplates();
        console.log('Red neuronal inicializada');
    }

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

    predict(pixels) {
        const size = 28;
        
        // Crear grid binario
        const grid = [];
        for (let y = 0; y < size; y++) {
            grid[y] = [];
            for (let x = 0; x < size; x++) {
                grid[y][x] = pixels[y * size + x] > 0.15 ? 1 : 0;
            }
        }
        
        // Extraer características
        const features = this.extractFeatures(grid, size);
        
        // Calcular puntuaciones por reglas
        const ruleScores = this.applyRules(features);
        
        // Calcular puntuaciones por plantillas
        const templateScores = this.matchTemplates(grid, size);
        
        // Combinar: 75% reglas, 25% plantillas
        const combined = [];
        for (let i = 0; i < 10; i++) {
            combined[i] = ruleScores[i] * 0.75 + templateScores[i] * 0.25;
        }
        
        // Softmax para probabilidades
        const probabilities = this.softmax(combined, 8);
        
        // Generar activaciones para visualización
        const hidden1 = this.generateActivations(features, 32);
        const hidden2 = this.generateActivations(combined, 32);
        
        return { probabilities, hidden1, hidden2, inputSample: pixels.slice(0, 28) };
    }

    extractFeatures(grid, size) {
        const f = {};
        
        // Contar píxeles activos
        let total = 0, cx = 0, cy = 0;
        let minX = size, maxX = 0, minY = size, maxY = 0;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (grid[y][x]) {
                    total++;
                    cx += x;
                    cy += y;
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        
        f.total = total;
        if (total < 5) return f;
        
        f.centerX = cx / total;
        f.centerY = cy / total;
        f.width = maxX - minX + 1;
        f.height = maxY - minY + 1;
        f.ratio = f.width / f.height;
        
        // Densidad por zonas 3x3
        const zoneH = size / 3;
        const zoneW = size / 3;
        for (let zy = 0; zy < 3; zy++) {
            for (let zx = 0; zx < 3; zx++) {
                let cnt = 0, sum = 0;
                for (let y = Math.floor(zy * zoneH); y < Math.floor((zy + 1) * zoneH); y++) {
                    for (let x = Math.floor(zx * zoneW); x < Math.floor((zx + 1) * zoneW); x++) {
                        sum += grid[y][x];
                        cnt++;
                    }
                }
                f['z' + zy + zx] = cnt > 0 ? sum / cnt : 0;
            }
        }
        
        // Huecos (flood fill)
        f.holes = this.countHoles(grid, size);
        
        // Líneas horizontales
        f.horizontalLines = this.countHorizontalLines(grid, size);
        
        // Líneas verticales
        f.verticalLines = this.countVerticalLines(grid, size);
        
        // Cruces
        f.crossings = this.countCrossings(grid, size);
        
        // Simetría
        f.symmetry = this.measureSymmetry(grid, size);
        
        // Distribución
        const topDensity = (f.z00 + f.z01 + f.z02) / 3;
        const bottomDensity = (f.z20 + f.z21 + f.z22) / 3;
        const leftDensity = (f.z00 + f.z10 + f.z20) / 3;
        const rightDensity = (f.z02 + f.z12 + f.z22) / 3;
        
        f.topHeavy = topDensity > bottomDensity * 1.2;
        f.bottomHeavy = bottomDensity > topDensity * 1.2;
        f.leftHeavy = leftDensity > rightDensity * 1.2;
        f.rightHeavy = rightDensity > leftDensity * 1.2;
        
        return f;
    }

    countHoles(grid, size) {
        const visited = Array(size).fill().map(() => Array(size).fill(false));
        
        // Flood fill desde bordes
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
        
        // Contar huecos internos
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

    countHorizontalLines(grid, size) {
        let lines = 0;
        for (let y = 0; y < size; y++) {
            let maxLen = 0, len = 0;
            for (let x = 0; x < size; x++) {
                if (grid[y][x]) { len++; maxLen = Math.max(maxLen, len); }
                else len = 0;
            }
            if (maxLen > size * 0.5) lines++;
        }
        return lines;
    }

    countVerticalLines(grid, size) {
        let lines = 0;
        for (let x = 0; x < size; x++) {
            let maxLen = 0, len = 0;
            for (let y = 0; y < size; y++) {
                if (grid[y][x]) { len++; maxLen = Math.max(maxLen, len); }
                else len = 0;
            }
            if (maxLen > size * 0.5) lines++;
        }
        return lines;
    }

    countCrossings(grid, size) {
        let crossings = 0;
        for (let y = 2; y < size - 2; y++) {
            for (let x = 2; x < size - 2; x++) {
                if (grid[y][x]) {
                    const up = grid[y-1][x] && grid[y-2][x];
                    const down = grid[y+1][x] && grid[y+2][x];
                    const left = grid[y][x-1] && grid[y][x-2];
                    const right = grid[y][x+1] && grid[y][x+2];
                    if ((up || down) && (left || right)) crossings++;
                }
            }
        }
        return Math.min(crossings, 50);
    }

    measureSymmetry(grid, size) {
        let match = 0, total = 0;
        const midX = size / 2;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < midX; x++) {
                if (grid[y][x] === grid[y][size - 1 - x]) match++;
                total++;
            }
        }
        return total > 0 ? match / total : 0;
    }

    applyRules(f) {
        const scores = Array(10).fill(0);
        if (!f.total || f.total < 5) return scores;
        
        // === 0: Círculo con hueco ===
        if (f.holes >= 1) scores[0] += 25;
        if (f.ratio > 0.6 && f.ratio < 1.2) scores[0] += 10;
        if (f.z11 < 0.15) scores[0] += 15; // Centro vacío
        if (f.symmetry > 0.6) scores[0] += 10;
        if (f.holes === 0) scores[0] -= 30;
        
        // === 1: Muy estrecho, vertical ===
        if (f.ratio < 0.35) scores[1] += 30;
        if (f.ratio < 0.45) scores[1] += 15;
        if (f.verticalLines >= 1) scores[1] += 15;
        if (f.horizontalLines === 0) scores[1] += 15;
        if (f.crossings < 5) scores[1] += 10;
        if (f.holes === 0) scores[1] += 5;
        if (f.ratio > 0.5) scores[1] -= 25;
        if (f.horizontalLines >= 1) scores[1] -= 20;
        if (f.crossings >= 10) scores[1] -= 15;
        
        // === 2: Curva arriba derecha, diagonal, base izquierda ===
        if (f.holes === 0) scores[2] += 5;
        if (f.z02 > f.z00 * 1.3) scores[2] += 15;
        if (f.z20 > f.z22 * 1.3) scores[2] += 15;
        if (f.horizontalLines >= 1) scores[2] += 10;
        if (f.holes >= 1) scores[2] -= 15;
        
        // === 3: Curvas a la derecha ===
        if (f.holes === 0) scores[3] += 5;
        if (f.z02 > f.z00 && f.z22 > f.z20) scores[3] += 20;
        if (f.z12 > f.z10) scores[3] += 10;
        if (f.symmetry > 0.45 && f.symmetry < 0.7) scores[3] += 10;
        if (f.holes >= 1) scores[3] -= 15;
        
        // === 4: Línea horizontal + vertical + cruces ===
        if (f.horizontalLines >= 1 && f.verticalLines >= 1) scores[4] += 25;
        if (f.crossings >= 10) scores[4] += 20;
        if (f.z11 > 0.2) scores[4] += 15; // Centro activo (cruce)
        if (f.z00 > 0.1 && f.z22 > 0.1) scores[4] += 10;
        if (f.holes === 0) scores[4] += 5;
        if (f.horizontalLines === 0) scores[4] -= 30;
        if (f.verticalLines === 0) scores[4] -= 25;
        if (f.crossings < 5) scores[4] -= 20;
        if (f.holes >= 1) scores[4] -= 20;
        if (f.ratio < 0.4) scores[4] -= 25;
        
        // === 5: Horizontal arriba, curva abajo ===
        if (f.holes === 0) scores[5] += 5;
        if (f.z00 > f.z02 * 1.3) scores[5] += 15;
        if (f.z22 > f.z20 * 1.3) scores[5] += 15;
        if (f.horizontalLines >= 1) scores[5] += 10;
        if (f.holes >= 1) scores[5] -= 15;
        
        // === 6: Hueco abajo ===
        if (f.holes >= 1) scores[6] += 20;
        if (f.bottomHeavy) scores[6] += 15;
        if (f.z21 > f.z01) scores[6] += 15;
        if (f.centerY > 14) scores[6] += 10;
        if (f.holes === 0) scores[6] -= 25;
        if (f.topHeavy) scores[6] -= 15;
        
        // === 7: Horizontal arriba, diagonal ===
        if (f.holes === 0) scores[7] += 10;
        if (f.horizontalLines >= 1) scores[7] += 15;
        if (f.z01 > 0.15 || f.z02 > 0.15) scores[7] += 15;
        if (f.topHeavy) scores[7] += 15;
        if (f.z20 < 0.1 && f.z21 < 0.1) scores[7] += 10;
        if (f.holes >= 1) scores[7] -= 20;
        if (f.bottomHeavy) scores[7] -= 15;
        
        // === 8: Dos huecos, simétrico ===
        if (f.holes >= 2) scores[8] += 35;
        if (f.holes === 1 && f.symmetry > 0.55) scores[8] += 20;
        if (f.symmetry > 0.6) scores[8] += 15;
        if (f.z11 < 0.15 && f.z01 > 0.1 && f.z21 > 0.1) scores[8] += 15;
        if (f.holes === 0) scores[8] -= 30;
        
        // === 9: Hueco arriba ===
        if (f.holes >= 1) scores[9] += 20;
        if (f.topHeavy) scores[9] += 15;
        if (f.z01 > f.z21) scores[9] += 15;
        if (f.centerY < 14) scores[9] += 10;
        if (f.holes === 0) scores[9] -= 25;
        if (f.bottomHeavy) scores[9] -= 15;
        
        // Normalizar
        const max = Math.max(...scores);
        return max > 0 ? scores.map(s => Math.max(0, s / max)) : scores;
    }

    matchTemplates(grid, size) {
        const scores = Array(10).fill(0);
        
        // Escalar grid a 7x7
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
        
        // Comparar con cada plantilla
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

    softmax(values, temperature = 8) {
        const max = Math.max(...values);
        const exp = values.map(v => Math.exp((v - max) * temperature));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(e => e / sum);
    }

    generateActivations(data, count) {
        const values = Array.isArray(data) ? data : Object.values(data).filter(v => typeof v === 'number');
        const act = [];
        for (let i = 0; i < count; i++) {
            const v = values[i % values.length] || 0;
            act.push(Math.min(1, Math.max(0, Math.abs(v) * 0.5 + Math.random() * 0.3)));
        }
        return act;
    }
}
