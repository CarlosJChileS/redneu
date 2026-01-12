#!/bin/bash

# Script para ejecutar el proyecto de Reconocimiento de Dígitos
# Ejecuta el servidor y el cliente simultáneamente

echo "🚀 Iniciando proyecto de Reconocimiento de Dígitos..."
echo ""

# Colores para diferenciar los outputs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # Sin color

# Función para matar procesos hijos al salir
cleanup() {
    echo ""
    echo -e "${RED}⏹️  Deteniendo servicios...${NC}"
    kill $SERVER_PID $CLIENT_PID 2>/dev/null
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT SIGTERM

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${GREEN}📁 Directorio del proyecto: $PROJECT_DIR${NC}"
echo ""

# Iniciar servidor
echo -e "${BLUE}🖥️  Iniciando servidor en puerto 4000...${NC}"
cd server && npm run dev &
SERVER_PID=$!

# Volver al directorio raíz
cd "$PROJECT_DIR"

# Iniciar cliente
echo -e "${BLUE}🌐 Iniciando cliente en puerto 3000...${NC}"
cd client && npm run dev &
CLIENT_PID=$!

echo ""
echo -e "${GREEN}✅ Servicios iniciados:${NC}"
echo -e "   ${BLUE}→ Cliente:${NC}  http://localhost:3000"
echo -e "   ${BLUE}→ Servidor:${NC} http://localhost:4000"
echo ""
echo -e "${RED}Presiona Ctrl+C para detener ambos servicios${NC}"
echo ""

# Esperar a que terminen los procesos
wait $SERVER_PID $CLIENT_PID

