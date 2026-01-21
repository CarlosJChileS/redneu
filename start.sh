#!/bin/bash

# Iniciar proyecto de Reconocimiento de Digitos
# Uso: ./start.sh

cd "$(dirname "$0")"

echo "[>] Iniciando proyecto..."
echo ""

# Matar procesos previos en los puertos (opcional)
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:4000 | xargs kill -9 2>/dev/null

# Iniciar servidor y cliente con npm
npm run dev
echo -e "${RED}Presiona Ctrl+C para detener ambos servicios${NC}"
echo ""

# Esperar a que terminen los procesos
wait $SERVER_PID $CLIENT_PID

