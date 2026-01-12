@echo off
REM Script Batch para ejecutar el proyecto de Reconocimiento de Digitos
REM Ejecuta el servidor y el cliente simultaneamente

echo.
echo ========================================
echo   Reconocimiento de Digitos - Inicio
echo ========================================
echo.

cd /d "%~dp0"

echo [INFO] Directorio del proyecto: %CD%
echo.

echo [INFO] Iniciando servidor en puerto 4000...
start "Servidor - Puerto 4000" cmd /k "cd server && npm run dev"

echo [INFO] Iniciando cliente en puerto 3000...
start "Cliente - Puerto 3000" cmd /k "cd client && npm run dev"

echo.
echo ========================================
echo   Servicios iniciados correctamente
echo ========================================
echo.
echo   Cliente:  http://localhost:3000
echo   Servidor: http://localhost:4000
echo.
echo   Cierra las ventanas CMD para detener
echo ========================================
echo.
pause

