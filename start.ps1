# Script PowerShell para ejecutar el proyecto de Reconocimiento de Dígitos
# Ejecuta el servidor y el cliente simultáneamente

Write-Host "🚀 Iniciando proyecto de Reconocimiento de Dígitos..." -ForegroundColor Cyan
Write-Host ""

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "📁 Directorio del proyecto: $ProjectDir" -ForegroundColor Green
Write-Host ""

# Iniciar servidor en una nueva ventana
Write-Host "🖥️  Iniciando servidor en puerto 4000..." -ForegroundColor Blue
$serverProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectDir\server'; npm run dev" -PassThru

# Iniciar cliente en una nueva ventana
Write-Host "🌐 Iniciando cliente en puerto 3000..." -ForegroundColor Blue
$clientProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectDir\client'; npm run dev" -PassThru

Write-Host ""
Write-Host "✅ Servicios iniciados:" -ForegroundColor Green
Write-Host "   → Cliente:  http://localhost:3000" -ForegroundColor Yellow
Write-Host "   → Servidor: http://localhost:4000" -ForegroundColor Yellow
Write-Host ""
Write-Host "📌 Se abrieron dos ventanas de PowerShell para cada servicio." -ForegroundColor Magenta
Write-Host "   Cierra las ventanas para detener los servicios." -ForegroundColor Magenta

