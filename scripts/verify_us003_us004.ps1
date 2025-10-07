# Script de verificación para US-003 y US-004 (Windows)

$ErrorActionPreference = "Continue"

Write-Host "🔍 Verificando US-003 y US-004..." -ForegroundColor Cyan
Write-Host ""

# Función para verificar
function Check-Item {
    param($Message, $Condition)
    if ($Condition) {
        Write-Host "✅ $Message" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ $Message" -ForegroundColor Red
        return $false
    }
}

# US-003: Docker + FastAPI
Write-Host "📦 US-003: Docker + FastAPI" -ForegroundColor Cyan
Write-Host "─────────────────────────────"

# 1. Verificar archivos
$files = @(
    "Dockerfile.api",
    "docker-compose.yml",
    "requirements-api.txt",
    ".dockerignore",
    "src/api/main.py"
)

foreach ($file in $files) {
    Check-Item "Archivo $file existe" (Test-Path $file) | Out-Null
}

# 2. Verificar servicios
Write-Host ""
Write-Host "🚀 Verificando servicios..." -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    Check-Item "API responde en http://localhost:8000/health" ($response.StatusCode -eq 200) | Out-Null
} catch {
    Write-Host "⚠️  API no está corriendo. Ejecuta: docker-compose up api -d" -ForegroundColor Yellow
}

# US-004: MLflow + Prefect
Write-Host ""
Write-Host "📊 US-004: MLflow + Prefect" -ForegroundColor Cyan
Write-Host "─────────────────────────────"

# 1. Verificar archivos
$mlopsFiles = @(
    "src/experiments/example_mlflow.py",
    "src/flows/example_flow.py"
)

foreach ($file in $mlopsFiles) {
    Check-Item "Archivo $file existe" (Test-Path $file) | Out-Null
}

# 2. Verificar servicios MLOps
Write-Host ""
Write-Host "🚀 Verificando servicios MLOps..." -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 5
    Check-Item "MLflow responde en http://localhost:5000" ($response.StatusCode -eq 200) | Out-Null
} catch {
    Write-Host "⚠️  MLflow no está corriendo. Ejecuta: docker-compose up mlflow -d" -ForegroundColor Yellow
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:4200/api/health" -UseBasicParsing -TimeoutSec 5
    Check-Item "Prefect responde en http://localhost:4200" ($response.StatusCode -eq 200) | Out-Null
} catch {
    Write-Host "⚠️  Prefect no está corriendo. Ejecuta: docker-compose up prefect -d" -ForegroundColor Yellow
}

# Resumen
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ Verificación completada" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Servicios disponibles:" -ForegroundColor Cyan
Write-Host "   • API:     http://localhost:8000"
Write-Host "   • Docs:    http://localhost:8000/docs"
Write-Host "   • MLflow:  http://localhost:5000"
Write-Host "   • Prefect: http://localhost:4200"
Write-Host ""
Write-Host "🧪 Comandos de prueba:" -ForegroundColor Cyan
Write-Host "   • API:     curl http://localhost:8000/health"
Write-Host "   • MLflow:  poetry run python src/experiments/example_mlflow.py"
Write-Host "   • Prefect: poetry run python src/flows/example_flow.py"
Write-Host ""
