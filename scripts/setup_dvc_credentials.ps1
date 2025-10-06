# Script para configurar credenciales de DVC en GCS (Windows)
# Uso: .\scripts\setup_dvc_credentials.ps1

$ErrorActionPreference = "Stop"

Write-Host "🔧 Configurando credenciales de DVC para GCS..." -ForegroundColor Cyan
Write-Host ""

# Verificar que gcloud esté instalado
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: gcloud CLI no está instalado" -ForegroundColor Red
    Write-Host "   Instala desde: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Verificar que dvc esté instalado
if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: DVC no está instalado" -ForegroundColor Red
    Write-Host "   Instala con: pip install dvc[gs]" -ForegroundColor Yellow
    exit 1
}

# Autenticarse con GCP
Write-Host "📝 Paso 1: Autenticación con GCP" -ForegroundColor Green
Write-Host "   Se abrirá tu navegador para autenticarte..."
gcloud auth application-default login

# Configurar DVC localmente
Write-Host ""
Write-Host "📝 Paso 2: Configurando DVC localmente" -ForegroundColor Green

$credPath = "$env:APPDATA\gcloud\application_default_credentials.json"
dvc remote modify --local gcs-remote credentialpath $credPath

# Verificar configuración
Write-Host ""
Write-Host "✅ Configuración completada!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Verificando configuración:" -ForegroundColor Cyan
dvc remote list -v

Write-Host ""
Write-Host "🧪 Probando conexión con el bucket..." -ForegroundColor Cyan
try {
    dvc status --cloud | Out-Null
    Write-Host "✅ Conexión exitosa con GCS!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Advertencia: No se pudo conectar con GCS" -ForegroundColor Yellow
    Write-Host "   Verifica que tengas permisos en el proyecto GCP" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 ¡Listo! Ya puedes usar DVC con GCS" -ForegroundColor Green
Write-Host ""
Write-Host "Comandos útiles:" -ForegroundColor Cyan
Write-Host "  dvc status --cloud  # Ver estado del cache remoto"
Write-Host "  dvc push            # Subir datos al bucket"
Write-Host "  dvc pull            # Descargar datos del bucket"
