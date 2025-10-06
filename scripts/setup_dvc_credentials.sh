#!/bin/bash
# Script para configurar credenciales de DVC en GCS
# Uso: ./scripts/setup_dvc_credentials.sh

set -e

echo "🔧 Configurando credenciales de DVC para GCS..."
echo ""

# Verificar que gcloud esté instalado
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI no está instalado"
    echo "   Instala desde: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verificar que dvc esté instalado
if ! command -v dvc &> /dev/null; then
    echo "❌ Error: DVC no está instalado"
    echo "   Instala con: pip install dvc[gs]"
    exit 1
fi

# Autenticarse con GCP
echo "📝 Paso 1: Autenticación con GCP"
echo "   Se abrirá tu navegador para autenticarte..."
gcloud auth application-default login

# Configurar DVC localmente
echo ""
echo "📝 Paso 2: Configurando DVC localmente"

# Detectar sistema operativo
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    CRED_PATH="$APPDATA\\gcloud\\application_default_credentials.json"
else
    # Linux/Mac
    CRED_PATH="$HOME/.config/gcloud/application_default_credentials.json"
fi

dvc remote modify --local gcs-remote credentialpath "$CRED_PATH"

# Verificar configuración
echo ""
echo "✅ Configuración completada!"
echo ""
echo "📋 Verificando configuración:"
dvc remote list -v

echo ""
echo "🧪 Probando conexión con el bucket..."
if dvc status --cloud &> /dev/null; then
    echo "✅ Conexión exitosa con GCS!"
else
    echo "⚠️  Advertencia: No se pudo conectar con GCS"
    echo "   Verifica que tengas permisos en el proyecto GCP"
fi

echo ""
echo "🎉 ¡Listo! Ya puedes usar DVC con GCS"
echo ""
echo "Comandos útiles:"
echo "  dvc status --cloud  # Ver estado del cache remoto"
echo "  dvc push            # Subir datos al bucket"
echo "  dvc pull            # Descargar datos del bucket"
