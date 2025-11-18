#!/bin/bash

# ============================================================================
# Script de Preparación para Deployment en Streamlit Cloud
# Energy Optimization AI - Streamlit Frontend
# ============================================================================
#
# Este script prepara el repositorio para deployment en Streamlit Cloud
# (opción gratuita y sencilla).
#
# Streamlit Cloud:
#   - Hosting gratuito
#   - Deploy desde GitHub
#   - SSL automático
#   - Dominio gratuito (.streamlit.app)
#
# Uso:
#   ./scripts/deploy_streamlit_cloud.sh
#
# Autor: Equipo Atreides
# User Story: US-033 - Deployment de Streamlit
# ============================================================================

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCIONES
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ============================================================================
# INICIO
# ============================================================================

print_header "☁️  Preparando deployment para Streamlit Cloud"

# ============================================================================
# GENERAR REQUIREMENTS.TXT
# ============================================================================

print_header "📦 Generando requirements.txt"

print_info "Exportando dependencias desde Poetry..."

if ! command -v poetry &> /dev/null; then
    print_error "Poetry no está instalado"
    echo "Instala Poetry desde: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Generar requirements.txt sin hashes (Streamlit Cloud no los soporta bien)
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Asegurar que streamlit y dependencias clave estén incluidas
if ! grep -q "streamlit" requirements.txt; then
    echo "streamlit>=1.28.0" >> requirements.txt
fi

if ! grep -q "plotly" requirements.txt; then
    echo "plotly>=5.17.0" >> requirements.txt
fi

if ! grep -q "requests" requirements.txt; then
    echo "requests>=2.31.0" >> requirements.txt
fi

print_success "requirements.txt generado"

# ============================================================================
# CREAR ARCHIVO DE SECRETOS DE EJEMPLO
# ============================================================================

print_header "🔐 Creando archivo de secretos de ejemplo"

mkdir -p .streamlit

cat > .streamlit/secrets.toml.example << 'EOF'
# ============================================================================
# Archivo de ejemplo de secretos para Streamlit Cloud
# ============================================================================
#
# Copia este archivo como referencia para configurar los secretos en
# Streamlit Cloud (Settings > Secrets)
#
# NO COMMITEAR .streamlit/secrets.toml con valores reales
#
# ============================================================================

[api]
# URL de la API backend
url = "https://your-api-url.run.app"

[mlflow]
# MLflow tracking URI (opcional)
tracking_uri = "https://your-mlflow-server.com"

# ============================================================================
# Ejemplo de configuración en Streamlit Cloud:
#
# 1. Ve a tu app en share.streamlit.io
# 2. Click en "Settings" > "Secrets"
# 3. Copia y pega el contenido de abajo (con valores reales):
#
# [api]
# url = "https://tu-api-real.run.app"
#
# [mlflow]
# tracking_uri = "https://tu-mlflow-real.com"
# ============================================================================
EOF

print_success "Archivo de secretos de ejemplo creado"

# ============================================================================
# CREAR ARCHIVO .STREAMLIT/CONFIG.TOML PARA CLOUD
# ============================================================================

print_header "⚙️  Verificando configuración"

if [ -f ".streamlit/config.toml" ]; then
    print_success "config.toml ya existe"
else
    print_warning "config.toml no encontrado, creando uno básico..."
    
    cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[runner]
magicEnabled = true
fastReruns = true
EOF
    
    print_success "config.toml creado"
fi

# ============================================================================
# VERIFICAR ESTRUCTURA DEL PROYECTO
# ============================================================================

print_header "🔍 Verificando estructura del proyecto"

REQUIRED_FILES=(
    "src/streamlit_app/app.py"
    "src/streamlit_app/pages/__init__.py"
    "src/streamlit_app/pages/home.py"
    "src/streamlit_app/pages/prediction.py"
    "src/streamlit_app/pages/chatbot.py"
    "requirements.txt"
    ".streamlit/config.toml"
)

ALL_EXIST=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "Encontrado: $file"
    else
        print_error "Faltante: $file"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    print_error "Algunos archivos requeridos faltan"
    exit 1
fi

print_success "Todos los archivos requeridos están presentes"

# ============================================================================
# ACTUALIZAR .GITIGNORE
# ============================================================================

print_header "📝 Actualizando .gitignore"

# Asegurar que secrets.toml NO se commitee
if ! grep -q "secrets.toml" .gitignore 2>/dev/null; then
    echo "" >> .gitignore
    echo "# Streamlit secrets" >> .gitignore
    echo ".streamlit/secrets.toml" >> .gitignore
    print_success ".gitignore actualizado"
else
    print_success ".gitignore ya contiene secrets.toml"
fi

# ============================================================================
# CREAR ARCHIVO DE DOCUMENTACIÓN
# ============================================================================

print_header "📚 Creando documentación de deployment"

cat > STREAMLIT_CLOUD_DEPLOYMENT.md << 'EOF'
# Deployment en Streamlit Cloud

## 🚀 Pasos para deployar

### 1. Preparar el repositorio

Ejecuta el script de preparación:
```bash
./scripts/deploy_streamlit_cloud.sh
```

### 2. Commit y push a GitHub

```bash
git add .
git commit -m "feat: add streamlit app (US-032, US-033)"
git push origin main
```

### 3. Deployar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io/)
2. Inicia sesión con GitHub
3. Click en "New app"
4. Configura el deployment:
   - **Repository**: DanteA0179/mlops_proyecto_atreides
   - **Branch**: main
   - **Main file path**: src/streamlit_app/app.py
   - **App URL**: energy-optimization-ai (o el que prefieras)

### 4. Configurar secretos

1. En la página de tu app, click en "Settings" > "Secrets"
2. Agrega la configuración:

```toml
[api]
url = "https://tu-api-url.run.app"

[mlflow]
tracking_uri = "https://tu-mlflow-url.com"
```

3. Click en "Save"

### 5. Deploy!

Click en "Deploy!" y espera a que la app se construya (3-5 minutos).

## 🔗 URLs

Una vez deployada, tu app estará disponible en:
- `https://[tu-app-name].streamlit.app`

## 🔄 Actualizar la app

Simplemente haz push de cambios a GitHub:
```bash
git add .
git commit -m "update: ..."
git push origin main
```

Streamlit Cloud detectará los cambios y re-deployará automáticamente.

## 🐛 Troubleshooting

### La app no inicia

1. Revisa los logs en Streamlit Cloud
2. Verifica que `requirements.txt` esté actualizado
3. Asegúrate de que `src/streamlit_app/app.py` existe

### Error de importación

- Verifica que todas las dependencias estén en `requirements.txt`
- Asegúrate de que los paths de import sean correctos

### API no conecta

- Verifica la URL de la API en Secrets
- Asegúrate de que la API esté deployada y accesible

## 📊 Recursos

- **Límites gratuitos**:
  - 1 GB RAM
  - 1 CPU core
  - Sin límite de apps públicas
  
- **Documentación**: https://docs.streamlit.io/streamlit-community-cloud

## 🔒 Seguridad

- **NO** commitear `secrets.toml` con valores reales
- Usar Streamlit Cloud Secrets para credenciales
- La app pública NO expone los secretos
EOF

print_success "Documentación creada: STREAMLIT_CLOUD_DEPLOYMENT.md"

# ============================================================================
# RESUMEN Y PRÓXIMOS PASOS
# ============================================================================

print_header "✅ Preparación Completa"

echo "Archivos generados/actualizados:"
echo "  ✓ requirements.txt"
echo "  ✓ .streamlit/secrets.toml.example"
echo "  ✓ .streamlit/config.toml"
echo "  ✓ .gitignore"
echo "  ✓ STREAMLIT_CLOUD_DEPLOYMENT.md"
echo ""

print_header "📝 Próximos Pasos"

cat << 'EOF'

1️⃣  Commit y push a GitHub:
    git add .
    git commit -m "feat: add streamlit app (US-032, US-033)"
    git push origin main

2️⃣  Ir a Streamlit Cloud:
    https://share.streamlit.io/

3️⃣  Crear nueva app:
    • Repository: DanteA0179/mlops_proyecto_atreides
    • Branch: main
    • Main file: src/streamlit_app/app.py
    • Python version: 3.11

4️⃣  Configurar secretos:
    Settings > Secrets > Pegar contenido de .streamlit/secrets.toml.example
    (con valores reales)

5️⃣  Deploy!
    Click en "Deploy!" y esperar 3-5 minutos

📖 Lee STREAMLIT_CLOUD_DEPLOYMENT.md para más detalles

EOF

print_header "🎉 Script Completado"

print_success "Repositorio listo para deployment en Streamlit Cloud"
