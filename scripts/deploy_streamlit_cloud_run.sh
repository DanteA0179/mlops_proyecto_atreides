#!/bin/bash

# ============================================================================
# Script de Deployment a Google Cloud Run
# Energy Optimization AI - Streamlit Frontend
# ============================================================================
#
# Este script automatiza el deployment de la aplicación Streamlit a
# Google Cloud Run.
#
# Prerrequisitos:
#   - gcloud CLI instalado y configurado
#   - Proyecto de GCP creado
#   - Permisos necesarios en GCP
#   - Docker instalado
#
# Uso:
#   ./scripts/deploy_streamlit_cloud_run.sh
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
# CONFIGURACIÓN
# ============================================================================

# Configuración del proyecto GCP
PROJECT_ID="${GCP_PROJECT_ID:-mlops-atreides-2025}"
REGION="${GCP_REGION:-us-central1}"

# Configuración del servicio
SERVICE_NAME="energy-optimization-streamlit"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Configuración de recursos
MEMORY="2Gi"
CPU="2"
MIN_INSTANCES="0"
MAX_INSTANCES="5"
TIMEOUT="300"
PORT="8501"

# URL de la API (debe estar deployada previamente)
API_URL="${API_URL:-https://energy-optimization-api-xxxxx.run.app}"

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
# VERIFICACIONES PREVIAS
# ============================================================================

print_header "🚀 Iniciando deployment de Streamlit a Cloud Run"

# Verificar que gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI no está instalado"
    echo "Instala gcloud desde: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_success "gcloud CLI encontrado"

# Verificar autenticación
print_info "Verificando autenticación..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "No hay cuenta activa"
    print_info "Iniciando proceso de autenticación..."
    gcloud auth login
fi

print_success "Autenticación verificada"

# Configurar proyecto
print_info "Configurando proyecto: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# ============================================================================
# HABILITAR APIs NECESARIAS
# ============================================================================

print_header "🔧 Habilitando APIs necesarias"

APIS=(
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "cloudbuild.googleapis.com"
)

for api in "${APIS[@]}"; do
    print_info "Habilitando ${api}..."
    gcloud services enable ${api} --project=${PROJECT_ID}
done

print_success "APIs habilitadas"

# ============================================================================
# BUILD DE LA IMAGEN
# ============================================================================

print_header "🏗️  Construyendo imagen Docker"

print_info "Imagen: ${IMAGE_NAME}"
print_info "Esto puede tomar varios minutos..."

# Opción 1: Build local y push
# docker build -f Dockerfile.streamlit -t ${IMAGE_NAME} .
# docker push ${IMAGE_NAME}

# Opción 2: Cloud Build (recomendado)
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=20m \
    --machine-type=n1-highcpu-8 \
    --project=${PROJECT_ID} \
    -f Dockerfile.streamlit \
    .

print_success "Imagen construida y subida a Container Registry"

# ============================================================================
# DEPLOYMENT A CLOUD RUN
# ============================================================================

print_header "🚢 Deployando a Cloud Run"

print_info "Servicio: ${SERVICE_NAME}"
print_info "Región: ${REGION}"
print_info "Memoria: ${MEMORY}"
print_info "CPU: ${CPU}"
print_info "Instancias: ${MIN_INSTANCES}-${MAX_INSTANCES}"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port ${PORT} \
    --memory ${MEMORY} \
    --cpu ${CPU} \
    --min-instances ${MIN_INSTANCES} \
    --max-instances ${MAX_INSTANCES} \
    --timeout ${TIMEOUT} \
    --set-env-vars "API_URL=${API_URL}" \
    --set-env-vars "STREAMLIT_SERVER_PORT=${PORT}" \
    --set-env-vars "STREAMLIT_SERVER_ADDRESS=0.0.0.0" \
    --set-env-vars "STREAMLIT_SERVER_HEADLESS=true" \
    --project=${PROJECT_ID}

print_success "Deployment completado"

# ============================================================================
# OBTENER URL DEL SERVICIO
# ============================================================================

print_header "📋 Información del Servicio"

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project=${PROJECT_ID})

print_success "Deployment exitoso!"
echo ""
print_info "URL del servicio: ${SERVICE_URL}"
echo ""

# ============================================================================
# VERIFICACIÓN
# ============================================================================

print_header "🔍 Verificando deployment"

print_info "Esperando 30 segundos para que el servicio esté listo..."
sleep 30

print_info "Verificando health endpoint..."
if curl -f -s "${SERVICE_URL}/_stcore/health" > /dev/null; then
    print_success "Servicio respondiendo correctamente"
else
    print_warning "El servicio aún no responde. Puede tardar unos minutos."
fi

# ============================================================================
# PRÓXIMOS PASOS
# ============================================================================

print_header "📝 Próximos Pasos"

echo "1. Accede a la aplicación en: ${SERVICE_URL}"
echo ""
echo "2. Actualiza la variable API_URL si es necesario:"
echo "   gcloud run services update ${SERVICE_NAME} \\"
echo "     --set-env-vars API_URL=https://tu-api-url.run.app \\"
echo "     --region ${REGION}"
echo ""
echo "3. Ver logs del servicio:"
echo "   gcloud run logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "4. Para actualizar el servicio después de cambios:"
echo "   ./scripts/deploy_streamlit_cloud_run.sh"
echo ""

print_header "✨ Deployment Completo"

# ============================================================================
# COMANDOS ÚTILES
# ============================================================================

cat << EOF

🔧 Comandos útiles:

# Ver logs en tiempo real
gcloud run logs tail ${SERVICE_NAME} --region ${REGION}

# Ver detalles del servicio
gcloud run services describe ${SERVICE_NAME} --region ${REGION}

# Listar revisiones
gcloud run revisions list --service ${SERVICE_NAME} --region ${REGION}

# Actualizar variables de entorno
gcloud run services update ${SERVICE_NAME} \\
  --set-env-vars KEY=VALUE \\
  --region ${REGION}

# Escalar servicio
gcloud run services update ${SERVICE_NAME} \\
  --min-instances 1 \\
  --max-instances 10 \\
  --region ${REGION}

# Eliminar servicio
gcloud run services delete ${SERVICE_NAME} --region ${REGION}

EOF

print_success "Script completado exitosamente"
