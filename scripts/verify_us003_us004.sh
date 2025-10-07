#!/bin/bash
# Script de verificación para US-003 y US-004

set -e

echo "🔍 Verificando US-003 y US-004..."
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para verificar
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1${NC}"
        exit 1
    fi
}

# US-003: Docker + FastAPI
echo "📦 US-003: Docker + FastAPI"
echo "─────────────────────────────"

# 1. Verificar archivos
echo -n "Verificando Dockerfile.api... "
[ -f "Dockerfile.api" ]
check "Dockerfile.api existe"

echo -n "Verificando docker-compose.yml... "
[ -f "docker-compose.yml" ]
check "docker-compose.yml existe"

echo -n "Verificando requirements-api.txt... "
[ -f "requirements-api.txt" ]
check "requirements-api.txt existe"

echo -n "Verificando .dockerignore... "
[ -f ".dockerignore" ]
check ".dockerignore existe"

echo -n "Verificando src/api/main.py... "
[ -f "src/api/main.py" ]
check "src/api/main.py existe"

# 2. Verificar que los servicios están corriendo
echo ""
echo "🚀 Verificando servicios..."
echo ""

echo -n "Verificando API (http://localhost:8000/health)... "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    check "API responde"
else
    echo -e "${YELLOW}⚠️  API no está corriendo. Ejecuta: docker-compose up api -d${NC}"
fi

# US-004: MLflow + Prefect
echo ""
echo "📊 US-004: MLflow + Prefect"
echo "─────────────────────────────"

# 1. Verificar archivos
echo -n "Verificando src/experiments/example_mlflow.py... "
[ -f "src/experiments/example_mlflow.py" ]
check "example_mlflow.py existe"

echo -n "Verificando src/flows/example_flow.py... "
[ -f "src/flows/example_flow.py" ]
check "example_flow.py existe"

# 2. Verificar servicios
echo ""
echo "🚀 Verificando servicios MLOps..."
echo ""

echo -n "Verificando MLflow (http://localhost:5000)... "
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    check "MLflow responde"
else
    echo -e "${YELLOW}⚠️  MLflow no está corriendo. Ejecuta: docker-compose up mlflow -d${NC}"
fi

echo -n "Verificando Prefect (http://localhost:4200/api/health)... "
if curl -s http://localhost:4200/api/health > /dev/null 2>&1; then
    check "Prefect responde"
else
    echo -e "${YELLOW}⚠️  Prefect no está corriendo. Ejecuta: docker-compose up prefect -d${NC}"
fi

# Resumen
echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Verificación completada${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📋 Servicios disponibles:"
echo "   • API:     http://localhost:8000"
echo "   • Docs:    http://localhost:8000/docs"
echo "   • MLflow:  http://localhost:5000"
echo "   • Prefect: http://localhost:4200"
echo ""
echo "🧪 Comandos de prueba:"
echo "   • API:     curl http://localhost:8000/health"
echo "   • MLflow:  poetry run python src/experiments/example_mlflow.py"
echo "   • Prefect: poetry run python src/flows/example_flow.py"
echo ""
