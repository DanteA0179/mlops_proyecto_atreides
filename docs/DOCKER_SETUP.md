# Docker Setup - Energy Optimization API

Guía completa para Docker, FastAPI, MLflow y Prefect (US-003 & US-004).

## 🚀 Quick Start

### Opción 1: Docker Compose (Recomendado)

```bash
# Construir y levantar todos los servicios
docker-compose up --build

# Solo la API (más rápido)
docker-compose up api

# En background
docker-compose up -d
```

### Opción 2: Docker Build Manual

```bash
# Build de la API ligera (más rápido, ~2-3 minutos)
docker build -f Dockerfile.api -t energy-optimization-api:latest .

# Run
docker run -p 8000:8000 energy-optimization-api:latest
```

## 📦 Dockerfiles Disponibles

### `Dockerfile.api` - API + MLOps (Recomendado)
- ✅ **Rápido**: Build en 3-5 minutos
- ✅ **Balanceado**: ~800MB
- ✅ **Incluye**: FastAPI, MLflow, Prefect, scikit-learn, XGBoost, LightGBM
- ❌ **Excluye**: PyTorch, Transformers (foundation models pesados)

**Usar para:**
- US-003: Docker + FastAPI
- US-004: MLflow + Prefect
- Desarrollo local
- CI/CD
- Deployment de producción (API + MLOps)

### `Dockerfile` - Completo (Para foundation models)
- ⚠️ **Lento**: Build en 15-20 minutos
- ⚠️ **Pesado**: ~5GB
- ✅ **Completo**: Todas las dependencias incluyendo PyTorch, Transformers
- ✅ Incluye foundation models (Chronos, TimesFM)

**Usar para:**
- Entrenamiento de foundation models
- Experimentación con LLMs
- Notebooks con modelos pesados

## ⚡ Optimizaciones Implementadas

### 1. Multi-stage Build
Separa la construcción de dependencias del runtime final.

### 2. .dockerignore
Excluye archivos innecesarios:
- Data files (usar DVC)
- Models (usar DVC)
- Tests
- Documentation
- IDE configs

### 3. Requirements Mínimos
`requirements-api.txt` solo incluye lo necesario para la API.

### 4. Cache de Layers
Las dependencias se instalan antes del código para aprovechar el cache.

## 🐛 Troubleshooting

### Error: Timeout al instalar dependencias

**Problema:** Red lenta o paquetes muy grandes (PyTorch, etc.)

**Solución 1:** Usar Dockerfile.api (más ligero)
```bash
docker build -f Dockerfile.api -t energy-optimization-api:latest .
```

**Solución 2:** Aumentar timeout de pip
```dockerfile
RUN pip install --no-cache-dir --timeout=300 -r requirements-api.txt
```

**Solución 3:** Usar mirror local de PyPI
```dockerfile
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements-api.txt
```

### Error: Build muy lento

**Causas comunes:**
1. Instalando todas las dependencias de Poetry (incluye dev)
2. Instalando PyTorch con CUDA
3. No usando .dockerignore
4. Copiando archivos grandes (data, models)

**Soluciones:**
```bash
# 1. Usar Dockerfile.api
docker build -f Dockerfile.api -t energy-optimization-api:latest .

# 2. Verificar .dockerignore
cat .dockerignore

# 3. Limpiar cache de Docker
docker builder prune -a

# 4. Build sin cache (para debugging)
docker build --no-cache -f Dockerfile.api -t energy-optimization-api:latest .
```

### Error: Imagen muy grande

```bash
# Ver tamaño de la imagen
docker images energy-optimization-api

# Analizar layers
docker history energy-optimization-api:latest

# Usar Dockerfile.api (más ligero)
docker build -f Dockerfile.api -t energy-optimization-api:latest .
```

## 🔧 Desarrollo Local

### Hot Reload con Docker Compose

```bash
# Levantar con hot reload
docker-compose up api

# Los cambios en ./src se reflejan automáticamente
```

### Ejecutar comandos dentro del container

```bash
# Shell interactivo
docker-compose exec api bash

# Ejecutar tests
docker-compose exec api pytest

# Ver logs
docker-compose logs -f api
```

## 🚢 Deployment

### Build para producción

```bash
# Build optimizado
docker build -f Dockerfile.api -t energy-optimization-api:prod .

# Tag para registry
docker tag energy-optimization-api:prod gcr.io/mlops-equipo-37/energy-api:latest

# Push a GCR
docker push gcr.io/mlops-equipo-37/energy-api:latest
```

### Variables de entorno

```bash
# Crear .env para producción
cat > .env.prod << EOF
LOG_LEVEL=info
WORKERS=4
PYTHONUNBUFFERED=1
EOF

# Run con env file
docker run --env-file .env.prod -p 8000:8000 energy-optimization-api:prod
```

## 📊 Comparación de Dockerfiles

| Característica | Dockerfile.api | Dockerfile |
|----------------|----------------|------------|
| Build Time | 3-5 min | 15-20 min |
| Image Size | ~800MB | ~5GB |
| FastAPI | ✅ | ✅ |
| MLflow | ✅ | ✅ |
| Prefect | ✅ | ✅ |
| scikit-learn | ✅ | ✅ |
| XGBoost/LightGBM | ✅ | ✅ |
| PyTorch | ❌ | ✅ |
| Transformers | ❌ | ✅ |
| US-003/004 | ✅ | ✅ |
| Producción | ✅ | ❌ |

## 🎯 Recomendaciones

### Para Desarrollo de API
```bash
docker-compose up api
```

### Para Producción
```bash
docker build -f Dockerfile.api -t energy-optimization-api:latest .
docker run -p 8000:8000 energy-optimization-api:latest
```

### Para Entrenamiento
```bash
# Usar Poetry localmente (más rápido)
poetry install
poetry run python src/training/train.py

# O usar Dockerfile completo
docker build -f Dockerfile -t energy-optimization-ml:latest .
```

## 📋 US-003 & US-004: Verificación

### US-003: Docker + FastAPI ✅

**Criterios cumplidos:**
- ✅ Dockerfile multi-stage funcional (`Dockerfile.api`)
- ✅ docker-compose.yml para desarrollo local
- ✅ FastAPI app responde en `/health`
- ✅ Build ~3-5 minutos (balanceado con MLOps)

**Endpoints disponibles:**

| Endpoint | Método | Descripción | Status |
|----------|--------|-------------|--------|
| `/` | GET | Root con información de API | 200 |
| `/health` | GET | Health check con timestamp | 200 |
| `/docs` | GET | Swagger UI (auto-generado) | 200 |
| `/redoc` | GET | ReDoc (auto-generado) | 200 |

**Respuesta de /health:**
```json
{
  "status": "healthy",
  "service": "energy-optimization-api",
  "version": "0.1.0",
  "timestamp": "2025-10-07T14:52:42.319019"
}
```

**Verificar:**
```bash
# 1. Build y run
docker-compose up api -d

# 2. Health check
curl http://localhost:8000/health

# 3. Docs
open http://localhost:8000/docs

# 4. Verificación automática
poetry run python scripts/verify_us003_us004.ps1  # Windows
./scripts/verify_us003_us004.sh                    # Linux/Mac
```

### US-004: MLflow + Prefect ✅

**Servicios disponibles:**
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200
- API: http://localhost:8000

**Verificar MLflow:**
```bash
# Opción 1: Test rápido local (< 5 segundos)
poetry run python src/experiments/simple_mlflow_test.py

# Opción 2: Con servidor Docker (puede tardar)
docker-compose up mlflow -d
poetry run python src/experiments/example_mlflow.py
```

**Verificar Prefect:**
```bash
# Ejecutar flow de ejemplo
poetry run python src/flows/example_flow.py

# Resultado esperado:
# ✅ Flow completa en ~1 segundo
# ✅ Logs visibles en terminal
# ✅ 100 records procesados
```

## 🎯 Servicios y Puertos

| Servicio | Puerto | URL | Descripción |
|----------|--------|-----|-------------|
| FastAPI | 8000 | http://localhost:8000 | API principal |
| FastAPI Docs | 8000 | http://localhost:8000/docs | Swagger UI |
| MLflow | 5000 | http://localhost:5000 | Tracking de experimentos |
| Prefect | 4200 | http://localhost:4200 | Orquestación de workflows |

## 📚 Referencias

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prefect Documentation](https://docs.prefect.io/)
