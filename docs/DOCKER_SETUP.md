# Docker Setup - Energy Optimization API

Guía completa para Docker, FastAPI, MLflow y Prefect (US-003, US-004 & US-022).

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

### `Dockerfile.api` - API Production-Ready (Recomendado)
- ✅ **Build**: 3-5 minutos
- ✅ **Tamaño**: ~2.97GB
- ✅ **Incluye**: FastAPI, scikit-learn, XGBoost, LightGBM, CatBoost
- ✅ **Modelos**: Pickle models (stacking_ensemble)
- ❌ **Excluye**: PyTorch, Transformers, ONNX Runtime

**Usar para:**
- US-003: Docker + FastAPI
- US-020: API con modelos pickle
- US-022: Dockerización optimizada
- Desarrollo local
- CI/CD
- Deployment de producción

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


---

## 🐳 US-022: Dockerización Production-Ready

### Multi-stage Dockerfile

El `Dockerfile.api` usa un build multi-stage para optimizar el build:

**Stage 1 (Builder)**:
- Instala build dependencies
- Crea virtual environment
- Instala requirements-api-minimal.txt

**Stage 2 (Runtime)**:
- Imagen de producción
- Copia solo el venv
- Non-root user (appuser)
- Modelos pickle embebidos

**Resultado**: ~2.97GB (incluye XGBoost, LightGBM, CatBoost completos)

### Estrategia de Modelos

**Modelos Pickle (Solución Final)**:
- Modelos en `/app/models/ensembles`, `/app/models/gradient_boosting`, `/app/models/baselines`
- Compatible con NumPy 2.x
- Configuración probada en US-020
- Modelo por defecto: `stacking_ensemble`

**¿Por qué NO ONNX?**:
- ONNX Runtime 1.18.0 incompatible con NumPy 2.x
- Preferimos NumPy actualizado sobre ONNX
- Modelos pickle funcionan perfectamente

### Scripts de Validación

```bash
# Validar setup completo
.\scripts\validate_docker_setup.ps1

# Build con validación automática
.\scripts\docker_build.ps1  # Windows
bash scripts/docker_build.sh  # Linux/Mac
```

### Métricas Alcanzadas

| Métrica | Resultado | Estado |
|---------|-----------|--------|
| Tamaño de imagen | 2.97GB | ✅ |
| Build time | ~3 min | ✅ |
| Startup time | ~15 seg | ✅ |
| Health check | healthy | ✅ |
| Predicción | Funciona | ✅ |

### CI/CD con GitHub Actions

**Workflows disponibles**:
- `.github/workflows/docker-build.yml` - Build automático y tests
- `.github/workflows/deploy-cloudrun.yml` - Deployment a GCP Cloud Run

**Features**:
- Build automático en push/PR
- Tests de health y prediction
- Validación de tamaño de imagen
- Cache de layers con GitHub Actions
- Deployment automático a Cloud Run

### Deployment Multi-Cloud

**GCP Cloud Run**:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/energy-api
gcloud run deploy energy-api --image gcr.io/PROJECT_ID/energy-api
```

**AWS ECS**: Ver task definition en `docs/us-resolved/us-022.md`

**Azure Container Apps**:
```bash
az acr build --registry myregistry --image energy-api:latest .
az containerapp create --name energy-api --image myregistry.azurecr.io/energy-api:latest
```

### Comandos Útiles

```bash
# Build
docker build -f Dockerfile.api -t energy-api:latest .

# Run con variables de entorno
docker run -p 8000:8000 \
  -e MODEL_TYPE=xgboost \
  -e LOG_LEVEL=debug \
  energy-api:latest

# Ver logs
docker logs -f CONTAINER_ID

# Stats (CPU, memoria)
docker stats CONTAINER_ID

# Inspeccionar imagen
docker history energy-api:latest
```

### Troubleshooting US-022

**Problema: Imagen > 1.5GB**
```bash
# Verificar .dockerignore
cat .dockerignore

# Ver history de layers
docker history energy-api:latest
```

**Problema: Modelos no cargan**
```bash
# Verificar modelos en imagen
docker run -it energy-api:latest ls -la /app/models/onnx

# Ver logs de carga
docker logs CONTAINER_ID | grep "model"
```

**Problema: Health check falla**
```bash
# Verificar healthcheck
docker inspect --format='{{json .State.Health}}' CONTAINER_ID

# Test manual
docker exec CONTAINER_ID curl -f http://localhost:8000/health
```

---

## 📚 Documentación Adicional

- **US-022 Completa**: `docs/us-resolved/us-022.md`
- **Planeación US-022**: `docs/us-planning/us-022.md`
- **Scripts**: `scripts/docker_build.ps1`, `scripts/validate_docker_setup.ps1`
- **CI/CD**: `.github/workflows/docker-build.yml`

---

**Última actualización**: 15 de Noviembre, 2025  
**Versión**: 2.0 (incluye US-022)
