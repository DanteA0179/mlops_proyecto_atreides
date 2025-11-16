# Docker Usage Guide

## 📋 Archivos Docker Disponibles

### 1. `docker-compose.yml` - PRODUCCIÓN ✅

**Usar para**: Testing de producción, deployment, CI/CD

**Características**:
- ✅ Gunicorn con 2 workers
- ✅ Modelos embebidos en imagen
- ✅ Sin hot reload
- ✅ Resource limits (CPU: 2, Memory: 2G)
- ✅ Healthcheck configurado

**Comandos**:
```bash
# Build y run
docker-compose up api

# En background
docker-compose up -d api

# Ver logs
docker-compose logs -f api

# Detener
docker-compose down
```

### 2. `docker-compose.dev.yml` - DESARROLLO 🔧

**Usar para**: Desarrollo local con hot reload

**Características**:
- ✅ Uvicorn con --reload
- ✅ Volúmenes montados (cambios en código se reflejan automáticamente)
- ✅ LOG_LEVEL=debug
- ✅ Acceso a data/ y reports/

**Comandos**:
```bash
# Build y run en modo desarrollo
docker-compose -f docker-compose.dev.yml up

# En background
docker-compose -f docker-compose.dev.yml up -d

# Ver logs
docker-compose -f docker-compose.dev.yml logs -f api

# Detener
docker-compose -f docker-compose.dev.yml down
```

---

## 🎯 ¿Cuál usar?

### Desarrollo Local (editando código)
```bash
docker-compose -f docker-compose.dev.yml up
```
- Cambios en `src/` se reflejan automáticamente
- No necesitas rebuild
- Logs más verbosos

### Testing de Producción
```bash
docker-compose up api
```
- Simula ambiente de producción
- Usa Gunicorn (múltiples workers)
- Modelos embebidos

### CI/CD
```bash
docker-compose build api
docker-compose up -d api
# Run tests
docker-compose down
```

---

## 📊 Comparación

| Característica | Production | Development |
|----------------|------------|-------------|
| Archivo | `docker-compose.yml` | `docker-compose.dev.yml` |
| Server | Gunicorn + Uvicorn | Uvicorn solo |
| Workers | 2 | 1 |
| Hot Reload | ❌ | ✅ |
| Volúmenes | ❌ | ✅ |
| Resource Limits | ✅ | ❌ |
| Log Level | info | debug |
| Uso | Producción, CI/CD | Desarrollo local |

---

## 🚀 Endpoints Disponibles

Ambos modos exponen los mismos endpoints:

- `GET /` - Root
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `POST /predict` - Predicción individual
- `POST /predict/batch` - Predicción batch
- `GET /model/info` - Información del modelo

---

## 💡 Tips

### Desarrollo Rápido
```bash
# Terminal 1: API con hot reload
docker-compose -f docker-compose.dev.yml up api

# Terminal 2: MLflow
docker-compose -f docker-compose.dev.yml up mlflow

# Edita código en src/ y los cambios se aplican automáticamente
```

### Testing de Producción
```bash
# Build imagen de producción
docker-compose build api

# Run y test
docker-compose up api
curl http://localhost:8000/health
```

### Limpiar Todo
```bash
# Detener y eliminar containers
docker-compose down
docker-compose -f docker-compose.dev.yml down

# Eliminar imágenes
docker rmi energy-optimization-api:latest
```

---

**Recomendación**: Usa `docker-compose.dev.yml` para desarrollo diario y `docker-compose.yml` para testing final antes de deployment.
