# Energy Optimization Copilot API - Documentación Completa

**Versión**: 1.0.0  
**Fecha**: 16 de Noviembre, 2025  
**Estado**: Producción

---

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Arquitectura](#arquitectura)
3. [Autenticación](#autenticación)
4. [Endpoints](#endpoints)
5. [Modelos de Datos](#modelos-de-datos)
6. [Códigos de Error](#códigos-de-error)
7. [Rate Limiting](#rate-limiting)
8. [Mejores Prácticas](#mejores-prácticas)
9. [Monitoreo](#monitoreo)
10. [FAQ](#faq)

---

## 🎯 Introducción

La **Energy Optimization Copilot API** es una API RESTful que proporciona predicciones precisas de consumo energético para la industria siderúrgica utilizando modelos de Machine Learning avanzados.

### Características Principales

- ✅ **Predicciones Precisas**: RMSE < 13 kWh en test set
- ✅ **Alta Performance**: Latencia P95 < 100ms
- ✅ **Batch Processing**: Hasta 1000 predicciones por request
- ✅ **Intervalos de Confianza**: Cuantificación de incertidumbre
- ✅ **Monitoreo Integrado**: Métricas en tiempo real
- ✅ **Production Ready**: Deployado en Google Cloud Run

### Base URL

```
# Desarrollo Local
http://localhost:8000

# Desarrollo Docker
http://0.0.0.0:8000

# Producción (GCP Cloud Run)
https://energy-optimization-api.run.app
```

### Documentación Interactiva

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Schema**: `/openapi.json`

---

## 🏗️ Arquitectura

### Flujo de Datos

```
Cliente → FastAPI → Feature Engineering → ML Model → Respuesta
                          ↓
                    Monitoring (Logs)
```

### Stack Tecnológico

**Backend**:
- FastAPI 0.104+
- Pydantic 2.0+
- Uvicorn/Gunicorn

**ML**:
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- Polars (feature engineering)

**Monitoreo**:
- MLflow (model tracking)
- Evidently AI (drift detection)
- Logging estructurado

**Deployment**:
- Docker
- Google Cloud Run
- Cloud Storage (DVC)

---

## 🔐 Autenticación

### Estado Actual (v1.0.0)

**La API actualmente NO requiere autenticación** para facilitar el desarrollo y testing inicial.

### Próximamente (v1.1.0)

Se implementarán los siguientes métodos de autenticación:

#### 1. API Key

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

#### 2. JWT Bearer Token

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

#### Obtener API Key (Futuro)

```bash
POST /auth/register
{
  "email": "user@example.com",
  "organization": "Steel Plant A"
}
```

---

## 📡 Endpoints

### Root

#### `GET /`

Información general de la API.

**Response**:
```json
{
  "message": "Welcome to Energy Optimization Copilot API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": { ... }
}
```

---

### Predictions

#### `POST /predict`

Predice consumo energético para una observación individual.

**Request Body**:
```json
{
  "lagging_reactive_power": 23.45,
  "leading_reactive_power": 12.30,
  "co2": 0.05,
  "lagging_power_factor": 0.85,
  "leading_power_factor": 0.92,
  "nsm": 36000,
  "day_of_week": 1,
  "load_type": "Medium"
}
```

**Response** (200 OK):
```json
{
  "predicted_usage_kwh": 45.67,
  "confidence_interval_lower": 42.10,
  "confidence_interval_upper": 49.24,
  "model_version": "stacking_ensemble_v1",
  "model_type": "stacking_ensemble",
  "prediction_timestamp": "2025-11-16T10:30:00Z",
  "features_used": 18,
  "prediction_id": "pred_8f3a9b2c"
}
```

**Validaciones**:
- `lagging_reactive_power` >= 0
- `leading_reactive_power` >= 0
- `co2` >= 0
- `lagging_power_factor`: 0-1
- `leading_power_factor`: 0-1
- `nsm`: 0-86400 (segundos en el día)
- `day_of_week`: 0-6 (0=Lunes)
- `load_type`: "Light", "Medium", "Maximum"

---

#### `POST /predict/batch`

Predice consumo para múltiples observaciones.

**Request Body**:
```json
{
  "predictions": [
    {
      "lagging_reactive_power": 15.20,
      "leading_reactive_power": 8.50,
      "co2": 0.03,
      "lagging_power_factor": 0.88,
      "leading_power_factor": 0.95,
      "nsm": 7200,
      "day_of_week": 1,
      "load_type": "Light"
    },
    { ... }
  ]
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "predicted_usage_kwh": 28.34,
      "prediction_id": "pred_abc123"
    },
    { ... }
  ],
  "summary": {
    "total_predictions": 2,
    "avg_predicted_usage": 37.00,
    "min_predicted_usage": 28.34,
    "max_predicted_usage": 45.67,
    "processing_time_ms": 45.32
  },
  "model_version": "stacking_ensemble_v1",
  "batch_timestamp": "2025-11-16T10:30:00Z"
}
```

**Límites**:
- Mínimo: 1 predicción
- Máximo: 1000 predicciones
- Timeout: 30 segundos

---

### Health

#### `GET /health`

Estado de salud del servicio.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "service": "energy-optimization-api",
  "version": "1.0.0",
  "timestamp": "2025-11-16T10:30:00Z",
  "model_loaded": true,
  "model_version": "stacking_ensemble_v1",
  "uptime_seconds": 3600.50,
  "memory_usage_mb": 512.34,
  "cpu_usage_percent": 15.20
}
```

**Estados Posibles**:
- `healthy`: Servicio operativo
- `degraded`: Servicio operativo con recursos limitados
- `unhealthy`: Servicio con errores críticos

**Uso en Load Balancers**:
```yaml
health_check:
  path: /health
  interval: 30s
  timeout: 5s
  healthy_threshold: 2
  unhealthy_threshold: 3
```

---

### Model

#### `GET /model/info`

Información detallada del modelo.

**Response** (200 OK):
```json
{
  "model_type": "stacking_ensemble",
  "model_version": "stacking_ensemble_v1",
  "model_name": "Stacking Ensemble",
  "trained_on": "2025-11-16T10:30:00Z",
  "training_dataset": {
    "name": "steel_featured.parquet",
    "samples": 27928,
    "features": 18
  },
  "base_models": [
    {"name": "XGBoost", "contribution_pct": 19.3},
    {"name": "LightGBM", "contribution_pct": 40.5},
    {"name": "CatBoost", "contribution_pct": 40.2}
  ],
  "training_metrics": {
    "rmse": 12.7982,
    "r2": 0.8702,
    "mae": 3.4731,
    "mape": 7.01
  },
  "features": [ ... ],
  "mlflow_run_id": "abc123",
  "artifact_location": "/path/to/model"
}
```

---

#### `GET /model/metrics`

Métricas actuales del modelo.

**Response** (200 OK):
```json
{
  "model_version": "stacking_ensemble_v1",
  "timestamp": "2025-11-16T10:30:00Z",
  "training_metrics": {
    "rmse": 12.7982,
    "r2": 0.8702,
    "mae": 3.4731
  },
  "production_metrics": {
    "total_predictions": 15234,
    "predictions_last_24h": 1523,
    "avg_prediction_time_ms": 45.32,
    "p95_prediction_time_ms": 78.10,
    "p99_prediction_time_ms": 95.50,
    "error_rate_percent": 0.02
  },
  "load_type_distribution": {
    "Light": 4523,
    "Medium": 7234,
    "Maximum": 3477
  },
  "prediction_distribution": {
    "min": 18.23,
    "max": 95.67,
    "mean": 45.32,
    "median": 43.10,
    "std": 12.45
  },
  "system_health": {
    "memory_usage_mb": 512.34,
    "cpu_usage_percent": 15.20,
    "uptime_seconds": 3600.50
  }
}
```

---

## 📊 Modelos de Datos

### PredictionRequest

| Campo | Tipo | Rango | Descripción |
|-------|------|-------|-------------|
| `lagging_reactive_power` | float | >= 0 | Potencia reactiva en atraso (kVarh) |
| `leading_reactive_power` | float | >= 0 | Potencia reactiva en adelanto (kVarh) |
| `co2` | float | >= 0 | Emisiones de CO2 (tCO2) |
| `lagging_power_factor` | float | 0-1 | Factor de potencia en atraso |
| `leading_power_factor` | float | 0-1 | Factor de potencia en adelanto |
| `nsm` | int | 0-86400 | Segundos desde medianoche |
| `day_of_week` | int | 0-6 | Día de la semana (0=Lunes) |
| `load_type` | string | enum | "Light", "Medium", "Maximum" |

### PredictionResponse

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `predicted_usage_kwh` | float | Consumo predicho en kWh |
| `confidence_interval_lower` | float | Límite inferior (95%) |
| `confidence_interval_upper` | float | Límite superior (95%) |
| `model_version` | string | Versión del modelo |
| `model_type` | string | Tipo de arquitectura |
| `prediction_timestamp` | string | Timestamp ISO 8601 |
| `features_used` | int | Número de features |
| `prediction_id` | string | ID único |

---

## ⚠️ Códigos de Error

### HTTP Status Codes

| Código | Significado | Descripción |
|--------|-------------|-------------|
| 200 | OK | Solicitud exitosa |
| 400 | Bad Request | Error en parámetros del batch |
| 422 | Unprocessable Entity | Error de validación |
| 500 | Internal Server Error | Error del servidor |
| 503 | Service Unavailable | Modelo no cargado |

### Errores Comunes

#### 422 - Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "load_type"],
      "msg": "load_type must be one of ['Light', 'Medium', 'Maximum']",
      "type": "value_error"
    }
  ]
}
```

**Solución**: Verificar que todos los campos cumplan con las validaciones.

#### 400 - Batch Limit Exceeded

```json
{
  "detail": "Batch cannot exceed 1000 predictions"
}
```

**Solución**: Dividir el batch en lotes más pequeños.

#### 503 - Model Not Loaded

```json
{
  "detail": "Model not loaded"
}
```

**Solución**: Verificar el health check (`/health`).

---

## 🚦 Rate Limiting

### Estado Actual (v1.0.0)

**No hay límites de rate** actualmente implementados.

### Próximamente (v1.1.0)

Se implementarán los siguientes límites:

| Tier | Límite | Burst |
|------|--------|-------|
| Free | 100 req/min | 20 |
| Basic | 1000 req/min | 50 |
| Pro | 10000 req/min | 200 |

**Headers de Respuesta**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1700000000
```

---

## 🎯 Mejores Prácticas

### 1. Uso Eficiente de Batch

```python
# ✅ BUENO: Usar batch para múltiples predicciones
batch_data = {"predictions": [pred1, pred2, pred3, ...]}
response = requests.post(f"{API_URL}/predict/batch", json=batch_data)

# ❌ MALO: Múltiples requests individuales
for pred in predictions:
    response = requests.post(f"{API_URL}/predict", json=pred)
```

### 2. Manejo de Errores

```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(f"{API_URL}/predict", json=data, timeout=5)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code}")
    print(e.response.json())
except RequestException as e:
    print(f"Request failed: {e}")
```

### 3. Validación Previa

```python
def validate_prediction_request(data):
    """Validate request before sending to API."""
    assert 0 <= data["lagging_power_factor"] <= 1, "Invalid power factor"
    assert 0 <= data["nsm"] <= 86400, "Invalid NSM"
    assert data["load_type"] in ["Light", "Medium", "Maximum"]
    return data
```

### 4. Retry con Backoff

```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### 5. Caching de Resultados

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def get_prediction(request_hash):
    """Cache predictions for identical requests."""
    # Implementation
    pass

# Usage
request_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
result = get_prediction(request_hash)
```

---

## 📈 Monitoreo

### Health Checks

```bash
# Check cada 30 segundos
while true; do
  curl -s http://localhost:8000/health | jq '.status'
  sleep 30
done
```

### Métricas con Grafana

```python
# Exportar métricas para Prometheus
import prometheus_client

prediction_counter = prometheus_client.Counter(
    'predictions_total',
    'Total predictions made'
)

prediction_latency = prometheus_client.Histogram(
    'prediction_latency_seconds',
    'Prediction latency'
)
```

### Alertas Sugeridas

1. **Latencia Alta**: `p95_prediction_time_ms > 200`
2. **Error Rate Alto**: `error_rate_percent > 1.0`
3. **Memoria Alta**: `memory_usage_mb > 1000`
4. **CPU Alta**: `cpu_usage_percent > 80`
5. **Modelo No Cargado**: `model_loaded == false`

---

## ❓ FAQ

### ¿Cuál es la latencia típica?

- **Single prediction**: ~50ms (P95: <100ms)
- **Batch 100 items**: ~200ms
- **Batch 1000 items**: ~1500ms

### ¿Qué modelo está activo?

Consultar `/model/info` para ver el modelo actual.

### ¿Cómo interpretar intervalos de confianza?

Los intervalos representan el rango donde el valor real tiene 95% de probabilidad de estar.

### ¿Puedo hacer predicciones para fechas futuras?

Sí, ajustar `nsm` y `day_of_week` según la fecha deseada.

### ¿Hay límite de requests?

Actualmente no, pero se implementarán en v1.1.0.

### ¿Qué hacer si el modelo falla?

1. Verificar `/health`
2. Reiniciar el servicio
3. Consultar logs
4. Contactar soporte

---

## 📞 Soporte

**GitHub Issues**: [Reportar problema](https://github.com/DanteA0179/mlops_proyecto_atreides/issues)  
**Email**: mlops@atreides.com  
**Documentación**: [GitHub Docs](https://github.com/DanteA0179/mlops_proyecto_atreides/tree/main/docs)

---

**Documento generado**: 16 de Noviembre, 2025  
**Versión**: 1.0  
**Última actualización**: 2025-11-16
