# Energy Optimization API

API RESTful para predicción de consumo energético en la industria siderúrgica.

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalar dependencias
poetry install

# Activar entorno virtual
poetry shell
```

### Ejecutar la API

```bash
# Opción 1: Usando uvicorn directamente
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Opción 2: Ejecutar el script main.py
python src/api/main.py
```

La API estará disponible en:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

## 📚 Endpoints Disponibles

### 1. POST /predict - Predicción Individual

Predice el consumo energético para una única observación.

**Request:**
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

**Response:**
```json
{
  "predicted_usage_kwh": 45.67,
  "confidence_interval_lower": 42.10,
  "confidence_interval_upper": 49.24,
  "model_version": "lightgbm_ensemble_v1",
  "model_type": "stacking_ensemble",
  "prediction_timestamp": "2025-11-05T10:30:00Z",
  "features_used": 18,
  "prediction_id": "pred_8f3a9b2c"
}
```

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "nsm": 36000,
    "day_of_week": 1,
    "load_type": "Medium"
  }'
```

### 2. POST /predict/batch - Predicción Batch

Predice el consumo energético para múltiples observaciones (máx. 1000).

**Request:**
```json
{
  "predictions": [
    {
      "lagging_reactive_power": 23.45,
      "leading_reactive_power": 12.30,
      "co2": 0.05,
      "lagging_power_factor": 0.85,
      "leading_power_factor": 0.92,
      "nsm": 36000,
      "day_of_week": 1,
      "load_type": "Medium"
    },
    ...
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "predicted_usage_kwh": 45.67,
      "prediction_id": "pred_8f3a9b2c"
    },
    ...
  ],
  "summary": {
    "total_predictions": 100,
    "avg_predicted_usage": 48.32,
    "min_predicted_usage": 12.45,
    "max_predicted_usage": 89.21,
    "processing_time_ms": 234
  },
  "model_version": "lightgbm_ensemble_v1",
  "batch_timestamp": "2025-11-05T10:30:00Z"
}
```

### 3. GET /health - Health Check

Verifica el estado del servicio.

**Response:**
```json
{
  "status": "healthy",
  "service": "energy-optimization-api",
  "version": "1.0.0",
  "timestamp": "2025-11-05T10:30:00Z",
  "model_loaded": true,
  "model_version": "lightgbm_ensemble_v1",
  "uptime_seconds": 3600,
  "memory_usage_mb": 256.5,
  "cpu_usage_percent": 15.2
}
```

### 4. GET /model/info - Información del Modelo

Retorna metadata detallada del modelo.

**Response:**
```json
{
  "model_type": "stacking_ensemble",
  "model_version": "lightgbm_ensemble_v1",
  "model_name": "LightGBM Stacking Ensemble",
  "trained_on": "2025-10-30T15:30:00Z",
  "training_dataset": {
    "name": "steel_featured.parquet",
    "samples": 27928,
    "features": 18
  },
  "base_models": [
    {
      "name": "XGBoost",
      "contribution_pct": 19.3
    },
    ...
  ],
  "training_metrics": {
    "rmse": 12.7982,
    "r2": 0.8702,
    "mae": 3.4731,
    "mape": 7.01
  },
  "features": [...],
  "mlflow_run_id": "fb35e48c...",
  "artifact_location": "models/ensembles/ensemble_lightgbm_v1.pkl"
}
```

### 5. GET /model/metrics - Métricas del Modelo

Retorna métricas de performance en producción.

**Response:**
```json
{
  "model_version": "lightgbm_ensemble_v1",
  "timestamp": "2025-11-05T10:30:00Z",
  "training_metrics": {
    "rmse": 12.7982,
    "r2": 0.8702,
    "mae": 3.4731,
    "mape": 7.01
  },
  "production_metrics": {
    "total_predictions": 5420,
    "predictions_last_24h": 1234,
    "avg_prediction_time_ms": 8.5,
    "p95_prediction_time_ms": 15.2,
    "error_rate_percent": 0.02
  },
  "system_health": {
    "memory_usage_mb": 256.5,
    "cpu_usage_percent": 15.2,
    "uptime_seconds": 3600
  }
}
```

## 🔧 Configuración

La configuración se encuentra en `src/api/utils/config.py`:

```python
class Config:
    APP_NAME = "Energy Optimization Copilot API"
    APP_VERSION = "1.0.0"
    LOG_LEVEL = "INFO"
    MODEL_TYPE = "stacking_ensemble"
    MLFLOW_TRACKING_URI = "http://localhost:5000"
```

## 🧪 Testing

### Ejecutar Tests Unitarios

```bash
# Todos los tests
poetry run pytest tests/unit/test_api_endpoints.py -v

# Con coverage
poetry run pytest tests/unit/test_api_endpoints.py --cov=src/api --cov-report=html

# Ver reporte de coverage
open htmlcov/index.html
```

### Ejemplo de Test
```python
def test_predict_valid_request():
    """Test successful prediction with valid request"""
    request_data = {
        "lagging_reactive_power": 23.45,
        "leading_reactive_power": 12.30,
        "co2": 0.05,
        "lagging_power_factor": 0.85,
        "leading_power_factor": 0.92,
        "nsm": 36000,
        "day_of_week": 1,
        "load_type": "Medium",
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    assert "predicted_usage_kwh" in response.json()
```

## 📊 Métricas de Calidad

| Métrica | Target | Estado |
|---------|--------|--------|
| Type hints | 100% | ✅ |
| Docstrings | 100% | ✅ |
| Test coverage | >80% | ✅ |
| Ruff warnings | 0 | ✅ |
| Black formatted | Sí | ✅ |

## 🏗️ Arquitectura

```
src/api/
├── main.py                     # FastAPI app + startup/shutdown
├── models/
│   ├── requests.py            # Pydantic request models
│   └── responses.py           # Pydantic response models
├── routes/
│   ├── predict.py             # Prediction endpoints
│   ├── health.py              # Health check
│   └── model.py               # Model info/metrics
├── services/
│   ├── model_service.py       # Model loading/inference
│   └── feature_engineering.py # Feature transformation
├── middleware/
│   ├── logging_middleware.py  # Request/response logging
│   └── error_handler.py       # Global error handling
└── utils/
    └── config.py              # Configuration
```

## 🔄 Flujo de Predicción

```
1. Request → Middleware (Logging)
2. Pydantic Validation
3. Feature Engineering (US-011 + US-012)
   - 9 original features
   - 7 temporal features
   - 2 one-hot encoded features
   = 18 features total
4. Model Inference (LightGBM Ensemble)
5. Response Formatting
6. Middleware (Logging) → Response
```

## 🐛 Troubleshooting

### Error: Model not found

```bash
# Verificar que el modelo existe
ls models/ensembles/ensemble_lightgbm_v1.pkl

# Si no existe, entrenar el modelo
python src/models/train_stacking_ensemble.py
```

### Error: Preprocessing pipeline not found

```bash
# Verificar que el pipeline existe
ls models/preprocessing/preprocessing_pipeline.pkl

# Si no existe, ejecutar US-012
python src/features/build_preprocessed_dataset.py
```

### Error: Port 8000 already in use

```bash
# Usar otro puerto
python -m uvicorn src.api.main:app --port 8001
```

## 📝 Validaciones Pydantic

### Campos Obligatorios

| Campo | Tipo | Rango | Descripción |
|-------|------|-------|-------------|
| `lagging_reactive_power` | float | ≥0 | Potencia reactiva en atraso (kVarh) |
| `leading_reactive_power` | float | ≥0 | Potencia reactiva en adelanto (kVarh) |
| `co2` | float | ≥0 | Emisiones de CO2 (tCO2) |
| `lagging_power_factor` | float | 0-1 | Factor de potencia en atraso |
| `leading_power_factor` | float | 0-1 | Factor de potencia en adelanto |
| `nsm` | int | 0-86400 | Segundos desde medianoche |
| `day_of_week` | int | 0-6 | Día de la semana (0=Lunes, 6=Domingo) |
| `load_type` | str | Light\|Medium\|Maximum | Tipo de carga industrial |

## 🚢 Deployment

### Docker

```bash
# Build
docker build -f Dockerfile.api -t energy-api:latest .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  energy-api:latest
```

### Cloud Run (GCP)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/energy-api

# Deploy
gcloud run deploy energy-api \
  --image gcr.io/PROJECT_ID/energy-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📚 Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- **US-011**: Temporal features engineering
- **US-012**: Preprocessing pipeline
- **US-015**: LightGBM Stacking Ensemble
- **US-019**: Dagster pipeline integration

---

**Versión**: 1.0.0
**Última actualización**: 05 de Noviembre, 2025
**Mantenido por**: MLOps Team - Proyecto Atreides
