# Sistema de Monitoreo MVP

Sistema de monitoreo básico para detectar data drift en el modelo Stacking Ensemble en producción.

## 📋 Descripción

Este módulo proporciona funcionalidad mínima para monitorear el modelo en producción:

- **Logging de predicciones**: Captura automática de features y predicciones en CSV
- **Datos de referencia**: Dataset de 1,000 muestras del training set
- **Reporte de drift**: Análisis de drift usando Evidently AI

## 🚀 Inicio Rápido

### 1. Preparar Datos de Referencia

Antes de generar reportes, crea el dataset de referencia:

```bash
poetry run python scripts/prepare_reference_data.py
```

Esto crea `data/monitoring/reference_data.csv` con 1,000 muestras aleatorias del training set.

### 2. Generar Predicciones

Las predicciones se loggean automáticamente cuando usas la API:

```bash
# Iniciar la API
poetry run uvicorn src.api.main:app --reload

# Hacer predicciones (en otra terminal)
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

Cada predicción se guarda automáticamente en `data/monitoring/predictions.csv`.

### 3. Generar Reporte de Drift

Una vez que tengas al menos 20 predicciones:

```bash
poetry run python scripts/generate_drift_report.py
```

Esto genera `reports/monitoring/drift_report.html`.

### 4. Ver Reporte

Abre el reporte HTML en tu navegador:

```bash
open reports/monitoring/drift_report.html
```

## 📁 Estructura de Archivos

```
src/monitoring/
├── __init__.py              # Exports del módulo
├── log_prediction.py        # Función de logging
└── README.md                # Esta documentación

scripts/
├── prepare_reference_data.py   # Genera datos de referencia
└── generate_drift_report.py    # Genera reporte Evidently

data/monitoring/
├── reference_data.csv       # 1,000 muestras de training (baseline)
└── predictions.csv          # Predicciones de producción (append)

reports/monitoring/
└── drift_report.html        # Reporte HTML de Evidently
```

## 🔧 Uso Programático

### Logging Manual de Predicciones

```python
from src.monitoring import log_prediction

# Features procesadas (18 features después de feature engineering)
features = {
    "Lagging_Current_Reactive.Power_kVarh": 23.45,
    "Leading_Current_Reactive_Power_kVarh": 12.30,
    # ... otras 16 features
}

# Prediction del modelo
prediction = 42.5

# Log (append a CSV)
log_prediction(features, prediction)
```

### Integración en API

El logging ya está integrado en `src/api/routes/predict.py`:

```python
# En el endpoint /predict
try:
    feature_names = feature_service.get_feature_names()
    features_dict = dict(zip(feature_names, features[0].tolist()))
    log_prediction(features_dict, float(prediction[0]))
except Exception as e:
    logger.warning(f"Failed to log prediction for monitoring: {e}")
```

## 📊 Interpretación del Reporte

El reporte de drift muestra:

1. **Drift Score**: Métrica general de drift (0-1)
2. **Feature Drift**: Drift por feature individual
3. **Distribuciones**: Comparación de distribuciones reference vs production
4. **Statistics**: Tests estadísticos (Kolmogorov-Smirnov, etc.)

### Umbrales Recomendados

- **Drift Score < 0.3**: Sin drift significativo
- **Drift Score 0.3-0.5**: Drift moderado - monitorear
- **Drift Score > 0.5**: Drift alto - considerar reentrenamiento

## ⚠️ Limitaciones del MVP

Este es un MVP ultra-simple. NO incluye:

- ❌ Alertas automáticas
- ❌ Dashboard en tiempo real
- ❌ Rotación automática de archivos
- ❌ Performance metrics (requiere ground truth)
- ❌ Buffering en memoria
- ❌ Formato Parquet (solo CSV)

## 🔄 Mantenimiento

### Limpiar Logs Antiguos

```bash
# Backup de logs
cp data/monitoring/predictions.csv data/monitoring/predictions_backup_$(date +%Y%m%d).csv

# Limpiar logs
rm data/monitoring/predictions.csv
```

### Actualizar Datos de Referencia

Si reentrenaste el modelo, regenera el dataset de referencia:

```bash
poetry run python scripts/prepare_reference_data.py
```

## 🧪 Testing

Ejecutar tests unitarios:

```bash
# Todos los tests
poetry run pytest tests/unit/test_monitoring.py -v

# Con coverage
poetry run pytest tests/unit/test_monitoring.py --cov=src.monitoring --cov-report=term-missing
```

## 📚 Referencias

- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [DataDriftPreset Reference](https://docs.evidentlyai.com/reference/all-metrics)
- US-020: FastAPI Endpoints
- US-012: Preprocessing Pipeline

## 🆘 Troubleshooting

### Error: "Reference data not found"

```bash
# Ejecutar script de preparación
poetry run python scripts/prepare_reference_data.py
```

### Error: "Production predictions not found"

Haz al menos 20 predicciones a través de la API antes de generar el reporte.

### Warning: "Production data has only X rows"

El reporte funcionará, pero se recomienda al menos 50 predicciones para análisis confiable.

### Error: "Column mismatch"

Asegúrate de que el modelo y el pipeline de preprocessing no hayan cambiado desde que se generó el dataset de referencia.

## 📝 Ejemplo Completo

```bash
# 1. Preparar referencia
poetry run python scripts/prepare_reference_data.py

# 2. Iniciar API y hacer 50 predicciones
poetry run uvicorn src.api.main:app --reload

# 3. Generar reporte
poetry run python scripts/generate_drift_report.py

# 4. Ver reporte
open reports/monitoring/drift_report.html
```

---

**Versión**: MVP 1.0
**Estado**: ✅ Funcional
**Tiempo de implementación**: 8 horas
**Coverage**: >70%
