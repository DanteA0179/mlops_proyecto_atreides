# Energy Optimization API - Guía de Inicio Rápido

Esta guía te ayudará a hacer tu primera predicción en menos de 5 minutos.

---

## 📋 Requisitos Previos

- **curl** o **Postman** (para testing)
- **Python 3.8+** (opcional, para ejemplos en Python)
- API en ejecución (local o producción)

---

## 🚀 Paso 1: Verificar que la API está activa

### Con curl

```bash
curl http://localhost:8000/health
```

### Con Python

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

**Respuesta esperada**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "service": "energy-optimization-api"
}
```

✅ Si ves `"status": "healthy"`, continúa al siguiente paso.

---

## 🎯 Paso 2: Tu Primera Predicción

### Predicción Individual

#### Con curl

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

#### Con Python

```python
import requests

API_URL = "http://localhost:8000"

# Datos de entrada
prediction_data = {
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "nsm": 36000,  # 10:00 AM
    "day_of_week": 1,  # Martes
    "load_type": "Medium"
}

# Hacer predicción
response = requests.post(f"{API_URL}/predict", json=prediction_data)
result = response.json()

print(f"✅ Consumo predicho: {result['predicted_usage_kwh']:.2f} kWh")
print(f"📊 Intervalo de confianza: [{result['confidence_interval_lower']:.2f}, {result['confidence_interval_upper']:.2f}]")
print(f"🔧 Modelo: {result['model_version']}")
```

**Respuesta esperada**:
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

---

## 📦 Paso 3: Predicción Batch

### Con Python

```python
import requests

API_URL = "http://localhost:8000"

# Lote de predicciones
batch_data = {
    "predictions": [
        {
            "lagging_reactive_power": 15.20,
            "leading_reactive_power": 8.50,
            "co2": 0.03,
            "lagging_power_factor": 0.88,
            "leading_power_factor": 0.95,
            "nsm": 7200,  # 2:00 AM
            "day_of_week": 1,
            "load_type": "Light"
        },
        {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,  # 10:00 AM
            "day_of_week": 1,
            "load_type": "Medium"
        },
        {
            "lagging_reactive_power": 45.80,
            "leading_reactive_power": 25.60,
            "co2": 0.12,
            "lagging_power_factor": 0.75,
            "leading_power_factor": 0.85,
            "nsm": 50400,  # 2:00 PM
            "day_of_week": 1,
            "load_type": "Maximum"
        }
    ]
}

# Hacer predicción batch
response = requests.post(f"{API_URL}/predict/batch", json=batch_data)
result = response.json()

# Mostrar resumen
summary = result["summary"]
print(f"✅ Total predicciones: {summary['total_predictions']}")
print(f"📊 Promedio: {summary['avg_predicted_usage']:.2f} kWh")
print(f"📈 Rango: [{summary['min_predicted_usage']:.2f}, {summary['max_predicted_usage']:.2f}] kWh")
print(f"⏱️ Tiempo de procesamiento: {summary['processing_time_ms']:.2f} ms")

# Mostrar predicciones individuales
for i, pred in enumerate(result["predictions"], 1):
    print(f"  {i}. {pred['predicted_usage_kwh']:.2f} kWh (ID: {pred['prediction_id']})")
```

---

## 📊 Paso 4: Explorar Información del Modelo

### Con curl

```bash
curl http://localhost:8000/model/info
```

### Con Python

```python
import requests

response = requests.get("http://localhost:8000/model/info")
info = response.json()

print(f"🤖 Modelo: {info['model_name']}")
print(f"📦 Versión: {info['model_version']}")
print(f"📊 RMSE: {info['training_metrics']['rmse']:.4f}")
print(f"📈 R²: {info['training_metrics']['r2']:.4f}")
print(f"🔧 Features: {len(info['features'])}")

if info.get('base_models'):
    print("\n🏗️ Modelos Base:")
    for model in info['base_models']:
        print(f"  - {model['name']}: {model['contribution_pct']:.1f}%")
```

---

## 📈 Paso 5: Monitorear Métricas

### Con Python

```python
import requests
import time

API_URL = "http://localhost:8000"

def monitor_metrics(interval=60):
    """Monitor API metrics every 'interval' seconds."""
    while True:
        response = requests.get(f"{API_URL}/model/metrics")
        metrics = response.json()
        
        prod = metrics['production_metrics']
        print(f"\n⏰ {metrics['timestamp']}")
        print(f"📊 Total predicciones: {prod['total_predictions']}")
        print(f"⚡ Latencia P95: {prod['p95_prediction_time_ms']:.2f} ms")
        print(f"❌ Error rate: {prod['error_rate_percent']:.2f}%")
        
        # Alertas
        if prod['p95_prediction_time_ms'] > 200:
            print("⚠️ ALERTA: Latencia alta!")
        if prod['error_rate_percent'] > 1.0:
            print("⚠️ ALERTA: Error rate alto!")
        
        time.sleep(interval)

# Ejecutar monitoreo
monitor_metrics(interval=60)  # Check cada minuto
```

---

## 🧪 Paso 6: Explorar Documentación Interactiva

### Swagger UI

Abre en tu navegador:
```
http://localhost:8000/docs
```

**Funcionalidades**:
- ✅ Ver todos los endpoints
- ✅ Probar requests con "Try it out"
- ✅ Ver ejemplos de requests/responses
- ✅ Ver esquemas de validación

### ReDoc

Abre en tu navegador:
```
http://localhost:8000/redoc
```

**Funcionalidades**:
- ✅ Documentación técnica completa
- ✅ Navegación por sidebar
- ✅ Búsqueda de endpoints
- ✅ Exportar a PDF

---

## 💡 Ejemplos de Casos de Uso

### Caso 1: Predicción para Turno Completo

Predecir consumo para un turno de 8 horas (cada 15 minutos):

```python
import requests
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"

# Configuración base
base_config = {
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "day_of_week": 1,
    "load_type": "Medium"
}

# Generar predicciones para 8 horas (cada 15 min = 32 predicciones)
predictions = []
start_time = 8 * 3600  # 8:00 AM
interval = 15 * 60  # 15 minutos

for i in range(32):
    nsm = start_time + (i * interval)
    pred = base_config.copy()
    pred["nsm"] = nsm
    predictions.append(pred)

# Hacer predicción batch
response = requests.post(
    f"{API_URL}/predict/batch",
    json={"predictions": predictions}
)
result = response.json()

# Analizar resultados
print(f"✅ Consumo total estimado: {sum(p['predicted_usage_kwh'] for p in result['predictions']):.2f} kWh")
print(f"📊 Promedio por intervalo: {result['summary']['avg_predicted_usage']:.2f} kWh")
```

### Caso 2: Análisis What-If

Comparar consumo entre diferentes tipos de carga:

```python
import requests

API_URL = "http://localhost:8000"

base_config = {
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "nsm": 36000,
    "day_of_week": 1
}

# Comparar cargas
load_types = ["Light", "Medium", "Maximum"]
results = {}

for load_type in load_types:
    config = base_config.copy()
    config["load_type"] = load_type
    
    response = requests.post(f"{API_URL}/predict", json=config)
    result = response.json()
    results[load_type] = result["predicted_usage_kwh"]

# Mostrar comparación
print("📊 Comparación de Consumo por Tipo de Carga:")
for load_type, usage in results.items():
    print(f"  {load_type:10s}: {usage:6.2f} kWh")

# Calcular ahorro potencial
saving = results["Maximum"] - results["Light"]
print(f"\n💰 Ahorro potencial (Maximum → Light): {saving:.2f} kWh ({saving/results['Maximum']*100:.1f}%)")
```

### Caso 3: Detección de Anomalías

Detectar desviaciones del consumo esperado:

```python
import requests

API_URL = "http://localhost:8000"

def detect_anomaly(actual_usage, predicted_usage, threshold=0.15):
    """Detect if actual usage is anomalous."""
    deviation = abs(actual_usage - predicted_usage) / predicted_usage
    return deviation > threshold

# Datos reales
actual_usage = 55.30  # kWh real

# Predicción
prediction_data = {
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "nsm": 36000,
    "day_of_week": 1,
    "load_type": "Medium"
}

response = requests.post(f"{API_URL}/predict", json=prediction_data)
result = response.json()
predicted_usage = result["predicted_usage_kwh"]

# Detectar anomalía
is_anomaly = detect_anomaly(actual_usage, predicted_usage)

if is_anomaly:
    deviation_pct = abs(actual_usage - predicted_usage) / predicted_usage * 100
    print(f"⚠️ ANOMALÍA DETECTADA!")
    print(f"   Real: {actual_usage:.2f} kWh")
    print(f"   Esperado: {predicted_usage:.2f} kWh")
    print(f"   Desviación: {deviation_pct:.1f}%")
else:
    print(f"✅ Consumo normal: {actual_usage:.2f} kWh")
```

---

## 🛠️ Troubleshooting

### Error: Connection refused

```bash
# Verificar que el servidor está corriendo
curl http://localhost:8000/health
```

**Solución**:
```bash
# Iniciar el servidor
python src/api/main.py
# o
uvicorn src.api.main:app --reload
```

### Error: 422 Validation Error

**Causa**: Datos de entrada inválidos.

**Solución**: Verificar que todos los campos cumplen con las validaciones:
- Factores de potencia: 0-1
- NSM: 0-86400
- load_type: "Light", "Medium", "Maximum"

### Error: 503 Model Not Loaded

**Causa**: El modelo no está cargado.

**Solución**: Verificar que el modelo existe en `models/` y reiniciar el servidor.

---

## 📚 Próximos Pasos

1. **Explorar Documentación Completa**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
2. **Ver Más Ejemplos**: [EXAMPLES.md](./EXAMPLES.md)
3. **Troubleshooting Avanzado**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
4. **Integrar con tu aplicación**: Usar los ejemplos de Python/curl

---

## 📞 ¿Necesitas Ayuda?

- **Documentación**: `/docs` o `/redoc`
- **GitHub**: [Issues](https://github.com/DanteA0179/mlops_proyecto_atreides/issues)
- **Email**: mlops@atreides.com

---

**¡Felicidades! Ya puedes usar la Energy Optimization API** 🎉
