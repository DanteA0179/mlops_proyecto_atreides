# Energy Optimization API - Troubleshooting

Guía completa para solucionar problemas comunes con la API.

---

## 📋 Tabla de Contenidos

1. [Errores de Conexión](#errores-de-conexión)
2. [Errores de Validación (422)](#errores-de-validación-422)
3. [Errores del Servidor (500)](#errores-del-servidor-500)
4. [Problemas de Performance](#problemas-de-performance)
5. [Problemas con el Modelo](#problemas-con-el-modelo)
6. [Debugging](#debugging)
7. [FAQ](#faq)

---

## 🔌 Errores de Conexión

### Connection Refused

**Síntoma**:
```
requests.exceptions.ConnectionError: Connection refused
```

**Causas Posibles**:
1. Servidor no está corriendo
2. Puerto incorrecto
3. Firewall bloqueando conexión

**Soluciones**:

#### Verificar que el servidor está corriendo
```bash
# Check proceso
ps aux | grep uvicorn

# Check puerto
netstat -tuln | grep 8000
```

#### Iniciar el servidor
```bash
# Opción 1: Directo con Python
python src/api/main.py

# Opción 2: Con uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Opción 3: Con Docker
docker-compose up
```

#### Verificar puerto correcto
```python
# Asegurarse de usar el puerto correcto
API_URL = "http://localhost:8000"  # ✅
# No usar:
API_URL = "http://localhost:5000"  # ❌
```

---

### Timeout

**Síntoma**:
```
requests.exceptions.Timeout: Request timed out
```

**Causas Posibles**:
1. Servidor sobrecargado
2. Modelo muy grande
3. Batch demasiado grande
4. Red lenta

**Soluciones**:

#### Aumentar timeout
```python
import requests

response = requests.post(
    f"{API_URL}/predict",
    json=data,
    timeout=30  # Aumentar de 5 a 30 segundos
)
```

#### Reducir tamaño del batch
```python
# ❌ MALO: Batch muy grande
batch = {"predictions": [pred] * 1000}  # 1000 predicciones

# ✅ BUENO: Batch más pequeño
batch = {"predictions": [pred] * 100}  # 100 predicciones
```

#### Verificar salud del servidor
```bash
curl http://localhost:8000/health
```

---

## ⚠️ Errores de Validación (422)

### Invalid load_type

**Error**:
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

**Solución**:
```python
# ❌ INCORRECTO
data = {"load_type": "medium"}  # Minúsculas
data = {"load_type": "HIGH"}    # Valor inválido

# ✅ CORRECTO
data = {"load_type": "Medium"}  # Capitalizado correcto
data = {"load_type": "Light"}   # Otro valor válido
data = {"load_type": "Maximum"} # Otro valor válido
```

---

### Invalid power_factor Range

**Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "lagging_power_factor"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error"
    }
  ]
}
```

**Solución**:
```python
# ❌ INCORRECTO
data = {
    "lagging_power_factor": 1.5,  # > 1.0
    "leading_power_factor": -0.5  # < 0.0
}

# ✅ CORRECTO
data = {
    "lagging_power_factor": 0.85,  # 0.0 <= x <= 1.0
    "leading_power_factor": 0.92   # 0.0 <= x <= 1.0
}
```

---

### Invalid nsm (Seconds from Midnight)

**Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "nsm"],
      "msg": "ensure this value is less than or equal to 86400",
      "type": "value_error"
    }
  ]
}
```

**Solución**:
```python
# ❌ INCORRECTO
data = {"nsm": 90000}  # > 86400 (más de 24 horas)

# ✅ CORRECTO
# Convertir hora a segundos
hour = 10  # 10:00 AM
minute = 30
nsm = (hour * 3600) + (minute * 60)  # 37800 segundos

data = {"nsm": nsm}  # 0 <= nsm <= 86400
```

**Helper Function**:
```python
def time_to_nsm(hour, minute=0, second=0):
    """Convert time to NSM (Number of Seconds from Midnight)."""
    return (hour * 3600) + (minute * 60) + second

# Usage
data = {"nsm": time_to_nsm(10, 30)}  # 10:30 AM
```

---

### Invalid day_of_week

**Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "day_of_week"],
      "msg": "ensure this value is less than or equal to 6",
      "type": "value_error"
    }
  ]
}
```

**Solución**:
```python
# ❌ INCORRECTO
data = {"day_of_week": 7}  # Solo 0-6

# ✅ CORRECTO
# 0 = Lunes, 1 = Martes, ..., 6 = Domingo
data = {"day_of_week": 0}  # Lunes
data = {"day_of_week": 5}  # Sábado
```

**Helper Function**:
```python
from datetime import datetime

def get_day_of_week():
    """Get current day of week (0=Monday)."""
    return datetime.now().weekday()

# Usage
data = {"day_of_week": get_day_of_week()}
```

---

### Negative Values

**Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "co2"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error"
    }
  ]
}
```

**Solución**:
```python
# ❌ INCORRECTO
data = {
    "lagging_reactive_power": -5.0,  # Negativo
    "co2": -0.01                     # Negativo
}

# ✅ CORRECTO
data = {
    "lagging_reactive_power": 23.45,  # >= 0
    "co2": 0.05                       # >= 0
}
```

---

### Batch Empty or Too Large

**Error**:
```json
{
  "detail": "Batch cannot be empty"
}
```

o

```json
{
  "detail": "Batch cannot exceed 1000 predictions"
}
```

**Solución**:
```python
# ❌ INCORRECTO
batch = {"predictions": []}  # Vacío
batch = {"predictions": [pred] * 2000}  # Muy grande

# ✅ CORRECTO
batch = {"predictions": [pred1, pred2]}  # 1-1000 items
```

---

## 🔥 Errores del Servidor (500)

### Internal Server Error

**Síntoma**:
```json
{
  "detail": "Prediction failed: Model not found"
}
```

**Causas Posibles**:
1. Modelo no cargado
2. Archivo del modelo corrupto
3. Dependencias faltantes
4. Error en feature engineering

**Soluciones**:

#### 1. Verificar Health Check
```bash
curl http://localhost:8000/health
```

Buscar:
```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

#### 2. Verificar logs del servidor
```bash
# Ver logs en tiempo real
tail -f logs/api.log

# Buscar errores
grep "ERROR" logs/api.log
```

#### 3. Verificar que el modelo existe
```bash
# Listar modelos disponibles
ls -la models/

# Verificar tamaño del archivo
du -h models/stacking_ensemble.pkl
```

#### 4. Reinstalar dependencias
```bash
poetry install
# o
pip install -r requirements.txt
```

#### 5. Reiniciar el servidor
```bash
# Matar proceso
pkill -f uvicorn

# Reiniciar
python src/api/main.py
```

---

### Model Loading Failed

**Error en logs**:
```
ERROR: Failed to load model: No such file or directory
```

**Solución**:

#### Descargar modelo con DVC
```bash
dvc pull models/stacking_ensemble.pkl.dvc
```

#### Verificar configuración
```python
# src/api/utils/config.py
MODEL_PATH = Path("models")
MODEL_TYPE = "stacking_ensemble"
```

#### Re-entrenar modelo
```bash
python src/models/train_stacking_ensemble.py
```

---

## 🐌 Problemas de Performance

### Latencia Alta

**Síntoma**: Respuestas > 200ms

**Diagnóstico**:
```bash
curl http://localhost:8000/model/metrics
```

Revisar `p95_prediction_time_ms`.

**Causas y Soluciones**:

#### 1. Batch muy grande
```python
# ❌ MALO
batch = {"predictions": [pred] * 1000}  # Muy grande

# ✅ BUENO
batch = {"predictions": [pred] * 100}  # Más manejable
```

#### 2. Servidor sobrecargado
```bash
# Ver uso de recursos
curl http://localhost:8000/health | jq '.memory_usage_mb, .cpu_usage_percent'
```

**Solución**: Escalar horizontalmente o agregar más recursos.

#### 3. Modelo no optimizado
- Usar modelo más rápido (LightGBM en lugar de ensemble)
- Reducir número de features
- Cuantizar modelo

---

### Memory Usage Alto

**Síntoma**: `memory_usage_mb > 1000`

**Soluciones**:

#### 1. Reiniciar servidor periódicamente
```bash
# Cron job para reiniciar diariamente
0 3 * * * systemctl restart energy-api
```

#### 2. Limitar tamaño de batch
```python
MAX_BATCH_SIZE = 500  # En lugar de 1000
```

#### 3. Optimizar modelo
- Usar formato ONNX
- Comprimir modelo
- Usar modelo más pequeño

---

## 🤖 Problemas con el Modelo

### Model Not Loaded

**Síntoma**:
```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

**Soluciones**:

1. **Verificar modelo existe**:
```bash
ls -la models/stacking_ensemble.pkl
```

2. **Descargar con DVC**:
```bash
dvc pull
```

3. **Verificar permisos**:
```bash
chmod 644 models/stacking_ensemble.pkl
```

4. **Reiniciar servidor**:
```bash
python src/api/main.py
```

---

### Predictions Not Matching

**Síntoma**: Predicciones inconsistentes con valores esperados

**Diagnóstico**:

#### 1. Verificar versión del modelo
```bash
curl http://localhost:8000/model/info | jq '.model_version'
```

#### 2. Verificar métricas de entrenamiento
```bash
curl http://localhost:8000/model/info | jq '.training_metrics'
```

#### 3. Verificar features
```python
# Asegurarse de usar todas las features requeridas
response = requests.get(f"{API_URL}/model/info")
required_features = [f["name"] for f in response.json()["features"]]
print(required_features)
```

---

## 🐛 Debugging

### Modo Debug

#### Habilitar logs detallados
```bash
# Modificar config
export LOG_LEVEL=DEBUG

# Reiniciar servidor
python src/api/main.py
```

#### Ver logs en tiempo real
```bash
tail -f logs/api.log
```

### Request/Response Logging

```python
import requests
import logging

# Habilitar logging de requests
logging.basicConfig(level=logging.DEBUG)

response = requests.post(f"{API_URL}/predict", json=data)
```

### Validación Manual

```python
from src.api.models.requests import PredictionRequest
from pydantic import ValidationError

# Validar datos antes de enviar
try:
    request = PredictionRequest(**data)
    print("✅ Datos válidos")
except ValidationError as e:
    print("❌ Errores de validación:")
    print(e.json())
```

---

## ❓ FAQ

### ¿Por qué mi predicción es muy diferente de lo esperado?

1. Verificar que los datos de entrada son correctos
2. Verificar unidades (kVarh, tCO2, etc.)
3. Verificar que `load_type` es apropiado
4. Comparar con intervalos de confianza

### ¿Cómo sé qué versión del modelo está activa?

```bash
curl http://localhost:8000/model/info | jq '.model_version'
```

### ¿Puedo usar la API sin internet?

Sí, la API funciona localmente sin conexión a internet una vez que:
- El código está clonado
- Las dependencias están instaladas
- El modelo está descargado con DVC

### ¿Qué hacer si el servidor crashea constantemente?

1. Verificar logs: `tail -f logs/api.log`
2. Verificar memoria disponible
3. Reducir `MAX_BATCH_SIZE`
4. Verificar que el modelo no está corrupto
5. Reinstalar dependencias

### ¿Cómo actualizar a una nueva versión del modelo?

```bash
# Pull nueva versión con DVC
dvc pull models/new_model.pkl.dvc

# Actualizar configuración
export MODEL_TYPE=new_model

# Reiniciar servidor
python src/api/main.py
```

---

## 📞 Soporte

Si ninguna de estas soluciones funciona:

1. **Verificar GitHub Issues**: [Issues](https://github.com/DanteA0179/mlops_proyecto_atreides/issues)
2. **Crear nuevo issue** con:
   - Descripción del problema
   - Logs relevantes
   - Pasos para reproducir
   - Versión de la API
3. **Email**: mlops@atreides.com

---

## 📚 Recursos Adicionales

- [Documentación Completa](./API_DOCUMENTATION.md)
- [Guía de Inicio Rápido](./QUICK_START.md)
- [Ejemplos de Código](./EXAMPLES.md)
- [Swagger UI](http://localhost:8000/docs)

---

**Última actualización**: 16 de Noviembre, 2025
