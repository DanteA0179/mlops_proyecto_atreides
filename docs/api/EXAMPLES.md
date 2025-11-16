# Energy Optimization API - Ejemplos de Código

Esta guía contiene ejemplos prácticos en diferentes lenguajes para integrar la API.

---

## 📋 Tabla de Contenidos

1. [Python](#python)
2. [JavaScript](#javascript)
3. [curl](#curl)
4. [Postman](#postman)
5. [Casos de Uso Avanzados](#casos-de-uso-avanzados)

---

## 🐍 Python

### Instalación de Dependencias

```bash
pip install requests
```

### Ejemplo Básico - Predicción Individual

```python
import requests

API_URL = "http://localhost:8000"

def predict_energy(data):
    """Make single energy prediction."""
    response = requests.post(f"{API_URL}/predict", json=data)
    response.raise_for_status()
    return response.json()

# Datos de entrada
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

# Hacer predicción
result = predict_energy(prediction_data)
print(f"Consumo predicho: {result['predicted_usage_kwh']:.2f} kWh")
```

### Ejemplo - Predicción Batch

```python
import requests

API_URL = "http://localhost:8000"

def predict_batch(predictions):
    """Make batch energy predictions."""
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"predictions": predictions}
    )
    response.raise_for_status()
    return response.json()

# Lote de predicciones
batch = [
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
    {
        "lagging_reactive_power": 45.80,
        "leading_reactive_power": 25.60,
        "co2": 0.12,
        "lagging_power_factor": 0.75,
        "leading_power_factor": 0.85,
        "nsm": 50400,
        "day_of_week": 1,
        "load_type": "Maximum"
    }
]

result = predict_batch(batch)
print(f"Total predicciones: {result['summary']['total_predictions']}")
print(f"Promedio: {result['summary']['avg_predicted_usage']:.2f} kWh")
```

### Ejemplo - Cliente con Manejo de Errores

```python
import requests
from requests.exceptions import RequestException
from typing import Dict, Optional
import time

class EnergyOptimizationClient:
    """Client for Energy Optimization API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 5):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def predict(self, data: Dict, retries: int = 3) -> Optional[Dict]:
        """
        Make single prediction with retry logic.
        
        Parameters
        ----------
        data : Dict
            Prediction request data
        retries : int
            Number of retries on failure
            
        Returns
        -------
        Optional[Dict]
            Prediction result or None on failure
        """
        for attempt in range(retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt + 1}: Timeout")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error {e.response.status_code}: {e.response.json()}")
                break  # Don't retry on HTTP errors
            
            except RequestException as e:
                print(f"Request failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            data = response.json()
            return data.get("status") == "healthy"
        except:
            return False

# Usage
client = EnergyOptimizationClient()

if client.health_check():
    result = client.predict({
        "lagging_reactive_power": 23.45,
        "leading_reactive_power": 12.30,
        "co2": 0.05,
        "lagging_power_factor": 0.85,
        "leading_power_factor": 0.92,
        "nsm": 36000,
        "day_of_week": 1,
        "load_type": "Medium"
    })
    
    if result:
        print(f"✅ Predicción exitosa: {result['predicted_usage_kwh']:.2f} kWh")
    else:
        print("❌ Predicción falló")
else:
    print("⚠️ API no está saludable")
```

---

## 🌐 JavaScript

### Instalación (Node.js)

```bash
npm install axios
```

### Ejemplo Básico - Fetch API

```javascript
const API_URL = 'http://localhost:8000';

async function predictEnergy(data) {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
}

// Uso
const predictionData = {
  lagging_reactive_power: 23.45,
  leading_reactive_power: 12.30,
  co2: 0.05,
  lagging_power_factor: 0.85,
  leading_power_factor: 0.92,
  nsm: 36000,
  day_of_week: 1,
  load_type: 'Medium'
};

predictEnergy(predictionData)
  .then(result => {
    console.log(`Consumo predicho: ${result.predicted_usage_kwh.toFixed(2)} kWh`);
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

### Ejemplo - Con Axios

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

class EnergyOptimizationClient {
  constructor(baseURL = API_URL) {
    this.client = axios.create({
      baseURL: baseURL,
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async predict(data) {
    try {
      const response = await this.client.post('/predict', data);
      return response.data;
    } catch (error) {
      if (error.response) {
        console.error(`HTTP ${error.response.status}:`, error.response.data);
      } else if (error.request) {
        console.error('No response received:', error.request);
      } else {
        console.error('Error:', error.message);
      }
      throw error;
    }
  }

  async predictBatch(predictions) {
    try {
      const response = await this.client.post('/predict/batch', { predictions });
      return response.data;
    } catch (error) {
      console.error('Batch prediction failed:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data.status === 'healthy';
    } catch {
      return false;
    }
  }
}

// Uso
const client = new EnergyOptimizationClient();

async function main() {
  const isHealthy = await client.healthCheck();
  
  if (isHealthy) {
    const result = await client.predict({
      lagging_reactive_power: 23.45,
      leading_reactive_power: 12.30,
      co2: 0.05,
      lagging_power_factor: 0.85,
      leading_power_factor: 0.92,
      nsm: 36000,
      day_of_week: 1,
      load_type: 'Medium'
    });
    
    console.log(`✅ Consumo predicho: ${result.predicted_usage_kwh.toFixed(2)} kWh`);
  } else {
    console.log('⚠️ API no está saludable');
  }
}

main();
```

---

## 💻 curl

### Predicción Individual

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

### Predicción Batch

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
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
      {
        "lagging_reactive_power": 45.80,
        "leading_reactive_power": 25.60,
        "co2": 0.12,
        "lagging_power_factor": 0.75,
        "leading_power_factor": 0.85,
        "nsm": 50400,
        "day_of_week": 1,
        "load_type": "Maximum"
      }
    ]
  }'
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

### Model Info

```bash
curl -X GET "http://localhost:8000/model/info"
```

### Model Metrics

```bash
curl -X GET "http://localhost:8000/model/metrics"
```

### Con jq (Pretty Print)

```bash
curl -s http://localhost:8000/health | jq '.'
```

---

## 📮 Postman

### Colección Postman

Crear una colección con estas requests:

#### 1. Health Check

```
GET http://localhost:8000/health
```

#### 2. Predict Single

```
POST http://localhost:8000/predict
Content-Type: application/json

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

#### 3. Predict Batch

```
POST http://localhost:8000/predict/batch
Content-Type: application/json

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
    }
  ]
}
```

### Variables de Entorno

```json
{
  "base_url": "http://localhost:8000",
  "api_key": "your-api-key-here"
}
```

---

## 🚀 Casos de Uso Avanzados

### Caso 1: Planificación de Turno Completo

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"

def plan_shift(start_hour, duration_hours, load_type):
    """
    Plan energy consumption for complete shift.
    
    Parameters
    ----------
    start_hour : int
        Shift start hour (0-23)
    duration_hours : int
        Shift duration in hours
    load_type : str
        Load type ("Light", "Medium", "Maximum")
    """
    predictions = []
    base_config = {
        "lagging_reactive_power": 23.45,
        "leading_reactive_power": 12.30,
        "co2": 0.05,
        "lagging_power_factor": 0.85,
        "leading_power_factor": 0.92,
        "day_of_week": datetime.now().weekday(),
        "load_type": load_type
    }
    
    # Generate predictions every 15 minutes
    intervals = (duration_hours * 60) // 15  # 15-minute intervals
    start_nsm = start_hour * 3600
    
    for i in range(intervals):
        config = base_config.copy()
        config["nsm"] = start_nsm + (i * 15 * 60)
        predictions.append(config)
    
    # Make batch prediction
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"predictions": predictions}
    )
    result = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "interval": i + 1,
            "time": f"{(start_hour + (i * 15 // 60)) % 24:02d}:{(i * 15 % 60):02d}",
            "predicted_kwh": p["predicted_usage_kwh"]
        }
        for i, p in enumerate(result["predictions"])
    ])
    
    return df, result["summary"]

# Usage
df, summary = plan_shift(start_hour=8, duration_hours=8, load_type="Medium")

print("📊 Planificación de Turno")
print(df)
print(f"\n📈 Consumo total estimado: {summary['total_predictions'] * summary['avg_predicted_usage']:.2f} kWh")
```

### Caso 2: Optimización de Horarios

```python
import requests
import numpy as np

API_URL = "http://localhost:8000"

def find_optimal_hours(load_type, num_hours=8):
    """
    Find optimal hours with lowest predicted consumption.
    
    Parameters
    ----------
    load_type : str
        Load type to analyze
    num_hours : int
        Number of consecutive hours needed
        
    Returns
    -------
    dict
        Optimal hours and estimated consumption
    """
    predictions = []
    base_config = {
        "lagging_reactive_power": 23.45,
        "leading_reactive_power": 12.30,
        "co2": 0.05,
        "lagging_power_factor": 0.85,
        "leading_power_factor": 0.92,
        "day_of_week": 1,
        "load_type": load_type
    }
    
    # Test all 24 hours
    for hour in range(24):
        config = base_config.copy()
        config["nsm"] = hour * 3600
        predictions.append(config)
    
    # Make batch prediction
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"predictions": predictions}
    )
    result = response.json()
    
    # Extract predictions
    hourly_usage = [p["predicted_usage_kwh"] for p in result["predictions"]]
    
    # Find optimal window
    min_sum = float('inf')
    optimal_start = 0
    
    for start_hour in range(24 - num_hours + 1):
        window_sum = sum(hourly_usage[start_hour:start_hour + num_hours])
        if window_sum < min_sum:
            min_sum = window_sum
            optimal_start = start_hour
    
    return {
        "optimal_start_hour": optimal_start,
        "optimal_end_hour": (optimal_start + num_hours) % 24,
        "estimated_consumption": min_sum / num_hours,  # Average per hour
        "total_consumption": min_sum
    }

# Usage
result = find_optimal_hours("Medium", num_hours=8)

print("🎯 Horario Óptimo Encontrado:")
print(f"   Inicio: {result['optimal_start_hour']:02d}:00")
print(f"   Fin: {result['optimal_end_hour']:02d}:00")
print(f"   Consumo promedio: {result['estimated_consumption']:.2f} kWh/hora")
print(f"   Consumo total: {result['total_consumption']:.2f} kWh")
```

### Caso 3: Monitoreo en Tiempo Real

```python
import requests
import time
from datetime import datetime

API_URL = "http://localhost:8000"

class RealTimeMonitor:
    """Real-time API monitoring."""
    
    def __init__(self, alert_thresholds=None):
        self.thresholds = alert_thresholds or {
            "p95_latency_ms": 200,
            "error_rate_pct": 1.0,
            "memory_mb": 1000
        }
    
    def monitor(self, interval=60):
        """
        Monitor API metrics continuously.
        
        Parameters
        ----------
        interval : int
            Check interval in seconds
        """
        print("🔍 Iniciando monitoreo en tiempo real...")
        
        while True:
            try:
                # Get metrics
                response = requests.get(f"{API_URL}/model/metrics")
                metrics = response.json()
                
                # Display status
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n⏰ {timestamp}")
                print(f"📊 Predicciones totales: {metrics['production_metrics']['total_predictions']}")
                print(f"⚡ Latencia P95: {metrics['production_metrics']['p95_prediction_time_ms']:.2f} ms")
                print(f"💾 Memoria: {metrics['system_health']['memory_usage_mb']:.2f} MB")
                print(f"🖥️ CPU: {metrics['system_health']['cpu_usage_percent']:.1f}%")
                
                # Check alerts
                self._check_alerts(metrics)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n⏹️ Monitoreo detenido")
                break
            except Exception as e:
                print(f"⚠️ Error en monitoreo: {e}")
                time.sleep(interval)
    
    def _check_alerts(self, metrics):
        """Check if any threshold is exceeded."""
        prod = metrics['production_metrics']
        health = metrics['system_health']
        
        if prod['p95_prediction_time_ms'] > self.thresholds['p95_latency_ms']:
            print(f"🚨 ALERTA: Latencia alta ({prod['p95_prediction_time_ms']:.2f} ms)")
        
        if prod['error_rate_percent'] > self.thresholds['error_rate_pct']:
            print(f"🚨 ALERTA: Error rate alto ({prod['error_rate_percent']:.2f}%)")
        
        if health['memory_usage_mb'] > self.thresholds['memory_mb']:
            print(f"🚨 ALERTA: Memoria alta ({health['memory_usage_mb']:.2f} MB)")

# Usage
monitor = RealTimeMonitor()
monitor.monitor(interval=60)  # Check every minute
```

---

## 📚 Recursos Adicionales

- **Documentación Completa**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- **Guía de Inicio Rápido**: [QUICK_START.md](./QUICK_START.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

**Última actualización**: 16 de Noviembre, 2025
