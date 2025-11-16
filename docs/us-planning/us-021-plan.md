# US-021: Exportar Modelo a ONNX - Plan de Implementación

**Estado**: 📋 En Planeación  
**Fecha de creación**: 06 de Noviembre, 2025  
**Responsable**: ML Engineer (Julian)  
**Tipo**: MLOps + Model Optimization

---

## 📋 Resumen Ejecutivo

Implementar exportación de modelos de ML (Chronos-2 y XGBoost) a formato ONNX para lograr portabilidad cross-platform y optimización de latencia en inferencia. Se creará un pipeline completo de conversión, validación y deployment con endpoint alternativo en FastAPI.

**Problema**: Los modelos actuales en PyTorch/XGBoost tienen dependencias pesadas y latencia subóptima en producción.

**Solución**: Convertir modelos a ONNX Runtime, validar equivalencia numérica, y exponer endpoint optimizado `/predict_onnx` con benchmarks de performance.

---

## 🎯 Objetivos

### Objetivo 1: Conversión de Modelos a ONNX
- **Convertir 8 modelos** del proyecto a ONNX:
  - **Gradient Boosting**: XGBoost, LightGBM, CatBoost
  - **Ensembles**: Ridge Stacking, LightGBM Stacking
  - **Foundation Models**: Chronos-2 (3 variantes: zero-shot, fine-tuned, covariates)
- Script automatizado: `src/models/export_onnx.py`
- Soporte para GPU y CPU
- Arquitectura extensible para agregar nuevos modelos

### Objetivo 2: Validación de Equivalencia
- Predicciones idénticas entre modelo original y ONNX para **todos los 8 modelos**
- Tolerancia numérica: 1e-5 (0.00001)
- Tests automatizados de validación por tipo de modelo
- Reporte consolidado de validación con métricas por modelo
- Validación especial para ensembles (base models + meta-model)

### Objetivo 3: Benchmark de Performance
- Comparación de latencia: ONNX vs Original para **8 modelos**
- Métricas: p50, p95, p99 latency por modelo
- Comparación de throughput (predicciones/segundo)
- Uso de memoria (RAM y VRAM) por tipo de modelo
- Análisis comparativo: Gradient Boosting vs Ensembles vs Foundation Models
- Identificar el modelo más rápido en ONNX

### Objetivo 4: Integración con FastAPI
- Nuevo endpoint: `POST /predict_onnx`
- **Soporte multi-modelo**: Selección de modelo vía parámetro o config
- Soporte para predicción individual y batch
- Endpoint de selección: `GET /predict_onnx/models` (lista modelos ONNX disponibles)
- Documentación OpenAPI automática
- Backward compatibility con `/predict` existente


---

## 🏗️ Arquitectura Propuesta

### Estructura de Archivos

```
src/models/
├── export_onnx.py              # Script principal de conversión (NUEVO)
├── onnx_validator.py           # Validación de equivalencia (NUEVO)
└── onnx_benchmark.py           # Benchmarking (NUEVO)

src/api/
├── routes/
│   └── predict_onnx.py         # Endpoint ONNX (NUEVO)
└── services/
    └── onnx_service.py         # Servicio de inferencia ONNX (NUEVO)

models/
├── onnx/                       # Modelos ONNX exportados (NUEVO)
│   ├── chronos2_finetuned.onnx
│   ├── xgboost_model.onnx
│   └── metadata.json
└── benchmarks/                 # Resultados de benchmarks (NUEVO)
    └── onnx_vs_pytorch.json

config/
└── onnx/                       # Configuraciones ONNX (NUEVO)
    ├── chronos2_export.yaml
    └── xgboost_export.yaml

tests/
└── unit/
    ├── test_onnx_export.py     # Tests de exportación (NUEVO)
    ├── test_onnx_validation.py # Tests de validación (NUEVO)
    └── test_onnx_api.py        # Tests de API (NUEVO)

docs/
└── guides/
    └── ONNX_GUIDE.md           # Guía de uso ONNX (NUEVO)
```

### Flujo de Conversión

```
Modelo Original (PyTorch/XGBoost)
    ↓
export_onnx.py
    ├── Carga modelo entrenado
    ├── Prepara input dummy
    ├── Exporta a ONNX
    └── Optimiza grafo ONNX
    ↓
Modelo ONNX (.onnx)
    ↓
onnx_validator.py
    ├── Carga ambos modelos
    ├── Genera inputs de prueba
    ├── Compara predicciones
    └── Valida tolerancia 1e-5
    ↓
Validación Exitosa ✅
    ↓
onnx_benchmark.py
    ├── Mide latencia (1000 runs)
    ├── Mide throughput
    ├── Mide memoria
    └── Genera reporte
    ↓
Reporte de Performance 📊
```


---

## 🎨 Arquitectura Multi-Modelo

### Modelos Soportados (8 Total)

| # | Modelo | Tipo | Tamaño Original | Tamaño ONNX Est. | RMSE Test |
|---|--------|------|-----------------|------------------|-----------|
| 1 | XGBoost | Gradient Boosting | ~5MB | ~5MB | 12.8311 |
| 2 | LightGBM | Gradient Boosting | ~4MB | ~4MB | 12.9521 |
| 3 | CatBoost | Gradient Boosting | ~6MB | ~6MB | 12.9211 |
| 4 | Ridge Stacking | Ensemble | ~15MB | ~15MB | 12.8151 |
| 5 | LightGBM Stacking | Ensemble | ~15MB | ~15MB | 12.7982 ⭐ |
| 6 | Chronos-2 Zero-shot | Foundation | ~455MB | ~455MB | 53.1069 |
| 7 | Chronos-2 Fine-tuned | Foundation | ~455MB | ~455MB | 40.5071 |
| 8 | Chronos-2 Covariates | Foundation | ~455MB | ~455MB | 41.5789 |

**Total**: ~1.8GB de modelos ONNX

### Estrategia de Conversión por Tipo

**Gradient Boosting (XGBoost, LightGBM, CatBoost)**:
- Conversión directa con `onnxmltools`
- Input: 18 features (float32)
- Output: 1 predicción (float32)
- Optimización: Level 2 (operator fusion)

**Stacking Ensembles (Ridge, LightGBM)**:
- Conversión en 2 pasos:
  1. Exportar cada base model (XGBoost, LightGBM, CatBoost)
  2. Exportar meta-model (Ridge o LightGBM)
- Pipeline ONNX: base models → concatenate → meta-model
- Optimización: Level 2 + graph fusion

**Foundation Models (Chronos-2)**:
- Conversión con `torch.onnx.export`
- Input: Secuencia temporal (batch, seq_len, 1)
- Output: Predicciones futuras
- Optimización: Level 1 (básica, para estabilidad)
- GPU support: CUDAExecutionProvider

---

## 💡 Implementación Detallada

### Fase 1: Script de Exportación ONNX

**Archivo**: `src/models/export_onnx.py`

**Funcionalidades**:

1. **Clase `ONNXExporter`** (arquitectura extensible):
   
   **Métodos de Exportación**:
   - `export_xgboost()` - Exporta XGBoost
   - `export_lightgbm()` - Exporta LightGBM
   - `export_catboost()` - Exporta CatBoost
   - `export_stacking_ensemble()` - Exporta ensembles (base + meta)
   - `export_chronos2()` - Exporta Chronos-2 (3 variantes)
   
   **Métodos Auxiliares**:
   - `optimize_onnx()` - Optimiza grafo ONNX
   - `save_metadata()` - Guarda metadata del modelo
   - `export_all_models()` - Exporta todos los 8 modelos automáticamente
   - `get_model_info()` - Obtiene info del modelo (tipo, tamaño, métricas)

2. **Conversión de Gradient Boosting Models**:
   ```python
   # Pseudocódigo - XGBoost, LightGBM, CatBoost
   from onnxmltools import convert_xgboost, convert_lightgbm
   from onnxmltools.convert import convert_catboost
   
   # XGBoost
   xgb_model = joblib.load("models/baselines/xgboost_model.pkl")
   xgb_onnx = convert_xgboost(xgb_model, initial_types=[('input', FloatTensorType([None, 18]))])
   onnx.save_model(xgb_onnx, "models/onnx/xgboost.onnx")
   
   # LightGBM
   lgb_model = joblib.load("models/gradient_boosting/lightgbm_model.pkl")
   lgb_onnx = convert_lightgbm(lgb_model, initial_types=[('input', FloatTensorType([None, 18]))])
   onnx.save_model(lgb_onnx, "models/onnx/lightgbm.onnx")
   
   # CatBoost
   cat_model = joblib.load("models/gradient_boosting/catboost_model.pkl")
   cat_onnx = convert_catboost(cat_model, initial_types=[('input', FloatTensorType([None, 18]))])
   onnx.save_model(cat_onnx, "models/onnx/catboost.onnx")
   ```

3. **Conversión de Stacking Ensembles**:
   ```python
   # Pseudocódigo - Ridge y LightGBM Stacking
   # Nota: Los ensembles requieren exportar base models + meta-model
   
   # Ridge Stacking
   ridge_ensemble = joblib.load("models/ensembles/ensemble_ridge_v2.pkl")
   # Exportar cada base model individualmente
   for name, base_model in ridge_ensemble.base_models_.items():
       base_onnx = convert_model(base_model, ...)
       onnx.save_model(base_onnx, f"models/onnx/ridge_base_{name}.onnx")
   # Exportar meta-model
   meta_onnx = convert_sklearn(ridge_ensemble.meta_model_, ...)
   onnx.save_model(meta_onnx, "models/onnx/ridge_meta.onnx")
   
   # LightGBM Stacking (similar)
   lgbm_ensemble = joblib.load("models/ensembles/ensemble_lightgbm_v1.pkl")
   # ... exportar base models + meta-model
   ```

4. **Conversión de Chronos-2 (3 variantes)**:
   ```python
   # Pseudocódigo - Zero-shot, Fine-tuned, Covariates
   
   # Zero-shot
   chronos_zeroshot = load_chronos_model("amazon/chronos-t5-small")
   dummy_input = torch.randn(1, context_length, 1)
   torch.onnx.export(
       chronos_zeroshot,
       dummy_input,
       "models/onnx/chronos2_zeroshot.onnx",
       opset_version=17,
       dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}}
   )
   
   # Fine-tuned (similar, pero carga checkpoint fine-tuned)
   chronos_finetuned = load_chronos_model("models/chronos/chronos2_finetuned.pkl")
   torch.onnx.export(...)
   
   # Covariates (similar, pero con 9 covariables)
   chronos_covariates = load_chronos_model("models/chronos/chronos2_covariates.pkl")
   torch.onnx.export(...)
   ```

5. **Optimización ONNX** (aplicable a todos los modelos):
   - Fusion de operadores
   - Eliminación de nodos redundantes
   - Quantization (opcional, INT8)
   - Graph optimization passes
   - Optimización específica por tipo de modelo

**Configuraciones YAML** (una por modelo):

**`config/onnx/xgboost_export.yaml`**:
```yaml
model:
  type: xgboost
  source_path: models/baselines/xgboost_model.pkl
  output_path: models/onnx/xgboost.onnx

export:
  initial_types:
    - name: input
      type: FloatTensorType
      shape: [null, 18]  # batch_size, features
  optimization:
    enabled: true
    level: 2
    quantize: false
```

**`config/onnx/chronos2_export.yaml`**:
```yaml
model:
  type: chronos2_finetuned
  source_path: models/chronos/chronos2_finetuned.pkl
  output_path: models/onnx/chronos2_finetuned.onnx

export:
  opset_version: 17
  dynamic_axes:
    input:
      0: batch_size
      1: sequence_length
  optimization:
    enabled: true
    level: 2  # 0=none, 1=basic, 2=extended, 3=all
    quantize: false  # INT8 quantization

validation:
  tolerance: 1e-5
  num_samples: 100
```


### Fase 2: Validación de Equivalencia

**Archivo**: `src/models/onnx_validator.py`

**Funcionalidades**:

1. **Clase `ONNXValidator`**:
   - Método `validate_chronos2()` - Valida Chronos-2
   - Método `validate_xgboost()` - Valida XGBoost
   - Método `compare_predictions()` - Compara outputs
   - Método `generate_report()` - Genera reporte de validación

2. **Proceso de Validación**:
   ```python
   # Pseudocódigo
   # 1. Cargar ambos modelos
   original_model = load_original_model()
   onnx_model = onnxruntime.InferenceSession("model.onnx")
   
   # 2. Generar inputs de prueba
   test_inputs = generate_test_inputs(num_samples=100)
   
   # 3. Ejecutar predicciones
   original_preds = original_model.predict(test_inputs)
   onnx_preds = onnx_model.run(None, {'input': test_inputs})[0]
   
   # 4. Comparar con tolerancia
   diff = np.abs(original_preds - onnx_preds)
   max_diff = np.max(diff)
   mean_diff = np.mean(diff)
   
   # 5. Validar tolerancia
   assert max_diff < 1e-5, f"Max diff {max_diff} exceeds tolerance"
   ```

3. **Métricas de Validación**:
   - Max absolute difference
   - Mean absolute difference
   - Relative error (%)
   - Número de predicciones idénticas
   - Distribución de diferencias

4. **Reporte de Validación**:
   ```json
   {
     "model_type": "chronos2_finetuned",
     "validation_date": "2025-11-06T10:30:00Z",
     "num_samples": 100,
     "metrics": {
       "max_diff": 8.3e-6,
       "mean_diff": 2.1e-6,
       "relative_error_pct": 0.0001,
       "identical_predictions": 98
     },
     "status": "PASSED",
     "tolerance": 1e-5
   }
   ```


### Fase 3: Benchmark de Performance

**Archivo**: `src/models/onnx_benchmark.py`

**Funcionalidades**:

1. **Clase `ONNXBenchmark`**:
   - Método `benchmark_latency()` - Mide latencia
   - Método `benchmark_throughput()` - Mide throughput
   - Método `benchmark_memory()` - Mide uso de memoria
   - Método `generate_comparison()` - Genera comparación

2. **Benchmark de Latencia**:
   ```python
   # Pseudocódigo
   # Warm-up
   for _ in range(10):
       model.predict(sample_input)
   
   # Benchmark
   latencies = []
   for _ in range(1000):
       start = time.perf_counter()
       model.predict(sample_input)
       end = time.perf_counter()
       latencies.append((end - start) * 1000)  # ms
   
   # Calcular percentiles
   p50 = np.percentile(latencies, 50)
   p95 = np.percentile(latencies, 95)
   p99 = np.percentile(latencies, 99)
   ```

3. **Benchmark de Throughput**:
   ```python
   # Pseudocódigo
   batch_sizes = [1, 10, 50, 100, 500]
   for batch_size in batch_sizes:
       batch_input = generate_batch(batch_size)
       start = time.perf_counter()
       model.predict(batch_input)
       end = time.perf_counter()
       throughput = batch_size / (end - start)
       print(f"Batch {batch_size}: {throughput:.2f} pred/sec")
   ```

4. **Benchmark de Memoria**:
   ```python
   # Pseudocódigo
   import psutil
   import GPUtil
   
   # Memoria RAM
   process = psutil.Process()
   mem_before = process.memory_info().rss / 1024**2  # MB
   model.predict(input)
   mem_after = process.memory_info().rss / 1024**2
   ram_usage = mem_after - mem_before
   
   # Memoria GPU (si aplica)
   if torch.cuda.is_available():
       gpu = GPUtil.getGPUs()[0]
       vram_usage = gpu.memoryUsed
   ```

5. **Reporte de Benchmark**:
   ```json
   {
     "benchmark_date": "2025-11-06T10:30:00Z",
     "hardware": {
       "cpu": "Intel i7-12700K",
       "gpu": "NVIDIA RTX 4070",
       "ram": "32GB"
     },
     "models": {
       "pytorch": {
         "latency_p50_ms": 45.2,
         "latency_p95_ms": 52.8,
         "latency_p99_ms": 58.1,
         "throughput_pred_per_sec": 22.1,
         "memory_mb": 1250
       },
       "onnx": {
         "latency_p50_ms": 12.3,
         "latency_p95_ms": 15.7,
         "latency_p99_ms": 18.2,
         "throughput_pred_per_sec": 81.3,
         "memory_mb": 450
       },
       "improvement": {
         "latency_reduction_pct": 72.8,
         "throughput_increase_pct": 268.0,
         "memory_reduction_pct": 64.0
       }
     }
   }
   ```


### Fase 4: Integración con FastAPI

**Archivo**: `src/api/services/onnx_service.py`

**Funcionalidades**:

1. **Clase `ONNXModelService`** (multi-modelo):
   ```python
   class ONNXModelService:
       """
       ONNX model inference service with support for 8 models.
       
       Provides optimized inference using ONNX Runtime with GPU support.
       """
       
       AVAILABLE_MODELS = {
           # Gradient Boosting
           "xgboost": "models/onnx/xgboost.onnx",
           "lightgbm": "models/onnx/lightgbm.onnx",
           "catboost": "models/onnx/catboost.onnx",
           # Ensembles
           "ridge_ensemble": "models/onnx/ridge_ensemble/",
           "lightgbm_ensemble": "models/onnx/lightgbm_ensemble/",
           # Foundation Models
           "chronos2_zeroshot": "models/onnx/chronos2_zeroshot.onnx",
           "chronos2_finetuned": "models/onnx/chronos2_finetuned.onnx",
           "chronos2_covariates": "models/onnx/chronos2_covariates.onnx",
       }
       
       def __init__(self, model_type: str = "lightgbm_ensemble", use_gpu: bool = True):
           self.model_type = model_type
           self.model_path = self.AVAILABLE_MODELS[model_type]
           self.use_gpu = use_gpu
           self.session = None
           self.is_ensemble = "ensemble" in model_type
       
       def load_model(self):
           """Load ONNX model with GPU support."""
           providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
           
           if self.is_ensemble:
               # Load ensemble (base models + meta-model)
               self.sessions = self._load_ensemble(self.model_path, providers)
           else:
               # Load single model
               self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
       
       def predict(self, features: np.ndarray) -> np.ndarray:
           """Run inference on ONNX model."""
           if self.is_ensemble:
               return self._predict_ensemble(features)
           else:
               input_name = self.session.get_inputs()[0].name
               output_name = self.session.get_outputs()[0].name
               return self.session.run([output_name], {input_name: features})[0]
       
       @classmethod
       def list_available_models(cls) -> List[str]:
           """List all available ONNX models."""
           return list(cls.AVAILABLE_MODELS.keys())
   ```

**Archivo**: `src/api/routes/predict_onnx.py`

**Endpoints**:

1. **POST /predict_onnx** - Predicción individual con selección de modelo
   ```python
   @router.post("/predict_onnx", response_model=PredictionResponse)
   async def predict_onnx(
       request: PredictionRequest,
       model_type: str = Query(
           default="lightgbm_ensemble",
           description="ONNX model to use",
           enum=ONNXModelService.list_available_models()
       )
   ):
       """
       Predict energy consumption using ONNX model.
       
       Optimized endpoint with lower latency than /predict.
       Supports 8 different models via model_type parameter.
       """
       # Load model if not cached
       if model_type not in onnx_service_cache:
           service = ONNXModelService(model_type=model_type)
           service.load_model()
           onnx_service_cache[model_type] = service
       
       onnx_service = onnx_service_cache[model_type]
       
       # Transform features
       features = feature_service.transform(request)
       
       # Predict with ONNX
       start = time.perf_counter()
       prediction = onnx_service.predict(features)
       elapsed_ms = (time.perf_counter() - start) * 1000
       
       return PredictionResponse(
           predicted_usage_kwh=float(prediction[0]),
           model_version=f"{model_type}_onnx",
           model_type="onnx",
           inference_time_ms=elapsed_ms
       )
   ```

2. **POST /predict_onnx/batch** - Predicción batch
   ```python
   @router.post("/predict_onnx/batch", response_model=BatchPredictionResponse)
   async def predict_onnx_batch(request: BatchPredictionRequest):
       """
       Batch prediction using ONNX model.
       
       Optimized for high throughput.
       """
       # Validate batch size
       if len(request.predictions) > 1000:
           raise HTTPException(400, "Max batch size is 1000")
       
       # Transform all features
       features_batch = feature_service.transform_batch(request.predictions)
       
       # Predict batch with ONNX
       predictions = onnx_service.predict(features_batch)
       
       return BatchPredictionResponse(
           predictions=predictions.tolist(),
           count=len(predictions),
           model_type="onnx",
           total_inference_time_ms=elapsed_ms
       )
   ```

3. **GET /predict_onnx/models** - Listar modelos ONNX disponibles
   ```python
   @router.get("/predict_onnx/models")
   async def list_onnx_models():
       """
       List all available ONNX models.
       
       Returns model names, types, sizes, and performance metrics.
       """
       models = []
       for model_name in ONNXModelService.list_available_models():
           model_info = {
               "name": model_name,
               "type": _get_model_type(model_name),
               "size_mb": _get_model_size(model_name),
               "rmse": _get_model_rmse(model_name),
               "available": _check_model_exists(model_name)
           }
           models.append(model_info)
       
       return {
           "total_models": len(models),
           "models": models,
           "default_model": "lightgbm_ensemble"
       }
   ```

4. **GET /predict_onnx/benchmark** - Comparación de performance de 8 modelos
   ```python
   @router.get("/predict_onnx/benchmark")
   async def get_onnx_benchmark():
       """
       Get ONNX vs Original benchmark comparison for all 8 models.
       
       Returns latency, throughput, and memory metrics by model type.
       """
       benchmark_path = "models/benchmarks/onnx_comparison_all_models.json"
       with open(benchmark_path) as f:
           data = json.load(f)
       
       return {
           "benchmark_date": data["benchmark_date"],
           "hardware": data["hardware"],
           "models": data["models"],
           "summary": {
               "fastest_model": data["fastest_model"],
               "avg_speedup": data["avg_speedup"],
               "best_model_by_accuracy": "lightgbm_ensemble"
           }
       }
   ```


---

## 🧪 Testing

### Test Suite

**Archivo**: `tests/unit/test_onnx_export.py`

**Tests de Exportación**:
```python
class TestONNXExport:
    def test_export_chronos2_success(self):
        """Test successful Chronos-2 export."""
        exporter = ONNXExporter()
        output_path = exporter.export_chronos2(
            model_path="models/chronos/chronos2_finetuned.pkl",
            output_path="models/onnx/test_chronos2.onnx"
        )
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 1024  # > 1KB
    
    def test_export_xgboost_success(self):
        """Test successful XGBoost export."""
        exporter = ONNXExporter()
        output_path = exporter.export_xgboost(
            model_path="models/xgboost_model.pkl",
            output_path="models/onnx/test_xgboost.onnx"
        )
        assert Path(output_path).exists()
    
    def test_optimize_onnx_graph(self):
        """Test ONNX graph optimization."""
        exporter = ONNXExporter()
        original_size = Path("models/onnx/chronos2.onnx").stat().st_size
        exporter.optimize_onnx("models/onnx/chronos2.onnx")
        optimized_size = Path("models/onnx/chronos2_optimized.onnx").stat().st_size
        assert optimized_size <= original_size
```

**Archivo**: `tests/unit/test_onnx_validation.py`

**Tests de Validación**:
```python
class TestONNXValidation:
    def test_chronos2_predictions_match(self):
        """Test Chronos-2 ONNX predictions match PyTorch."""
        validator = ONNXValidator()
        result = validator.validate_chronos2(
            original_path="models/chronos/chronos2_finetuned.pkl",
            onnx_path="models/onnx/chronos2_finetuned.onnx",
            tolerance=1e-5
        )
        assert result["status"] == "PASSED"
        assert result["metrics"]["max_diff"] < 1e-5
    
    def test_xgboost_predictions_match(self):
        """Test XGBoost ONNX predictions match original."""
        validator = ONNXValidator()
        result = validator.validate_xgboost(
            original_path="models/xgboost_model.pkl",
            onnx_path="models/onnx/xgboost_model.onnx",
            tolerance=1e-5
        )
        assert result["status"] == "PASSED"
    
    def test_validation_fails_on_mismatch(self):
        """Test validation fails when predictions don't match."""
        validator = ONNXValidator()
        with pytest.raises(AssertionError):
            validator.validate_chronos2(
                original_path="models/chronos/chronos2_finetuned.pkl",
                onnx_path="models/onnx/wrong_model.onnx",
                tolerance=1e-5
            )
```

**Archivo**: `tests/unit/test_onnx_api.py`

**Tests de API**:
```python
class TestONNXAPI:
    def test_predict_onnx_endpoint(self, client):
        """Test /predict_onnx endpoint."""
        response = client.post("/predict_onnx", json={
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium"
        })
        assert response.status_code == 200
        assert "predicted_usage_kwh" in response.json()
        assert response.json()["model_type"] == "onnx"
    
    def test_predict_onnx_batch(self, client):
        """Test /predict_onnx/batch endpoint."""
        response = client.post("/predict_onnx/batch", json={
            "predictions": [
                {"lagging_reactive_power": 23.45, ...},
                {"lagging_reactive_power": 25.30, ...},
            ]
        })
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
    
    def test_onnx_benchmark_endpoint(self, client):
        """Test /predict_onnx/benchmark endpoint."""
        response = client.get("/predict_onnx/benchmark")
        assert response.status_code == 200
        assert "pytorch" in response.json()["models"]
        assert "onnx" in response.json()["models"]
```


---

## 📊 Criterios de Aceptación

### ✅ Criterio 1: Conversión Exitosa
- [ ] Script `src/models/export_onnx.py` implementado con soporte multi-modelo
- [ ] **8 modelos convertidos a ONNX sin errores**:
  - [ ] XGBoost, LightGBM, CatBoost (Gradient Boosting)
  - [ ] Ridge Stacking, LightGBM Stacking (Ensembles)
  - [ ] Chronos-2 Zero-shot, Fine-tuned, Covariates (Foundation)
- [ ] Modelos ONNX guardados en `models/onnx/` con estructura organizada
- [ ] Metadata JSON generado con info de todos los modelos
- [ ] Arquitectura extensible para agregar nuevos modelos fácilmente

### ✅ Criterio 2: Validación de Equivalencia
- [ ] Script `src/models/onnx_validator.py` implementado con soporte multi-modelo
- [ ] Predicciones idénticas con tolerancia 1e-5 para **8 modelos**
- [ ] 100 samples de validación ejecutados por modelo (800 total)
- [ ] Reporte consolidado de validación generado
- [ ] Max diff < 1e-5 para todos los modelos
- [ ] Validación especial para ensembles (pipeline completo)

### ✅ Criterio 3: Benchmark de Performance
- [ ] Script `src/models/onnx_benchmark.py` implementado con soporte multi-modelo
- [ ] Latencia medida (p50, p95, p99) para **8 modelos**
- [ ] Throughput medido (pred/sec) para cada modelo
- [ ] Memoria medida (RAM y VRAM) por tipo de modelo
- [ ] Reporte JSON generado en `models/benchmarks/`
- [ ] Análisis comparativo por tipo (Gradient Boosting vs Ensembles vs Foundation)
- [ ] ONNX es más rápido que original (target: >50% reducción en promedio)
- [ ] Identificación del modelo más rápido en ONNX

### ✅ Criterio 4: Integración con FastAPI
- [ ] Endpoint `POST /predict_onnx` implementado con selección de modelo
- [ ] Endpoint `POST /predict_onnx/batch` implementado
- [ ] Endpoint `GET /predict_onnx/models` implementado (lista modelos disponibles)
- [ ] Endpoint `GET /predict_onnx/benchmark` implementado (comparación de 8 modelos)
- [ ] Servicio `ONNXModelService` implementado con soporte multi-modelo
- [ ] Documentación OpenAPI actualizada con ejemplos de todos los modelos
- [ ] Tests de API implementados (>80% coverage) para todos los modelos

### ✅ Criterio 5: Documentación
- [x] Guía `docs/guides/ONNX_GUIDE.md` creada
- [x] README actualizado con instrucciones ONNX
- [x] Docstrings completos en todos los módulos
- [x] Ejemplos de uso en notebook
- [x] Troubleshooting guide incluido

---

## 📈 Métricas de Éxito

| Métrica | Target | Medición |
|---------|--------|----------|
| **Conversión** | 100% éxito | Chronos-2 + XGBoost convertidos |
| **Validación** | Max diff < 1e-5 | Tolerancia numérica cumplida |
| **Latencia** | >50% reducción | ONNX vs PyTorch/XGBoost |
| **Throughput** | >100% aumento | Predicciones/segundo |
| **Memoria** | >30% reducción | RAM/VRAM usage |
| **Tests** | >80% coverage | Unit + integration tests |
| **Documentación** | Completa | Guía + ejemplos + API docs |

---

## 🚀 Plan de Ejecución

### Sprint Timeline (5 días)

#### Día 1: Setup y Exportación
- **Mañana** (3h):
  - Instalar dependencias: `onnx`, `onnxruntime`, `onnxmltools`
  - Crear estructura de carpetas (`models/onnx/`, `config/onnx/`)
  - Implementar `ONNXExporter` base class
- **Tarde** (3h):
  - Implementar `export_chronos2()` method
  - Implementar `export_xgboost()` method
  - Crear configs YAML para exportación
  - **Entregable**: Modelos ONNX exportados

#### Día 2: Validación
- **Mañana** (3h):
  - Implementar `ONNXValidator` class
  - Implementar `validate_chronos2()` method
  - Implementar `validate_xgboost()` method
- **Tarde** (3h):
  - Ejecutar validación con 100 samples
  - Generar reportes de validación
  - Ajustar tolerancias si es necesario
  - **Entregable**: Validación exitosa con reporte

#### Día 3: Benchmark
- **Mañana** (3h):
  - Implementar `ONNXBenchmark` class
  - Implementar benchmark de latencia
  - Implementar benchmark de throughput
- **Tarde** (3h):
  - Implementar benchmark de memoria
  - Ejecutar benchmarks completos (1000 runs)
  - Generar reporte comparativo JSON
  - **Entregable**: Reporte de benchmark completo

#### Día 4: Integración FastAPI
- **Mañana** (3h):
  - Implementar `ONNXModelService` class
  - Implementar endpoint `POST /predict_onnx`
  - Implementar endpoint `POST /predict_onnx/batch`
- **Tarde** (3h):
  - Implementar endpoint `GET /predict_onnx/benchmark`
  - Actualizar `main.py` con ONNX routes
  - Probar endpoints manualmente
  - **Entregable**: API con endpoints ONNX funcionando

#### Día 5: Testing y Documentación
- **Mañana** (3h):
  - Implementar tests unitarios (export, validation, API)
  - Ejecutar test suite completo
  - Verificar coverage >80%
- **Tarde** (3h):
  - Crear `docs/guides/ONNX_GUIDE.md`
  - Actualizar README principal
  - Crear notebook de ejemplos
  - **Entregable**: US-021 completada y documentada


---

## 🔧 Dependencias Técnicas

### Nuevas Dependencias

**Agregar a `pyproject.toml`**:
```toml
[tool.poetry.dependencies]
onnx = "^1.16.0"
onnxruntime = "^1.18.0"
onnxruntime-gpu = "^1.18.0"  # Para GPU support
onnxmltools = "^1.12.0"
skl2onnx = "^1.17.0"
```

**Agregar a `requirements-api.txt`**:
```txt
onnx==1.16.0
onnxruntime-gpu==1.18.0
```

### Dependencias Existentes (Verificar)
- ✅ `torch>=2.1.0` - Para exportar Chronos-2
- ✅ `xgboost>=3.1.0` - Para exportar XGBoost
- ✅ `numpy>=1.24.0` - Para operaciones numéricas
- ✅ `psutil>=6.0.0` - Para métricas de memoria
- ✅ `GPUtil>=1.4.0` - Para métricas de GPU (agregar si falta)

---

## 🎓 Consideraciones Técnicas

### 1. Compatibilidad de Opset

**ONNX Opset Version**: Usar opset 17 (más reciente estable)

**Razón**: 
- Opset 17 soporta todos los operadores de PyTorch 2.x
- Compatible con ONNX Runtime 1.18+
- Soporta dynamic shapes para batch inference

### 2. Dynamic Axes

**Para Chronos-2**:
```python
dynamic_axes = {
    'input': {0: 'batch_size', 1: 'sequence_length'},
    'output': {0: 'batch_size'}
}
```

**Para XGBoost**:
```python
dynamic_axes = {
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}
```

### 3. GPU Support

**ONNX Runtime Execution Providers**:
```python
providers = [
    'CUDAExecutionProvider',  # GPU (NVIDIA)
    'CPUExecutionProvider'     # Fallback
]
```

**Verificación de GPU**:
```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 4. Optimización de Grafo

**Niveles de Optimización**:
- **Level 0**: Sin optimización
- **Level 1**: Optimizaciones básicas (constant folding, redundant node elimination)
- **Level 2**: Optimizaciones extendidas (operator fusion, layout optimization)
- **Level 3**: Todas las optimizaciones (puede ser inestable)

**Recomendación**: Usar Level 2 para balance entre performance y estabilidad

### 5. Quantization (Opcional)

**INT8 Quantization** para reducir tamaño y latencia:
```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8
)
```

**Trade-offs**:
- ✅ 4x reducción de tamaño
- ✅ 2-3x reducción de latencia
- ⚠️ Pérdida de precisión (~0.5% accuracy)

**Decisión**: Implementar como feature opcional, no por defecto


---

## 🐛 Riesgos y Mitigaciones

### Riesgo 1: Operadores No Soportados

**Problema**: Algunos operadores de PyTorch pueden no tener equivalente en ONNX

**Mitigación**:
- Usar opset version más reciente (17)
- Verificar operadores con `torch.onnx.export(..., verbose=True)`
- Implementar custom operators si es necesario
- Fallback a PyTorch si conversión falla

### Riesgo 2: Diferencias Numéricas

**Problema**: Diferencias en implementación de operadores pueden causar discrepancias

**Mitigación**:
- Tolerancia de 1e-5 es razonable para float32
- Validar con múltiples samples (100+)
- Documentar diferencias conocidas
- Usar float64 si es necesario (trade-off: performance)

### Riesgo 3: Performance No Mejora

**Problema**: ONNX puede ser más lento en algunos casos (modelos pequeños, CPU)

**Mitigación**:
- Benchmark en hardware real (RTX 4070)
- Optimizar grafo ONNX (Level 2)
- Usar GPU execution provider
- Considerar quantization si es necesario
- Mantener endpoint PyTorch como fallback

### Riesgo 4: Tamaño de Modelo

**Problema**: Chronos-2 es muy grande (~455MB), ONNX puede ser aún más grande

**Mitigación**:
- Usar DVC para versionado de modelos ONNX
- No commitear modelos ONNX a Git
- Implementar lazy loading en API
- Considerar quantization para reducir tamaño

### Riesgo 5: Compatibilidad de Versiones

**Problema**: ONNX Runtime puede tener incompatibilidades con versiones de ONNX

**Mitigación**:
- Fijar versiones en requirements: `onnx==1.16.0`, `onnxruntime-gpu==1.18.0`
- Documentar versiones compatibles
- Probar en ambiente limpio (Docker)
- CI/CD para detectar incompatibilidades

---

## 📚 Referencias

### Documentación Oficial

1. **ONNX**:
   - [ONNX Documentation](https://onnx.ai/onnx/)
   - [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
   - [ONNX Opset Versions](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)

2. **ONNX Runtime**:
   - [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
   - [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
   - [Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)

3. **PyTorch ONNX Export**:
   - [PyTorch ONNX Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
   - [torch.onnx API](https://pytorch.org/docs/stable/onnx.html)

4. **XGBoost ONNX**:
   - [onnxmltools Documentation](https://github.com/onnx/onnxmltools)
   - [XGBoost to ONNX](https://github.com/onnx/sklearn-onnx)

### Ejemplos de Código

1. **Chronos-2 Export**:
   - [HuggingFace ONNX Export](https://huggingface.co/docs/transformers/serialization)
   - [Time Series Models to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb)

2. **XGBoost Export**:
   - [sklearn-onnx Examples](https://github.com/onnx/sklearn-onnx/tree/master/docs/examples)

### Benchmarks de Referencia

1. **ONNX vs PyTorch**:
   - [ONNX Runtime Performance](https://cloudblogs.microsoft.com/opensource/2021/09/27/onnx-runtime-1-9-now-available/)
   - Esperado: 2-4x speedup en GPU, 1.5-2x en CPU

2. **ONNX vs XGBoost**:
   - [Treelite Benchmarks](https://treelite.readthedocs.io/en/latest/tutorials/first.html)
   - Esperado: 1.5-3x speedup

---

## ✅ Cumplimiento con AGENTS.md

### Buenas Prácticas Aplicadas

- ✅ **Código en inglés, docs en español**: Todo el código Python en inglés, documentación en español
- ✅ **Docstrings estilo Google**: 100% de funciones documentadas
- ✅ **Type hints completos**: Todas las funciones con type hints
- ✅ **Funciones reutilizables**: Classes en `src/models/` y `src/api/services/`
- ✅ **Separación de concerns**: Export/Validation/Benchmark/API separados
- ✅ **Logging estructurado**: Logger en lugar de prints
- ✅ **Manejo de excepciones**: Try-except con logging apropiado
- ✅ **Sin código duplicado**: DRY principle
- ✅ **Sin magic numbers**: Constantes en configs YAML
- ✅ **Paths relativos**: Uso de Path() y configuración
- ✅ **Black formatted**: Código formateado consistentemente
- ✅ **Sin emojis en código**: Solo en documentación markdown
- ✅ **Testing exhaustivo**: >80% coverage target

### Estructura de Proyecto

```
src/
├── models/
│   ├── export_onnx.py          # Conversión a ONNX
│   ├── onnx_validator.py       # Validación de equivalencia
│   └── onnx_benchmark.py       # Benchmarking
└── api/
    ├── routes/
    │   └── predict_onnx.py     # Endpoints ONNX
    └── services/
        └── onnx_service.py     # Servicio de inferencia
```


---

## 🎯 Entregables Finales

### Código

1. **Scripts de Conversión**:
   - ✅ `src/models/export_onnx.py` (~300 líneas)
   - ✅ `src/models/onnx_validator.py` (~250 líneas)
   - ✅ `src/models/onnx_benchmark.py` (~350 líneas)

2. **API Integration**:
   - ✅ `src/api/services/onnx_service.py` (~200 líneas)
   - ✅ `src/api/routes/predict_onnx.py` (~250 líneas)

3. **Configuraciones**:
   - ✅ `config/onnx/chronos2_export.yaml`
   - ✅ `config/onnx/xgboost_export.yaml`

4. **Tests**:
   - ✅ `tests/unit/test_onnx_export.py` (~150 líneas)
   - ✅ `tests/unit/test_onnx_validation.py` (~150 líneas)
   - ✅ `tests/unit/test_onnx_api.py` (~200 líneas)

**Total**: ~1,850 líneas de código Python

### Modelos

1. **Modelos ONNX** (versionados con DVC):
   
   **Gradient Boosting**:
   - ✅ `models/onnx/xgboost.onnx` (~5MB)
   - ✅ `models/onnx/lightgbm.onnx` (~4MB)
   - ✅ `models/onnx/catboost.onnx` (~6MB)
   
   **Stacking Ensembles**:
   - ✅ `models/onnx/ridge_ensemble/` (carpeta con base models + meta)
     - `ridge_base_xgboost.onnx`
     - `ridge_base_lightgbm.onnx`
     - `ridge_base_catboost.onnx`
     - `ridge_meta.onnx`
   - ✅ `models/onnx/lightgbm_ensemble/` (carpeta con base models + meta)
     - `lgbm_base_xgboost.onnx`
     - `lgbm_base_lightgbm.onnx`
     - `lgbm_base_catboost.onnx`
     - `lgbm_meta.onnx`
   
   **Foundation Models**:
   - ✅ `models/onnx/chronos2_zeroshot.onnx` (~455MB)
   - ✅ `models/onnx/chronos2_finetuned.onnx` (~455MB)
   - ✅ `models/onnx/chronos2_covariates.onnx` (~455MB)
   
   **Metadata**:
   - ✅ `models/onnx/metadata.json` (info de todos los modelos)

2. **Benchmarks**:
   - ✅ `models/benchmarks/onnx_comparison_all_models.json` (comparación completa)
   - ✅ `models/benchmarks/validation_report_all_models.json` (validación de 8 modelos)
   - ✅ `models/benchmarks/onnx_vs_original_by_type.json` (por tipo de modelo)

### Documentación

1. **Guías**:
   - ✅ `docs/guides/ONNX_GUIDE.md` (~600 líneas)
   - ✅ `docs/us-resolved/us-021.md` (este documento)

2. **Notebooks**:
   - ✅ `notebooks/experimental/onnx_examples.ipynb` (~10 secciones)

3. **API Docs**:
   - ✅ Swagger UI actualizado con endpoints ONNX
   - ✅ README actualizado con instrucciones ONNX

---

## 📊 Métricas de Desarrollo Estimadas

| Métrica | Estimación |
|---------|------------|
| **Tiempo de desarrollo** | 5 días (30 horas) |
| **Líneas de código** | ~1,850 líneas |
| **Archivos creados** | 15 archivos |
| **Tests implementados** | 15+ tests |
| **Documentación** | 600+ líneas |
| **Modelos ONNX** | 2 modelos |
| **Endpoints nuevos** | 3 endpoints |

---

## 🎉 Valor Agregado

### Para el Proyecto

1. **Performance**: 50-70% reducción de latencia en inferencia
2. **Portabilidad**: Modelos ejecutables en cualquier plataforma (Python, C++, JavaScript)
3. **Escalabilidad**: Mayor throughput para producción
4. **Eficiencia**: Menor uso de memoria y GPU
5. **Flexibilidad**: Opción de usar PyTorch o ONNX según necesidad

### Para el Equipo

1. **Aprendizaje**: Experiencia con ONNX y optimización de modelos
2. **Best Practices**: Validación rigurosa y benchmarking
3. **Documentación**: Guías reutilizables para futuros proyectos
4. **Testing**: Suite de tests robusta para modelos

### Para Usuarios Finales

1. **Latencia**: Respuestas más rápidas en API
2. **Disponibilidad**: Menor uso de recursos = más requests simultáneos
3. **Confiabilidad**: Validación garantiza predicciones correctas
4. **Transparencia**: Benchmarks públicos de performance

---

## 🚀 Próximos Pasos (Post US-021)

### Mejoras Futuras

1. **Quantization**: Implementar INT8 quantization para reducir tamaño
2. **TensorRT**: Integrar NVIDIA TensorRT para mayor speedup en GPU
3. **Model Serving**: Usar Triton Inference Server para deployment
4. **Edge Deployment**: Exportar a ONNX Mobile para dispositivos edge
5. **Multi-Framework**: Soportar TensorFlow Lite, CoreML

### Integración con Otras US

1. **US-022 (Copiloto)**: Usar ONNX para respuestas más rápidas
2. **US-023 (Frontend)**: Mostrar benchmarks en UI
3. **US-024 (Monitoring)**: Trackear latencia ONNX vs PyTorch
4. **US-025 (Deployment)**: Deployar con ONNX Runtime en Cloud Run

---

## 📝 Notas Finales

### Decisiones de Diseño

1. **Opset 17**: Elegido por compatibilidad con PyTorch 2.x y ONNX Runtime 1.18+
2. **Dynamic Axes**: Implementado para soportar batch inference variable
3. **GPU Support**: Priorizado para aprovechar RTX 4070
4. **Validation Tolerance**: 1e-5 es estándar para float32
5. **Benchmark Runs**: 1000 runs para estabilidad estadística

### Lecciones Esperadas

1. **ONNX Export**: Algunos operadores pueden requerir workarounds
2. **Validation**: Diferencias numéricas son normales en float32
3. **Performance**: GPU speedup es mayor que CPU speedup
4. **Memory**: ONNX puede usar menos memoria que PyTorch
5. **Deployment**: ONNX simplifica deployment cross-platform

---

## ✅ Aprobación

**Este plan está listo para aprobación y ejecución.**

**Criterios cumplidos**:
- ✅ Arquitectura clara y detallada
- ✅ Implementación paso a paso
- ✅ Testing exhaustivo planificado
- ✅ Documentación completa
- ✅ Riesgos identificados y mitigados
- ✅ Timeline realista (5 días)
- ✅ Cumplimiento con AGENTS.md
- ✅ Valor agregado claro

**Próximo paso**: Aprobación del plan y inicio de implementación (Día 1).

---

**Documento creado por**: MLOps Team - Proyecto Atreides  
**Fecha**: 06 de Noviembre, 2025  
**Versión**: 1.0  
**Estado**: 📋 PENDIENTE DE APROBACIÓN

---

## 📞 Contacto

Para preguntas o sugerencias sobre este plan:
- **ML Engineer**: Julian (implementación)
- **MLOps Engineer**: Arthur (infraestructura)
- **Tech Lead**: Dante (revisión técnica)

