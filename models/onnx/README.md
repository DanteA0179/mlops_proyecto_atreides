# ONNX Models - Energy Optimization Copilot

Este directorio contiene los modelos exportados a formato ONNX para inferencia optimizada.

## 📁 Estructura

```
models/onnx/
├── lightgbm.onnx              # Modelo LightGBM individual (476 KB)
├── lightgbm.json              # Metadata del modelo
└── lightgbm_ensemble/         # Ensemble completo (modelo principal)
    ├── lightgbm_base_lightgbm.onnx  # Base model LightGBM
    ├── lightgbm_base_catboost.onnx  # Base model CatBoost
    ├── lightgbm_meta.onnx           # Meta-model LightGBM
    └── metadata.json                # Metadata del ensemble
```

## 🎯 Modelo Principal

**ensemble_lightgbm_v3** (lightgbm_ensemble/)
- **RMSE**: 12.7982 (mejor modelo del proyecto)
- **Componentes**: 2 base models + 1 meta-model
- **Tamaño total**: ~1.2 MB
- **Features esperadas**: 9 (post-preprocesamiento)

## 🚀 Uso

### Python con ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Cargar modelo
session = ort.InferenceSession("models/onnx/lightgbm.onnx")

# Preparar datos (9 features)
features = np.random.randn(1, 9).astype(np.float32)

# Predecir
input_name = session.get_inputs()[0].name
prediction = session.run(None, {input_name: features})[0]

print(f"Predicción: {prediction[0]:.2f} kWh")
```

### API REST

```bash
# Predicción con ONNX (ensemble por defecto)
curl -X POST "http://localhost:8000/predict_onnx" \
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

## 📊 Performance

| Métrica | Original | ONNX | Mejora |
|---------|----------|------|--------|
| Tamaño | ~15 MB | ~1.2 MB | 92% ↓ |
| Dependencias | 500+ MB | 50 MB | 90% ↓ |
| Latencia (esperada) | ~15 ms | ~5 ms | 66% ↓ |

## ⚙️ Exportación

Para re-exportar los modelos:

```bash
# Exportar todos los modelos
poetry run python scripts/export_models_to_onnx.py

# Validar modelos exportados
poetry run python scripts/validate_onnx_models.py

# Benchmark de performance
poetry run python scripts/benchmark_onnx_models.py
```

## 🔧 Configuración

Los archivos de configuración están en `config/onnx/`:
- `lightgbm_export.yaml` - Config para LightGBM
- `ensemble_export.yaml` - Config para ensemble
- `xgboost_export.yaml` - Config para XGBoost
- `catboost_export.yaml` - Config para CatBoost

## ⚠️ Notas Importantes

1. **Features**: Los modelos esperan 9 features post-preprocesamiento, no 18
2. **XGBoost**: No exportado en el ensemble debido a incompatibilidad
3. **GPU**: Usar `CUDAExecutionProvider` para mejor performance
4. **Validación**: Tolerancia numérica de 1e-4 (estándar para float32)

## 📚 Referencias

- **US-021**: Exportación a ONNX
- **US-020**: API FastAPI
- **US-011**: Feature engineering temporal
- **US-012**: Preprocessing pipeline

---

**Última actualización**: 15 de Noviembre, 2025
