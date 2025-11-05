# Guía Completa de Dagster - Energy Optimization Copilot

**Última actualización**: 5 de Noviembre, 2025  
**Versión Dagster**: 1.9.0

Esta guía unificada cubre todo lo que necesitas saber sobre Dagster en el proyecto: workflows, multi-modelo, configuración y troubleshooting.

---

## 📋 Tabla de Contenidos

1. [Introducción](#-introducción)
2. [Arquitectura](#️-arquitectura)
3. [Jobs Disponibles](#-jobs-disponibles)
4. [Inicio Rápido](#-inicio-rápido)
5. [Multi-Modelo con YAML](#️-multi-modelo-con-yaml)
6. [Configuración del Launchpad](#-configuración-del-launchpad)
7. [GPU Automático](#-gpu-automático)
8. [MLflow Integration](#-mlflow-integration)
9. [Modelos Soportados](#-modelos-soportados)
10. [Troubleshooting](#-troubleshooting)

---

## 🎯 Introducción

Este proyecto usa **Dagster** como orquestador de pipelines de ML, reemplazando a Prefect v3 que presentaba problemas de UI y dependencias de workers.

### Ventajas de Dagster

- ✅ UI funciona sin workers
- ✅ Visualización clara del DAG
- ✅ Type-safe con Python types
- ✅ Compatible con MLflow, DVC, Polars
- ✅ Soporte multi-modelo via YAML
- ✅ Configuración prellenada en Launchpad

---

## 🏗️ Arquitectura

### Estructura de Código

```
src/dagster_pipeline/
├── definitions.py          # Entry point - Define all jobs
├── ops.py                  # Ops para modelos tradicionales (10 ops)
├── chronos_ops.py          # Ops para Chronos-2 (8 ops)
├── jobs.py                 # Jobs de modelos tradicionales
├── chronos_jobs.py         # Jobs de Chronos-2 (3 jobs)
└── working_pipeline.py     # Pipeline principal multi-modelo
```

### Conceptos Clave

| Concepto | Descripción | Equivalente Prefect |
|----------|-------------|---------------------|
| **Op** | Unidad básica de computación | Task |
| **Job** | Grafo de ejecución de Ops | Flow |
| **Asset** | Dato materializable | Output |
| **Resource** | Dependencia externa (DB, API) | Block |
| **Config** | Configuración de ops | Parameters |

---

## 📦 Jobs Disponibles

### 1. `complete_training_job`

**Modelos**: XGBoost, LightGBM, CatBoost, 2 Ensembles

**Pipeline** (10 ops):
```
Load Config → Load Data → Validate Data → Train Model → Evaluate →
Check Threshold → Log MLflow → Save Artifacts → DVC Add → Notification
```

**Características**:
- ✅ Multi-modelo: Cambia modelo editando solo el YAML
- ✅ GPU con fallback automático a CPU
- ✅ Validación de calidad de datos
- ✅ Threshold checks (RMSE, R², MAE)
- ✅ MLflow tracking completo
- ✅ Versionado con DVC
- ✅ Configuración prellenada (XGBoost por defecto)

**Configuraciones disponibles**:
- `config/training/xgboost_config.yaml` (por defecto)
- `config/training/lightgbm_config.yaml`
- `config/training/catboost_config.yaml`
- `config/training/ensemble_lightgbm_config.yaml`
- `config/training/ensemble_ridge_config.yaml`

---

### 2. `chronos_zeroshot_job`

**Modelo**: Chronos-2 (amazon/chronos-t5-small) - Zero-shot

**Pipeline** (6 ops):
```
Load Config → Load Data → Load Pipeline → Prepare Data → Evaluate → Log MLflow
```

**Características**:
- ✅ **Sin entrenamiento** (usa modelo pre-entrenado)
- ✅ Inference directa en datos temporales
- ✅ GPU automático (RTX 4070)
- ✅ Batch processing eficiente
- ✅ Métricas solo en MLflow (modelo NO se guarda)
- ✅ Configuración prellenada

**Configuración**:
- `config/training/chronos2_zeroshot_config.yaml` (por defecto)

**Uso**: Para baseline rápido sin costo de entrenamiento (~2-3 min).

---

### 3. `chronos_finetuned_job`

**Modelo**: Chronos-2 fine-tuned (sin covariables)

**Pipeline** (8 ops):
```
Load Config → Load Data → Load Pipeline → Prepare Data → Fine-tune →
Evaluate → Save Model → Log MLflow
```

**Características**:
- ✅ Fine-tuning de 1000 steps (configurable)
- ✅ Learning rate: 1e-5
- ✅ Gradient accumulation: 4
- ✅ Batch size: 8
- ✅ GPU requerido (≈455MB modelo)
- ✅ Modelo guardado en `models/foundation/`
- ✅ Métricas en MLflow
- ✅ Configuración prellenada

**Configuración**:
- `config/training/chronos2_finetuned_config.yaml` (por defecto)

**Uso**: Para adaptar Chronos-2 a nuestro dominio (siderurgia).

---

### 4. `chronos_covariates_job`

**Modelo**: Chronos-2 fine-tuned con 9 covariables pasadas

**Pipeline** (8 ops):
```
Load Config → Load Data → Load Pipeline → Prepare Data (with covariates) →
Fine-tune → Evaluate → Save Model → Log MLflow
```

**Características**:
- ✅ Fine-tuning con contexto multivariado
- ✅ 9 past_covariates:
  - `Lagging_Current_Reactive.Power_kVarh`
  - `Leading_Current_Reactive_Power_kVarh`
  - `CO2(tCO2)`
  - `Lagging_Current_Power_Factor`
  - `Leading_Current_Power_Factor`
  - `NSM`
  - `WeekStatus`
  - `Day_of_week`
  - `Load_Type`
- ✅ Mejor rendimiento esperado (RMSE <42 kWh)
- ✅ GPU requerido
- ✅ Modelo guardado (~455MB)
- ✅ Configuración prellenada

**Configuración**:
- `config/training/chronos2_covariates_config.yaml` (por defecto)

**Uso**: Para máximo rendimiento aprovechando variables correlacionadas.

---

## 🚀 Inicio Rápido

### 1. Iniciar Dagster

**PowerShell (Windows)**:
```powershell
.\scripts\start-dagster.ps1

# O especificar puerto
.\scripts\start-dagster.ps1 -Port 3001
```

**Bash (Linux/macOS)**:
```bash
./scripts/start-dagster.sh

# O especificar puerto
./scripts/start-dagster.sh 3001
```

### 2. Abrir UI

Navegar a: **http://127.0.0.1:3000**

### 3. Ejecutar un Job

1. **Click en "Jobs"** (sidebar izquierdo)
2. **Selecciona un job** (ej: `chronos_finetuned_job`)
3. **Click en "Launchpad"**
4. **Revisar configuración prellenada** (ya viene con valores por defecto)
5. **Opcional**: Editar el `config_path` si quieres usar otra configuración
6. **Click en "Launch Run"**

### 4. Monitorear Ejecución

- Ver progreso en tiempo real
- Logs de cada op
- Tiempo de ejecución
- Success/failure status
- Gráfico del DAG

---

## 🎛️ Multi-Modelo con YAML

La arquitectura multi-modelo permite **cambiar de modelo SIN tocar código**, solo editando el YAML.

### Ejemplo: Cambiar de XGBoost a CatBoost

**Antes** (XGBoost):
```yaml
# config/training/my_model.yaml
model:
  type: xgboost
  parameters:
    max_depth: 10
    learning_rate: 0.01
```

**Después** (CatBoost):
```yaml
# config/training/my_model.yaml
model:
  type: catboost
  parameters:
    depth: 8
    learning_rate: 0.03
    iterations: 500
```

**Ejecución**: La misma (Dagster detecta el cambio automáticamente).

### Routing Interno

El routing se hace en `src/dagster_pipeline/ops.py` → `train_model_op()`:

```python
def train_model_op(context: OpExecutionContext, data: tuple, cfg: dict) -> Any:
    model_type = cfg['model']['type']
    
    if model_type == "xgboost":
        model = _train_xgboost(context, X_train, y_train, cfg)
    elif model_type == "lightgbm":
        model = _train_lightgbm(context, X_train, y_train, cfg)
    elif model_type == "catboost":
        model = _train_catboost(context, X_train, y_train, cfg)
    elif model_type in ["ensemble_lightgbm", "ensemble_ridge"]:
        model = _train_ensemble(context, X_train, y_train, cfg, data)
    # ...
```

Para Chronos-2, el routing está en `chronos_ops.py` → `train_chronos_model_op()`:

```python
def train_chronos_model_op(context, pipeline, prepared_data, cfg):
    model_type = cfg['model']['type']
    
    if model_type == "chronos2_zeroshot":
        return pipeline  # No training
    elif model_type == "chronos2_finetuned":
        return _finetune_chronos_simple(...)
    elif model_type == "chronos2_covariates":
        return _finetune_chronos_covariates(...)
```

---

## 🔧 Configuración del Launchpad

### Configuración Prellenada (Nuevo!)

Todos los jobs ahora vienen con **configuración prellenada** en el Launchpad. Ya no necesitas escribir el YAML manualmente.

#### Job: `complete_training_job`

**Configuración por defecto** (XGBoost):
```yaml
config:
  config_path: "config/training/xgboost_config.yaml"
```

**Para cambiar a otro modelo**, simplemente edita el path:
```yaml
config:
  config_path: "config/training/lightgbm_config.yaml"
```

#### Job: `chronos_zeroshot_job`

**Configuración por defecto**:
```yaml
config:
  config_path: "config/training/chronos2_zeroshot_config.yaml"
```

#### Job: `chronos_finetuned_job`

**Configuración por defecto**:
```yaml
config:
  config_path: "config/training/chronos2_finetuned_config.yaml"
```

#### Job: `chronos_covariates_job`

**Configuración por defecto**:
```yaml
config:
  config_path: "config/training/chronos2_covariates_config.yaml"
```

### Personalizar Configuración

Si quieres usar una configuración personalizada:

1. Crea tu archivo YAML en `config/training/`
2. En el Launchpad, edita el `config_path`:
```yaml
config:
  config_path: "config/training/mi_config_personalizado.yaml"
```

---

## 🎮 GPU Automático

Todos los modelos (XGBoost, LightGBM, CatBoost, Chronos-2) tienen **detección automática de GPU** con fallback a CPU.

### Implementación

```python
import torch

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
context.log.info(f"Using device: {device}")

# Configure model
if device == "cuda":
    # GPU config
    model = load_model(device_map="auto", torch_dtype=torch.bfloat16)
else:
    # CPU config
    model = load_model(device_map="cpu", torch_dtype=torch.float32)
```

### GPU Disponible

- **Hardware**: NVIDIA RTX 4070
- **VRAM**: 12GB
- **CUDA**: 12.1
- **PyTorch**: 2.1.0+cu121

### Logs

Con GPU:
```
INFO - Detecting GPU...
INFO - GPU detected: NVIDIA GeForce RTX 4070
INFO - Using device: cuda
INFO - Loading model with bfloat16 precision
```

Sin GPU:
```
INFO - Detecting GPU...
INFO - No GPU detected, using CPU
INFO - Using device: cpu
INFO - Loading model with float32 precision
```

---

## 📊 MLflow Integration

Todos los jobs loggean automáticamente a MLflow.

### Configuración

**MLflow Server**: http://localhost:5000

**Experimentos**:
- `steel_energy_optimization` - Modelos tradicionales
- `chronos2_zeroshot` - Chronos-2 zero-shot
- `chronos2_finetuned` - Chronos-2 fine-tuned
- `chronos2_covariates` - Chronos-2 con covariables

### Qué se Loggea

#### Modelos Tradicionales (XGBoost, etc.)
- ✅ Parameters (learning_rate, max_depth, etc.)
- ✅ Metrics (RMSE, MAE, R², MAPE)
- ✅ Tags (model_type, gpu_used, framework)
- ✅ **Modelo completo** (via `mlflow.sklearn.log_model`)
- ✅ Artifacts (model.pkl, metrics.json)

#### Chronos-2
- ✅ Parameters (context_length, num_steps, lr, etc.)
- ✅ Metrics (RMSE, MAE, R², MAPE)
- ✅ Tags (model_type, approach, device)
- ✅ **Path al modelo** (no el modelo completo)
- ⚠️ **Modelo NO loggeado** (demasiado grande: ~455MB)

**Razón**: MLflow tiene límite de 100MB por artifact. Chronos-2 fine-tuned pesa ~455MB.

### Ver Resultados

1. **Iniciar MLflow UI**:
   ```bash
   poetry run mlflow ui
   ```

2. **Abrir**: http://localhost:5000

3. **Comparar runs**:
   - Click en experimento
   - Selecciona múltiples runs
   - Click en "Compare"
   - Ver gráficas de métricas

---

## 🎯 Modelos Soportados

### ✅ Modelos Tradicionales (Solo cambiar YAML)

| Modelo | Config File | Tiempo | GPU |
|--------|-------------|--------|-----|
| **XGBoost** | `xgboost_config.yaml` | ~1 min | ✅ |
| **LightGBM** | `lightgbm_config.yaml` | ~1 min | ✅ |
| **CatBoost** | `catboost_config.yaml` | ~1 min | ✅ |
| **Ensemble LightGBM** | `ensemble_lightgbm_config.yaml` | ~3-5 min | ✅ |
| **Ensemble Ridge** | `ensemble_ridge_config.yaml` | ~3-5 min | ✅ |

### ✅ Foundation Models (Chronos-2)

| Modelo | Config File | Tiempo | GPU |
|--------|-------------|--------|-----|
| **Zero-Shot** | `chronos2_zeroshot_config.yaml` | ~30s | ✅ |
| **Fine-Tuned** | `chronos2_finetuned_config.yaml` | ~2-4 min (10 steps) | ✅ Requerido |
| **Covariates** | `chronos2_covariates_config.yaml` | ~4-8 min (10 steps) | ✅ Requerido |

### Cuándo usar cada modelo

| Modelo | Mejor Para | Ventajas | Desventajas |
|--------|-----------|----------|-------------|
| **XGBoost** | Baseline rápido | Rápido, robusto, GPU support | Puede overfittear |
| **LightGBM** | Datasets grandes | Muy rápido, eficiente memoria | Sensible a hiperparámetros |
| **CatBoost** | Features categóricas | Maneja categorías nativamente | Más lento |
| **Ensemble** | Máxima precisión | Mejor performance | Lento, complejo |
| **Chronos Zero-Shot** | Baseline temporal rápido | Sin entrenamiento | Performance limitado |
| **Chronos Fine-Tuned** | Forecasting adaptado | Captura patrones temporales | Requiere GPU, lento |
| **Chronos Covariates** | Máxima precisión temporal | Usa contexto multivariado | Más lento, más complejo |

### Recomendación de Workflow

```bash
# 1. Entrenar modelos tradicionales
# En Dagster UI, ejecutar secuencialmente:
# - xgboost_config.yaml
# - lightgbm_config.yaml  
# - catboost_config.yaml

# 2. Comparar en MLflow
# http://localhost:5000
# Ver cuál tiene mejor RMSE/R²

# 3. Entrenar ensemble con los 3 mejores
# - ensemble_lightgbm_config.yaml

# 4. Probar Chronos-2
# - chronos2_zeroshot_config.yaml (baseline)
# - chronos2_finetuned_config.yaml (adaptado)
# - chronos2_covariates_config.yaml (máxima precisión)

# 5. Comparar todos en MLflow
# Elegir el mejor modelo para producción
```

---

## 🔧 Troubleshooting

### Problema: "Port 3000 already in use"

**Solución**: Usa otro puerto
```powershell
.\scripts\start-dagster.ps1 -Port 3001
```

---

### Problema: "Module dagster not found"

**Causa**: Entorno virtual no activado

**Solución**:
```bash
poetry install
poetry shell
```

---

### Problema: "CUDA out of memory"

**Causa**: Batch size muy grande para GPU

**Solución**: Reducir batch_size en config
```yaml
chronos:
  batch_size: 4  # Era 8
```

---

### Problema: "Config file not found"

**Causa**: Path relativo incorrecto en Launchpad

**Solución**: Usar path relativo desde raíz del proyecto
```yaml
config:
  config_path: "config/training/chronos2_finetuned_config.yaml"  # ✅ Correcto
  # NO: "C:/Users/..." (path absoluto)
```

---

### Problema: "Chronos model too large for MLflow"

**Esperado**: Los modelos Chronos-2 fine-tuned NO se loggean a MLflow.

**Solución**: El path se loggea como parámetro. Para cargar modelo:
```python
import mlflow

run = mlflow.get_run(run_id)
model_path = run.data.params['model_path']

from chronos import Chronos2Pipeline
pipeline = Chronos2Pipeline.from_pretrained(model_path)
```

---

### Problema: "No jobs visible in UI"

**Causa**: Error al cargar definitions

**Solución**: Verificar logs del servidor
```bash
# Ver logs en la terminal donde ejecutaste start-dagster
# Buscar errores de import o sintaxis
```

---

### Problema: "Job execution failed"

**Pasos de debugging**:

1. **Ver logs del op que falló** en la UI
2. **Verificar configuración** del YAML
3. **Verificar datos** existen en `data/processed/`
4. **Verificar GPU** si es Chronos: `nvidia-smi`
5. **Ejecutar script directo** para debugging:
   ```bash
   poetry run python src/models/train_xgboost.py
   ```

---

## 📚 Referencias

- [Dagster Docs](https://docs.dagster.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Chronos-2 GitHub](https://github.com/amazon-science/chronos-forecasting)
- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [US-019 Documentation](../us-resolved/us-019.md) - Migración a Dagster

---

## 🎓 Best Practices

1. **Usa la UI** para desarrollo y debugging
2. **Usa CLI** para producción y automatización
3. **Valida configs** antes de ejecutar (especialmente paths)
4. **Monitorea GPU** con `nvidia-smi` durante fine-tuning
5. **Compara runs** en MLflow antes de elegir modelo
6. **Versiona modelos** grandes con DVC (XGBoost, CatBoost)
7. **NO versiones Chronos-2** con DVC (demasiado grande)
8. **Usa configuración prellenada** - ya viene lista para usar
9. **Empieza con modelos rápidos** (XGBoost, Zero-Shot) antes de entrenar modelos lentos

---

## ✅ Resumen

### Para modelos tradicionales (XGBoost, LightGBM, CatBoost, Ensemble):
- ✅ **Solo cambias el YAML** en el Launchpad
- ✅ **Configuración prellenada** - lista para usar
- ✅ **El código actual ya funciona**
- ✅ **GPU fallback automático**
- ✅ **Mismo pipeline de 10 ops**

### Para Chronos-2:
- ✅ **Pipeline separado** (6-8 ops)
- ✅ **Configuración prellenada** - lista para usar
- ✅ **3 variantes** (zero-shot, fine-tuned, covariates)
- ✅ **GPU automático** con fallback
- ✅ **MLflow integration** completa

**Conclusión**: Puedes entrenar **8 modelos diferentes** (5 tradicionales + 3 Chronos) solo cambiando el YAML en el Launchpad. Todo viene prellenado y listo para usar.

---

**Autor**: MLOps Team - Proyecto Atreides  
**Fecha**: 5 de Noviembre, 2025  
**Versión**: 2.0 (Consolidada)
