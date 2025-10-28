# Guía de Versionado de Modelos

## Descripción

El sistema de entrenamiento ahora soporta versionado automático de modelos, permitiendo guardar múltiples versiones sin sobrescribir resultados anteriores.

## Uso Básico

### 1. Versión Automática (Timestamp)

Si no especificas una versión, se genera automáticamente con timestamp:

```bash
poetry run python src/models/train_xgboost.py --n-trials 100 --cv-folds 5
```

**Genera**:
- Modelo: `models/baselines/xgboost_v_20251027_183045.pkl`
- Métricas: `reports/metrics/xgboost_test_metrics_v_20251027_183045.json`
- Plots: `reports/figures/xgboost_predictions_v_20251027_183045.png`
- MLflow run: `xgboost_baseline_v_20251027_183045`

### 2. Versión Manual

Especifica tu propia versión para experimentos:

```bash
poetry run python src/models/train_xgboost.py \
    --n-trials 100 \
    --cv-folds 5 \
    --model-version v1
```

**Genera**:
- Modelo: `models/baselines/xgboost_v1.pkl`
- Métricas: `reports/metrics/xgboost_test_metrics_v1.json`
- Plots: `reports/figures/xgboost_predictions_v1.png`
- MLflow run: `xgboost_baseline_v1`

## Ejemplos de Uso

### Experimento con Diferentes Hiperparámetros

```bash
# Baseline con pocos trials
poetry run python src/models/train_xgboost.py \
    --n-trials 50 \
    --cv-folds 3 \
    --model-version baseline_50trials

# Optimizado con más trials
poetry run python src/models/train_xgboost.py \
    --n-trials 200 \
    --cv-folds 5 \
    --model-version optimized_200trials

# Experimento rápido
poetry run python src/models/train_xgboost.py \
    --n-trials 10 \
    --cv-folds 2 \
    --model-version quick_test
```

### Comparación de Configuraciones

```bash
# Configuración conservadora
poetry run python src/models/train_xgboost.py \
    --n-trials 100 \
    --cv-folds 5 \
    --model-version conservative

# Configuración agresiva (más trials)
poetry run python src/models/train_xgboost.py \
    --n-trials 300 \
    --cv-folds 10 \
    --model-version aggressive
```

## Archivos Generados por Versión

Para cada versión `{version}`, se generan:

### Modelo
- `models/baselines/xgboost_{version}.pkl` - Modelo serializado
- `models/baselines/xgboost_{version}.json` - Metadata (checksum, size)

### Métricas
- `reports/metrics/xgboost_test_metrics_{version}.json` - Métricas de test
- `reports/metrics/optuna_trials_{version}.csv` - Historial de Optuna

### Visualizaciones
- `reports/figures/xgboost_predictions_{version}.png` - Predictions vs Actual
- `reports/figures/xgboost_residuals_{version}.png` - Análisis de residuos
- `reports/figures/xgboost_feature_importance_{version}.png` - Feature importance

### Reportes
- `reports/xgboost_evaluation_{version}.md` - Reporte completo

### MLflow
- Run name: `xgboost_baseline_{version}`
- Tag: `model_version: {version}`
- Todos los artifacts organizados por versión

## Comparación de Modelos

### Listar Modelos Disponibles

```bash
# Ver todos los modelos
ls models/baselines/xgboost_*.pkl

# Ver métricas de todos los modelos
ls reports/metrics/xgboost_test_metrics_*.json
```

### Comparar Métricas

```python
import json
from pathlib import Path

# Cargar métricas de diferentes versiones
metrics_dir = Path("reports/metrics")

versions = ["v1", "v2", "optimized_200trials"]
for version in versions:
    metrics_file = metrics_dir / f"xgboost_test_metrics_{version}.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"{version:20s} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
```

### Comparar en MLflow

1. Abrir MLflow UI: http://localhost:5000
2. Ir al experimento: `steel_energy_xgboost_baseline`
3. Filtrar por tag: `model_version`
4. Comparar métricas side-by-side

## Mejores Prácticas

### Nomenclatura de Versiones

**Recomendado**:
- `v1`, `v2`, `v3` - Versiones principales
- `baseline` - Modelo baseline inicial
- `optimized` - Modelo optimizado
- `experiment_feature_x` - Experimentos con features
- `ensemble_v1` - Modelos ensemble

**Evitar**:
- Nombres muy largos
- Caracteres especiales (solo letras, números, guiones bajos)
- Espacios

### Organización de Experimentos

```bash
# 1. Baseline inicial
poetry run python src/models/train_xgboost.py \
    --n-trials 50 \
    --cv-folds 3 \
    --model-version baseline

# 2. Optimización incremental
poetry run python src/models/train_xgboost.py \
    --n-trials 100 \
    --cv-folds 5 \
    --model-version v1_optimized

# 3. Mejor modelo final
poetry run python src/models/train_xgboost.py \
    --n-trials 200 \
    --cv-folds 5 \
    --model-version production_candidate
```

### Limpieza de Modelos Antiguos

```bash
# Eliminar modelos de prueba
rm models/baselines/xgboost_quick_test.*
rm reports/metrics/*_quick_test.*
rm reports/figures/*_quick_test.*

# Mantener solo los mejores modelos
# (revisar métricas antes de eliminar)
```

## Integración con DVC

Para versionar modelos con DVC:

```bash
# Agregar modelo específico a DVC
dvc add models/baselines/xgboost_v1.pkl

# Commit a git
git add models/baselines/xgboost_v1.pkl.dvc
git commit -m "model: add xgboost v1 (RMSE: 12.83)"

# Tag para versión importante
git tag -a model-xgboost-v1 -m "XGBoost baseline v1"
```

## Troubleshooting

### Error: Modelo ya existe

Si intentas usar una versión que ya existe, el modelo se sobrescribirá. Para evitarlo:

```bash
# Opción 1: Usar timestamp automático (no especificar --model-version)
poetry run python src/models/train_xgboost.py --n-trials 100

# Opción 2: Usar versión única
poetry run python src/models/train_xgboost.py --model-version v2_attempt2
```

### Espacio en Disco

Cada modelo ocupa aproximadamente:
- Modelo (.pkl): ~5-10 MB
- Métricas (.json, .csv): ~1-2 MB
- Plots (.png): ~1-2 MB
- **Total por versión**: ~10-15 MB

Monitorear espacio:

```bash
# Ver tamaño de modelos
du -sh models/baselines/

# Ver tamaño de reportes
du -sh reports/
```

## Ejemplo Completo

```bash
# 1. Entrenar modelo baseline
poetry run python src/models/train_xgboost.py \
    --n-trials 50 \
    --cv-folds 3 \
    --model-version baseline

# 2. Ver resultados
cat reports/metrics/xgboost_test_metrics_baseline.json

# 3. Si es bueno, entrenar versión optimizada
poetry run python src/models/train_xgboost.py \
    --n-trials 200 \
    --cv-folds 5 \
    --model-version v1_production

# 4. Comparar en MLflow
# http://localhost:5000

# 5. Versionar mejor modelo con DVC
dvc add models/baselines/xgboost_v1_production.pkl
git add models/baselines/xgboost_v1_production.pkl.dvc
git commit -m "model: add production candidate v1"
git tag model-v1-production
```

---

**Última actualización**: 2025-10-27
**Versión**: 1.0
