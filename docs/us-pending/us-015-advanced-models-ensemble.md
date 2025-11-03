# US-015: Advanced Models & Ensemble - Planning Document

**Estado**: 📋 PENDIENTE  
**Prioridad**: ALTA  
**Estimación**: 5-6 días  
**Responsables**: ML Engineer (Julian) + MLOps Engineer (Arthur)  
**Sprint**: 1 (Segunda Entrega)

---

## 📋 Resumen Ejecutivo

Implementar tres modelos adicionales de gradient boosting (LightGBM, CatBoost) y un modelo de stacking ensemble que combine las predicciones de XGBoost, LightGBM y CatBoost para superar el benchmark CUBIST (RMSE normalizado < 0.205). Esta US busca maximizar el performance con mínimo esfuerzo reutilizando la infraestructura existente de US-013.

### Objetivo Principal

**Superar el benchmark CUBIST**: RMSE normalizado < 0.205 (actualmente en 0.3614 con XGBoost)

### Estrategia

1. **LightGBM** (Día 1-2): Competidor directo de XGBoost, alta probabilidad de mejora
2. **CatBoost** (Día 3-4): Manejo sofisticado de categóricas, puede capturar interacciones únicas
3. **Stacking Ensemble** (Día 5): Combinar los 3 modelos para maximizar performance
4. **Análisis Comparativo** (Día 6): Selección del mejor modelo y documentación

---

## 🎯 Objetivos de Negocio

### Problema a Resolver


El modelo XGBoost baseline (US-013) logró RMSE normalizado de 0.3614, lo cual está 76.4% por encima del target de 0.205. Necesitamos explorar modelos alternativos y técnicas de ensemble para cerrar esta brecha.

### Valor Esperado

- **Mejora de Performance**: Reducir RMSE de 0.3614 a <0.205 (43% de mejora)
- **Robustez**: Ensemble reduce varianza y mejora generalización
- **Comparación Justa**: Evaluar múltiples algoritmos con misma metodología
- **Aprendizaje**: Identificar qué características del dataset favorecen cada modelo

### Métricas de Éxito

| Métrica | Baseline (XGBoost) | Target | Stretch Goal |
|---------|-------------------|--------|--------------|
| RMSE normalizado | 0.3614 | < 0.205 | < 0.180 |
| RMSE (kWh) | 12.84 | < 7.28 | < 6.40 |
| R² | 0.8693 | > 0.95 | > 0.96 |
| MAE (kWh) | 3.53 | < 2.50 | < 2.00 |

---

## 🎯 Criterios de Aceptación

### 1. Modelo LightGBM Entrenado y Optimizado

**Requisitos**:
- Pipeline sklearn con LGBMRegressor
- Reutilizar infraestructura de US-013 (xgboost_trainer.py como base)
- Optimización de hiperparámetros con Optuna (100 trials mínimo)
- Cross-validation 5-fold
- GPU acceleration habilitado (device="gpu")

**Hiperparámetros a optimizar**:
- `num_leaves`: (20, 150)
- `max_depth`: (-1, 15) # -1 = sin límite
- `learning_rate`: (0.01, 0.3)
- `n_estimators`: (50, 300)
- `min_child_samples`: (5, 100)
- `subsample`: (0.6, 1.0)
- `colsample_bytree`: (0.6, 1.0)
- `reg_alpha`: (0, 10)
- `reg_lambda`: (0, 10)

**Entregables**:
- `src/models/lightgbm_trainer.py` (adaptado de xgboost_trainer.py)
- `src/models/train_lightgbm.py` (script ejecutable)
- Modelo serializado: `models/baselines/lightgbm_{version}.pkl`
- Metadata JSON con checksum
- Feature importance (gain, split)
- MLflow tracking completo

**Métricas esperadas**:
- RMSE: 10-12 kWh (mejora de 10-20% vs XGBoost)
- R²: 0.88-0.90
- Training time: <5 min (100 trials)

---

### 2. Modelo CatBoost Entrenado y Optimizado

**Requisitos**:
- Pipeline sklearn con CatBoostRegressor
- Especificar features categóricas explícitamente: `Load_Type`, `WeekStatus`
- Optimización con Optuna (100 trials mínimo)
- Cross-validation 5-fold
- GPU acceleration habilitado (task_type="GPU")

**Hiperparámetros a optimizar**:
- `depth`: (4, 10)
- `learning_rate`: (0.01, 0.3)
- `iterations`: (50, 300)
- `l2_leaf_reg`: (1, 10)
- `border_count`: (32, 255)
- `bagging_temperature`: (0, 1)
- `random_strength`: (0, 10)

**Configuración especial**:
```python
cat_features = ['Load_Type', 'WeekStatus']  # Índices o nombres
model = CatBoostRegressor(
    cat_features=cat_features,
    task_type="GPU",
    verbose=False
)
```

**Entregables**:
- `src/models/catboost_trainer.py`
- `src/models/train_catboost.py`
- Modelo serializado: `models/baselines/catboost_{version}.pkl`
- Metadata JSON
- Feature importance (PredictionValuesChange)
- MLflow tracking completo

**Métricas esperadas**:
- RMSE: 11-13 kWh
- R²: 0.87-0.89
- Training time: <8 min (100 trials, CatBoost es más lento)

---

### 3. Stacking Ensemble Implementado

**Arquitectura**:

```
Level 0 (Base Models):
├── XGBoost (ya entrenado en US-013)
├── LightGBM (nuevo)
└── CatBoost (nuevo)
         ↓
    Predicciones
         ↓
Level 1 (Meta-Model):
└── Ridge Regression o LightGBM
         ↓
    Predicción Final
```

**Requisitos**:
- Usar modelos ya optimizados (no re-entrenar)
- Generar predicciones out-of-fold para training del meta-modelo
- Probar 2 meta-modelos:
  1. Ridge Regression (simple, rápido)
  2. LightGBM (puede capturar no-linealidades)
- Cross-validation para evaluar ensemble
- Análisis de pesos/importancia de cada modelo base

**Estrategia de implementación**:
1. Cargar modelos optimizados (XGBoost, LightGBM, CatBoost)
2. Generar predicciones out-of-fold en train set (5-fold CV)
3. Usar predicciones como features para meta-modelo
4. Entrenar meta-modelo en estas predicciones
5. Evaluar en test set

**Entregables**:
- `src/models/stacking_ensemble.py` (clase StackingEnsemble)
- `src/models/train_ensemble.py` (script ejecutable)
- Modelo serializado: `models/ensembles/stacking_{version}.pkl`
- Análisis de contribución de cada modelo base
- Comparación Ridge vs LightGBM como meta-modelo
- MLflow tracking completo

**Métricas esperadas**:
- RMSE: 9-11 kWh (mejora de 15-30% vs mejor modelo individual)
- R²: 0.90-0.92
- Reducción de varianza entre folds

---

### 4. Comparación Exhaustiva de Modelos

**Análisis requerido**:

**A. Tabla comparativa de métricas**:
| Modelo | RMSE (kWh) | RMSE Norm | MAE (kWh) | R² | MAPE (%) | Training Time |
|--------|------------|-----------|-----------|-----|----------|---------------|
| XGBoost | 12.84 | 0.3614 | 3.53 | 0.8693 | 31.46 | 4 min |
| LightGBM | TBD | TBD | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| Ensemble (Ridge) | TBD | TBD | TBD | TBD | TBD | TBD |
| Ensemble (LGBM) | TBD | TBD | TBD | TBD | TBD | TBD |
| **CUBIST (Target)** | **8.56** | **0.2410** | **-** | **-** | **-** | **-** |

**B. Análisis de errores por segmento**:
- Por `Load_Type` (Light, Medium, Maximum)
- Por `WeekStatus` (Weekday, Weekend)
- Por hora del día (picos vs valles)
- Por rango de consumo (bajo, medio, alto)

**C. Feature importance comparison**:
- Top 10 features de cada modelo
- Consenso entre modelos
- Features únicas importantes por modelo

**D. Análisis de correlación de errores**:
- ¿Los modelos cometen errores en los mismos puntos?
- Correlación de residuos entre modelos
- Diversidad del ensemble (baja correlación = mejor ensemble)

**E. Visualizaciones**:
- Predictions vs Actual (todos los modelos en un plot)
- Residuals distribution (boxplot comparativo)
- Error por segmento (bar plots)
- Feature importance comparison (side-by-side)
- Scatter matrix de predicciones (correlación entre modelos)

**Entregables**:
- `notebooks/exploratory/11_model_comparison.ipynb`
- `reports/model_comparison_report.md`
- `reports/figures/model_comparison_*.png` (5-8 visualizaciones)
- Recomendación final de modelo para producción

---

### 5. MLflow Tracking Completo

**Experimentos a crear**:
- `steel_energy_lightgbm_baseline`
- `steel_energy_catboost_baseline`
- `steel_energy_stacking_ensemble`

**Métricas a loggear** (para cada modelo):
- RMSE, MAE, R², MAPE (train/val/test/cv)
- RMSE normalizado
- Max error, Min error
- Percentiles de error (p50, p75, p90, p95)
- Training time, Inference time
- Model size

**Artifacts a loggear**:
- Modelo serializado (solo path, no el archivo completo)
- Feature importance (JSON + PNG)
- Predictions vs Actual plot
- Residuals plot
- Optuna trials (CSV)
- Evaluation report (Markdown)

**Tags a asignar**:
- `model_type`: "lightgbm", "catboost", "ensemble"
- `experiment_type`: "baseline", "optimized", "ensemble"
- `model_version`: timestamp o manual
- `gpu_enabled`: "true"/"false"
- `optimization_method`: "optuna"
- `ensemble_type`: "stacking" (si aplica)
- `meta_model`: "ridge"/"lightgbm" (si aplica)

---

### 6. Sistema de Versionado y Reproducibilidad

**Requisitos**:
- Todos los modelos con versionado automático (timestamp)
- Metadata JSON para cada modelo:
  - Checksum MD5
  - Hiperparámetros
  - Métricas de test
  - Fecha de entrenamiento
  - Versión de librerías (lightgbm, catboost)
  - Random seed usado
- Reproducibilidad 100% con random_state=42
- Scripts ejecutables con argumentos CLI

**Estructura de directorios**:
```
models/
├── baselines/
│   ├── xgboost_v1.pkl (ya existe)
│   ├── lightgbm_v1.pkl (nuevo)
│   ├── lightgbm_v1.json
│   ├── catboost_v1.pkl (nuevo)
│   └── catboost_v1.json
└── ensembles/
    ├── stacking_ridge_v1.pkl (nuevo)
    ├── stacking_ridge_v1.json
    ├── stacking_lgbm_v1.pkl (nuevo)
    └── stacking_lgbm_v1.json
```

---

## 🛠️ Implementación Técnica

### Fase 1: LightGBM (Días 1-2)

**Paso 1.1: Crear lightgbm_trainer.py**

**Base**: Copiar `src/models/xgboost_trainer.py` y adaptar

**Funciones a implementar**:
- `check_gpu_availability()` - Reutilizar de xgboost_trainer
- `create_lightgbm_pipeline()` - Similar a create_xgboost_pipeline
- `train_lightgbm_with_cv()` - Adaptar train_xgboost_with_cv
- `optimize_lightgbm_with_optuna()` - Adaptar optimize_xgboost_with_optuna
- `evaluate_model()` - Reutilizar de xgboost_trainer
- `get_feature_names_from_pipeline()` - Reutilizar

**Cambios clave**:
```python
# XGBoost → LightGBM
from xgboost import XGBRegressor  # ❌
from lightgbm import LGBMRegressor  # ✅

DEFAULT_PARAMS = {
    "device": "gpu" if GPU_AVAILABLE else "cpu",
    "n_jobs": 1 if GPU_AVAILABLE else -1,
    "random_state": 42,
    "verbose": -1,
}

SEARCH_SPACE = {
    "num_leaves": (20, 150),
    "max_depth": (-1, 15),
    # ... resto de hiperparámetros
}
```

**Paso 1.2: Crear train_lightgbm.py**

**Base**: Copiar `src/models/train_xgboost.py` y adaptar

**Pipeline de 10 pasos** (igual que XGBoost):
1. Setup y configuración
2. Generación de versión del modelo
3. Carga de datos preprocesados (US-012)
4. Optimización con Optuna (100 trials)
5. Cross-validation con mejores parámetros
6. Evaluación en train/val/test
7. Extracción de feature importance (gain + split)
8. Generación de visualizaciones
9. Guardado de modelo y artifacts
10. Logging a MLflow

**Paso 1.3: Ejecutar y validar**

```bash
# Prueba rápida (5 trials)
poetry run python src/models/train_lightgbm.py --n-trials 5 --cv-folds 3

# Entrenamiento completo
poetry run python src/models/train_lightgbm.py --n-trials 100 --cv-folds 5
```

**Validaciones**:
- ✅ GPU detection funciona
- ✅ Optuna converge
- ✅ Modelo se serializa correctamente
- ✅ MLflow registra todo
- ✅ Feature importance se genera

**Tiempo estimado**: 1.5-2 días

---

### Fase 2: CatBoost (Días 3-4)

**Paso 2.1: Crear catboost_trainer.py**

**Base**: Copiar `lightgbm_trainer.py` y adaptar

**Diferencias clave con LightGBM**:
```python
from catboost import CatBoostRegressor, Pool

# CatBoost requiere especificar features categóricas
def create_catboost_pipeline(model_params, cat_features=None):
    # cat_features puede ser lista de índices o nombres
    model = CatBoostRegressor(
        cat_features=cat_features,
        **model_params
    )
    # ... resto del pipeline

# Para Optuna, usar CatBoost Pool para eficiencia
def optimize_catboost_with_optuna(...):
    train_pool = Pool(
        X_train, 
        y_train,
        cat_features=cat_features
    )
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    # ... optimización
```

**Identificar features categóricas**:
```python
# En el dataset preprocesado (US-012)
categorical_features = [
    'Load_Type_Light_Load',      # One-hot encoded
    'Load_Type_Medium_Load',     # One-hot encoded
    'Load_Type_Maximum_Load',    # One-hot encoded
    'WeekStatus',                # 0/1 (Weekday/Weekend)
]

# O usar columnas originales si no están one-hot encoded
# Verificar en data/processed/steel_train.parquet
```

**Paso 2.2: Crear train_catboost.py**

**Similar a train_lightgbm.py** con ajustes:
- Especificar `cat_features` en todos los pasos
- Usar `task_type="GPU"` en lugar de `device="gpu"`
- Feature importance: usar `PredictionValuesChange` (más informativo para CatBoost)

**Paso 2.3: Ejecutar y validar**

```bash
# Prueba rápida
poetry run python src/models/train_catboost.py --n-trials 5 --cv-folds 3

# Entrenamiento completo
poetry run python src/models/train_catboost.py --n-trials 100 --cv-folds 5
```

**Tiempo estimado**: 1.5-2 días

---

### Fase 3: Stacking Ensemble (Día 5)

**Paso 3.1: Crear stacking_ensemble.py**

**Clase principal**:
```python
class StackingEnsemble:
    """
    Stacking ensemble of multiple models.
    
    Level 0: Base models (XGBoost, LightGBM, CatBoost)
    Level 1: Meta-model (Ridge or LightGBM)
    """
    
    def __init__(self, base_models, meta_model, cv_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
    
    def fit(self, X, y):
        # Generate out-of-fold predictions
        # Train meta-model on these predictions
    
    def predict(self, X):
        # Get predictions from base models
        # Feed to meta-model
```

**Funciones auxiliares**:
- `generate_oof_predictions()` - Out-of-fold predictions para training
- `train_meta_model()` - Entrenar meta-modelo
- `evaluate_ensemble()` - Evaluar ensemble completo
- `analyze_base_model_contributions()` - Pesos/importancia de cada modelo

**Paso 3.2: Crear train_ensemble.py**

**Pipeline**:
1. Cargar modelos optimizados:
   - `models/baselines/xgboost_v1.pkl`
   - `models/baselines/lightgbm_v1.pkl`
   - `models/baselines/catboost_v1.pkl`

2. Cargar datos preprocesados (US-012)

3. Generar predicciones out-of-fold (5-fold CV):
   - Para cada fold:
     - Train base models en 4 folds
     - Predict en 1 fold (out-of-fold)
   - Resultado: matriz (n_samples, 3) de predicciones

4. Entrenar meta-modelo Ridge:
   - Input: predicciones de los 3 modelos
   - Output: predicción final
   - Analizar coeficientes (pesos de cada modelo)

5. Entrenar meta-modelo LightGBM:
   - Similar a Ridge
   - Puede capturar interacciones no-lineales

6. Evaluar ambos ensembles en test set

7. Comparar y seleccionar mejor

8. Guardar ensemble completo (base models + meta-model)

9. Logging a MLflow

**Paso 3.3: Ejecutar y validar**

```bash
poetry run python src/models/train_ensemble.py \
    --xgboost-model models/baselines/xgboost_v1.pkl \
    --lightgbm-model models/baselines/lightgbm_v1.pkl \
    --catboost-model models/baselines/catboost_v1.pkl \
    --meta-model ridge
```

**Tiempo estimado**: 1 día

---

### Fase 4: Análisis Comparativo (Día 6)

**Paso 4.1: Crear notebook 11_model_comparison.ipynb**

**Secciones**:

1. **Setup y carga de modelos**
   - Cargar los 5 modelos (XGBoost, LightGBM, CatBoost, Ensemble Ridge, Ensemble LGBM)
   - Cargar datos de test

2. **Comparación de métricas**
   - Tabla comparativa
   - Bar plots de RMSE, MAE, R²
   - Identificar mejor modelo

3. **Análisis de errores por segmento**
   - Por Load_Type
   - Por WeekStatus
   - Por hora del día
   - Por rango de consumo

4. **Feature importance comparison**
   - Top 10 de cada modelo
   - Heatmap de importancias
   - Consenso entre modelos

5. **Análisis de diversidad del ensemble**
   - Correlación de predicciones entre modelos
   - Scatter matrix
   - Correlación de residuos

6. **Visualizaciones comparativas**
   - Predictions vs Actual (todos los modelos)
   - Residuals distribution
   - Error distribution por modelo

7. **Conclusiones y recomendaciones**
   - Mejor modelo para producción
   - Trade-offs (performance vs complejidad vs tiempo)
   - Próximos pasos

**Paso 4.2: Crear model_comparison_report.md**

**Contenido**:
- Resumen ejecutivo
- Tabla de métricas
- Análisis detallado de cada modelo
- Fortalezas y debilidades
- Recomendación final
- Apéndices con visualizaciones

**Paso 4.3: Generar visualizaciones**

**Figuras a crear** (guardar en `reports/figures/`):
1. `model_comparison_metrics.png` - Bar plot de métricas
2. `model_comparison_predictions.png` - Predictions vs Actual
3. `model_comparison_residuals.png` - Boxplot de residuos
4. `model_comparison_by_load_type.png` - Error por Load_Type
5. `model_comparison_feature_importance.png` - Heatmap
6. `ensemble_diversity.png` - Scatter matrix de predicciones
7. `ensemble_weights.png` - Contribución de cada modelo base

**Tiempo estimado**: 1 día

---

## 📊 Resultados Esperados

### Predicciones de Performance

**Escenario Conservador**:
| Modelo | RMSE (kWh) | RMSE Norm | R² | Mejora vs XGBoost |
|--------|------------|-----------|-----|-------------------|
| XGBoost (baseline) | 12.84 | 0.3614 | 0.8693 | - |
| LightGBM | 12.20 | 0.3434 | 0.8750 | 5% |
| CatBoost | 12.50 | 0.3519 | 0.8720 | 2.6% |
| Ensemble (Ridge) | 11.80 | 0.3321 | 0.8800 | 8.1% |
| Ensemble (LGBM) | 11.60 | 0.3265 | 0.8830 | 9.7% |

**Escenario Optimista**:
| Modelo | RMSE (kWh) | RMSE Norm | R² | Mejora vs XGBoost |
|--------|------------|-----------|-----|-------------------|
| XGBoost (baseline) | 12.84 | 0.3614 | 0.8693 | - |
| LightGBM | 11.50 | 0.3237 | 0.8850 | 10.4% |
| CatBoost | 11.80 | 0.3321 | 0.8800 | 8.1% |
| Ensemble (Ridge) | 10.80 | 0.3040 | 0.9000 | 15.9% |
| Ensemble (LGBM) | 10.50 | 0.2955 | 0.9050 | 18.2% |

**Target CUBIST**: RMSE Norm = 0.2410

**Análisis**:
- Escenario conservador: No alcanzamos target (0.3265 vs 0.2410)
- Escenario optimista: Nos acercamos pero no alcanzamos (0.2955 vs 0.2410)
- **Conclusión**: Necesitaremos feature engineering adicional o modelos más avanzados

---

## 🚧 Riesgos y Mitigaciones

### Riesgo 1: No alcanzar el target CUBIST

**Probabilidad**: Alta (70%)  
**Impacto**: Alto

**Mitigación**:
- Documentar claramente la metodología usada
- Comparar con CUBIST en términos de features y preprocessing
- Si no alcanzamos, proponer US-016 con feature engineering avanzado
- Considerar que CUBIST puede usar metodología diferente

### Riesgo 2: Overfitting del ensemble

**Probabilidad**: Media (40%)  
**Impacto**: Medio

**Mitigación**:
- Usar out-of-fold predictions para training del meta-modelo
- Cross-validation riguroso
- Regularización en meta-modelo (Ridge, L2 en LightGBM)
- Monitorear gap entre train y test metrics

### Riesgo 3: Modelos muy correlacionados (ensemble no mejora)

**Probabilidad**: Media (50%)  
**Impacto**: Medio

**Mitigación**:
- Analizar correlación de predicciones antes de crear ensemble
- Si correlación > 0.95, considerar solo el mejor modelo individual
- Diversificar hiperparámetros en optimización
- Considerar agregar modelo de familia diferente (ej. Random Forest)

### Riesgo 4: Tiempo de entrenamiento excesivo

**Probabilidad**: Baja (20%)  
**Impacto**: Bajo

**Mitigación**:
- GPU acceleration en todos los modelos
- Reducir trials de Optuna si es necesario (50 en lugar de 100)
- Paralelizar entrenamientos (LightGBM y CatBoost en paralelo)
- Usar early stopping en Optuna

### Riesgo 5: Problemas con GPU en CatBoost

**Probabilidad**: Media (30%)  
**Impacto**: Bajo

**Mitigación**:
- Fallback automático a CPU si GPU falla
- Documentar configuración de GPU para CatBoost
- Probar en CPU primero si hay problemas

---

## 📁 Estructura de Archivos

### Código Fuente (Nuevo)

```
src/models/
├── lightgbm_trainer.py          # ~500 líneas (adaptado de xgboost_trainer.py)
├── train_lightgbm.py            # ~380 líneas (adaptado de train_xgboost.py)
├── catboost_trainer.py          # ~520 líneas (similar a lightgbm_trainer.py)
├── train_catboost.py            # ~400 líneas (similar a train_lightgbm.py)
├── stacking_ensemble.py         # ~350 líneas (nuevo)
└── train_ensemble.py            # ~300 líneas (nuevo)
```

### Modelos Generados

```
models/
├── baselines/
│   ├── lightgbm_v1.pkl          # ~2-3 MB
│   ├── lightgbm_v1.json
│   ├── catboost_v1.pkl          # ~5-8 MB (CatBoost es más grande)
│   └── catboost_v1.json
└── ensembles/
    ├── stacking_ridge_v1.pkl    # ~15 MB (incluye 3 base models)
    ├── stacking_ridge_v1.json
    ├── stacking_lgbm_v1.pkl     # ~15 MB
    └── stacking_lgbm_v1.json
```

### Reportes y Visualizaciones

```
reports/
├── metrics/
│   ├── lightgbm_test_metrics_v1.json
│   ├── catboost_test_metrics_v1.json
│   ├── ensemble_ridge_test_metrics_v1.json
│   ├── ensemble_lgbm_test_metrics_v1.json
│   ├── optuna_trials_lightgbm_v1.csv
│   └── optuna_trials_catboost_v1.csv
├── figures/
│   ├── lightgbm_predictions_v1.png
│   ├── lightgbm_residuals_v1.png
│   ├── lightgbm_feature_importance_v1.png
│   ├── catboost_predictions_v1.png
│   ├── catboost_residuals_v1.png
│   ├── catboost_feature_importance_v1.png
│   ├── ensemble_predictions_v1.png
│   ├── model_comparison_metrics.png
│   ├── model_comparison_predictions.png
│   ├── model_comparison_residuals.png
│   ├── model_comparison_by_load_type.png
│   ├── model_comparison_feature_importance.png
│   ├── ensemble_diversity.png
│   └── ensemble_weights.png
└── model_comparison_report.md   # ~1000 líneas
```

### Notebooks

```
notebooks/exploratory/
└── 11_model_comparison.ipynb    # Análisis comparativo completo
```

---

## 💻 Comandos de Ejecución

### Entrenamiento Individual

```bash
# LightGBM
poetry run python src/models/train_lightgbm.py --n-trials 100 --cv-folds 5

# CatBoost
poetry run python src/models/train_catboost.py --n-trials 100 --cv-folds 5

# Con versión manual
poetry run python src/models/train_lightgbm.py \
    --n-trials 100 \
    --cv-folds 5 \
    --model-version production_v1
```

### Ensemble

```bash
# Stacking con Ridge
poetry run python src/models/train_ensemble.py \
    --xgboost-model models/baselines/xgboost_v1.pkl \
    --lightgbm-model models/baselines/lightgbm_v1.pkl \
    --catboost-model models/baselines/catboost_v1.pkl \
    --meta-model ridge \
    --cv-folds 5

# Stacking con LightGBM
poetry run python src/models/train_ensemble.py \
    --xgboost-model models/baselines/xgboost_v1.pkl \
    --lightgbm-model models/baselines/lightgbm_v1.pkl \
    --catboost-model models/baselines/catboost_v1.pkl \
    --meta-model lightgbm \
    --cv-folds 5
```

### Pruebas Rápidas

```bash
# LightGBM rápido (5 trials, 3 folds)
poetry run python src/models/train_lightgbm.py --n-trials 5 --cv-folds 3

# CatBoost rápido
poetry run python src/models/train_catboost.py --n-trials 5 --cv-folds 3
```

---

## 🧪 Testing y Validación

### Tests Unitarios a Crear

```
tests/unit/
├── test_lightgbm_trainer.py     # ~15 tests
├── test_catboost_trainer.py     # ~15 tests
└── test_stacking_ensemble.py    # ~20 tests
```

**Tests clave**:
- GPU detection funciona
- Pipeline se crea correctamente
- Modelos se serializan/deserializan
- Feature importance se extrae
- Ensemble genera predicciones correctas
- Out-of-fold predictions son válidas
- Meta-modelo se entrena correctamente

### Validación Manual

**Checklist**:
- [ ] LightGBM entrena sin errores
- [ ] CatBoost maneja categóricas correctamente
- [ ] GPU se usa en ambos modelos
- [ ] Optuna converge en <100 trials
- [ ] Modelos se guardan con versionado
- [ ] MLflow registra todo correctamente
- [ ] Ensemble mejora vs modelos individuales
- [ ] Predicciones son razonables (no NaN, no negativos)
- [ ] Reproducibilidad con random_state=42

---

## 📚 Dependencias

### Nuevas Librerías

```toml
# pyproject.toml
[tool.poetry.dependencies]
lightgbm = "^4.1.0"      # Ya debe estar instalado
catboost = "^1.2.2"      # NUEVO - agregar
```

**Instalación**:
```bash
poetry add catboost
```

**Verificar versiones**:
```bash
poetry show lightgbm catboost
```

---

## 🎓 Lecciones Esperadas

### Aprendizajes Técnicos

1. **Comparación de algoritmos de gradient boosting**
   - Diferencias entre XGBoost, LightGBM, CatBoost
   - Cuándo usar cada uno

2. **Manejo de features categóricas**
   - One-hot encoding vs categorical encoding nativo
   - Ventajas de CatBoost

3. **Stacking ensemble**
   - Cómo generar out-of-fold predictions
   - Selección de meta-modelo
   - Trade-off complejidad vs mejora

4. **Optimización de hiperparámetros**
   - Search spaces específicos por algoritmo
   - Convergencia de Optuna

### Aprendizajes de Negocio

1. **Benchmarking**
   - Importancia de metodología consistente
   - Comparación justa entre modelos

2. **Trade-offs**
   - Performance vs complejidad
   - Performance vs tiempo de entrenamiento
   - Performance vs interpretabilidad

---

## 🔄 Próximos Pasos (Post US-015)

### Si alcanzamos el target (RMSE < 0.205)

**US-016: Model Deployment & API**
- Optimizar modelo para inference
- Crear endpoint FastAPI
- Testing de latencia
- Deployment a Cloud Run

### Si NO alcanzamos el target

**US-016: Advanced Feature Engineering**
- Lag features (1h, 2h, 4h, 8h, 24h)
- Rolling statistics (mean, std, min, max)
- Interacciones entre features
- Polynomial features
- Target encoding para categóricas
- Time-based features (día del mes, semana del año)

**US-017: Deep Learning Models**
- Temporal Fusion Transformer (TFT)
- N-BEATS
- LSTM/GRU
- Comparar con gradient boosting

---

## 📊 Métricas de Calidad del Código

### Targets

| Métrica | Target | Cómo Medir |
|---------|--------|------------|
| Líneas de código | ~2,500 | `cloc src/models/` |
| Funciones | >20 | Contar funciones |
| Docstrings | 100% | Revisión manual |
| Type hints | 100% | Revisión manual |
| Tests coverage | >70% | `pytest --cov` |
| Ruff warnings | <5 | `ruff check .` |
| Black compliant | Sí | `black --check .` |

### Performance Targets

| Métrica | Target |
|---------|--------|
| LightGBM training (100 trials) | <5 min |
| CatBoost training (100 trials) | <8 min |
| Ensemble training | <2 min |
| Total time (todo el pipeline) | <20 min |

---

## ✅ Definition of Done

### Código
- [ ] `lightgbm_trainer.py` implementado y testeado
- [ ] `train_lightgbm.py` ejecutable con CLI args
- [ ] `catboost_trainer.py` implementado y testeado
- [ ] `train_catboost.py` ejecutable con CLI args
- [ ] `stacking_ensemble.py` implementado y testeado
- [ ] `train_ensemble.py` ejecutable con CLI args
- [ ] Todos los módulos con docstrings y type hints
- [ ] Código formateado con Black
- [ ] Sin warnings de Ruff (o <5 menores)

### Modelos
- [ ] LightGBM entrenado y optimizado (100 trials)
- [ ] CatBoost entrenado y optimizado (100 trials)
- [ ] Ensemble Ridge entrenado
- [ ] Ensemble LightGBM entrenado
- [ ] Todos los modelos serializados con versionado
- [ ] Metadata JSON generado para cada modelo

### MLflow
- [ ] 3 experimentos creados (lightgbm, catboost, ensemble)
- [ ] Todos los parámetros loggeados
- [ ] Todas las métricas loggeadas
- [ ] Artifacts subidos (plots, CSVs, reports)
- [ ] Tags asignados correctamente

### Análisis
- [ ] Notebook `11_model_comparison.ipynb` completo
- [ ] `model_comparison_report.md` generado
- [ ] 7 visualizaciones creadas
- [ ] Tabla comparativa de métricas
- [ ] Análisis de errores por segmento
- [ ] Feature importance comparison
- [ ] Recomendación final documentada

### Documentación
- [ ] `us-015.md` completion doc creado
- [ ] README actualizado con nuevos modelos
- [ ] Ejemplos de uso documentados
- [ ] Lecciones aprendidas documentadas

### Testing
- [ ] Tests unitarios para nuevos módulos (>70% coverage)
- [ ] Validación manual completada
- [ ] Reproducibilidad verificada

### Calidad
- [ ] Código revisado por peer
- [ ] Performance targets alcanzados
- [ ] Sin errores en ejecución
- [ ] Documentación clara y completa

---

## 🎯 Criterios de Éxito Final

### Mínimo Viable (Must Have)

✅ **3 modelos entrenados**: XGBoost (ya existe), LightGBM, CatBoost  
✅ **1 ensemble funcional**: Stacking con Ridge o LightGBM  
✅ **Mejora vs baseline**: Al menos 5% de mejora en RMSE  
✅ **MLflow tracking**: Completo para todos los modelos  
✅ **Análisis comparativo**: Notebook y reporte completos  
✅ **Documentación**: US completion doc detallado  

### Deseable (Should Have)

✅ **2 ensembles**: Ridge y LightGBM comparados  
✅ **Mejora vs baseline**: 10-15% de mejora en RMSE  
✅ **Análisis profundo**: Errores por segmento, feature importance  
✅ **Tests unitarios**: >70% coverage  
✅ **Visualizaciones**: 7+ figuras de alta calidad  

### Aspiracional (Nice to Have)

✅ **Alcanzar target CUBIST**: RMSE normalizado < 0.205  
✅ **Mejora vs baseline**: >20% de mejora en RMSE  
✅ **Modelo production-ready**: Seleccionado y optimizado  
✅ **Insights accionables**: Recomendaciones para feature engineering  

---

## 📞 Puntos de Contacto

### Revisiones Intermedias

**Día 2**: Revisión de LightGBM
- ¿Entrena correctamente?
- ¿Mejora vs XGBoost?
- ¿GPU funciona?

**Día 4**: Revisión de CatBoost
- ¿Maneja categóricas correctamente?
- ¿Performance comparable?
- ¿Listo para ensemble?

**Día 5**: Revisión de Ensemble
- ¿Mejora vs modelos individuales?
- ¿Qué meta-modelo es mejor?
- ¿Listo para análisis final?

### Decisiones Clave

**Decisión 1** (Día 2): ¿Continuar con CatBoost?
- Si LightGBM no mejora significativamente, reevaluar
- Considerar invertir tiempo en feature engineering

**Decisión 2** (Día 5): ¿Qué meta-modelo usar?
- Ridge (simple) vs LightGBM (complejo)
- Basado en performance y complejidad

**Decisión 3** (Día 6): ¿Modelo final para producción?
- Basado en métricas, complejidad, interpretabilidad
- Documentar decisión y trade-offs

---

## 🏆 Impacto Esperado

### Técnico

- **3 nuevos modelos baseline** de alta calidad
- **Sistema de ensemble** reutilizable
- **Framework de comparación** de modelos
- **Mejora de 10-20%** en RMSE vs XGBoost

### Académico

- **Comparación rigurosa** de algoritmos de gradient boosting
- **Análisis de ensemble methods** en series temporales
- **Insights sobre el dataset** (qué features importan, qué modelos funcionan mejor)
- **Documentación de alta calidad** para el proyecto

### Proyecto

- **Acercamiento al target** CUBIST (aunque probablemente no lo alcancemos)
- **Base sólida** para feature engineering adicional
- **Modelo candidato** para deployment
- **Aprendizajes** para Sprint 2

---

**Estimación Total**: 5-6 días  
**Prioridad**: ALTA  
**Dependencias**: US-012 (Preprocessing), US-013 (XGBoost)  
**Bloqueantes**: Ninguno  

---

*Documento de planeación creado por MLOps Team - Proyecto Atreides*  
*Fecha: 2025-10-30*  
*Versión: 1.0*
