# US-013: Baseline XGBoost Model - Plan de Implementación

**Estado**: PLANNING
**Fecha de inicio planeada**: 2025-10-27
**Estimación**: 6-8 horas
**Responsable**: ML Engineer (Julian) + MLOps Engineer (Arthur)

---

## Resumen Ejecutivo

Implementación de un modelo baseline XGBoost con hyperparameter tuning usando Optuna, evaluación con cross-validation 5-fold, logging completo en MLflow, y exportación de feature importance. Este modelo servirá como punto de referencia para comparar modelos más avanzados y cumplir con la meta de superar el benchmark CUBIST (RMSE: 0.2410) en al menos 15%, logrando RMSE < 0.205.

---

## User Story

**Como** ML Engineer
**Quiero** entrenar un modelo baseline robusto con XGBoost
**Para que** tengamos un punto de comparación interno sólido y reproducible

---

## Criterios de Aceptación

### 1. Modelo XGBoost Entrenado ✓
- Pipeline completo de sklearn con preprocessing integrado
- Entrenamiento en datos preprocesados de US-012
- Modelo optimizado para regresión (objetivo: Usage_kWh)
- Pipeline serializado en formato joblib

### 2. Hyperparameter Tuning con Optuna ✓
- 100 trials de optimización bayesiana
- Search space definido basado en mejores prácticas XGBoost
- Optimización de RMSE en validation set
- Pruning automático de trials poco prometedores
- Historial de trials exportado

### 3. Cross-Validation 5-Fold ✓
- Validación cruzada estratificada en training set
- Métricas: RMSE, MAE, R², MAPE
- Estadísticas: mean, std de cada métrica
- Análisis de varianza entre folds

### 4. Métricas Loggeadas en MLflow ✓
- Parameters: todos los hiperparámetros del modelo
- Metrics: RMSE, MAE, R², MAPE (train/val/test)
- Artifacts: modelo, feature importance, plots
- Tags: experiment_type, model_version, dataset_version
- System metrics: training time, memory usage

### 5. Feature Importance Exportado ✓
- Feature importance de 3 tipos:
  - Gain (information gain)
  - Weight (frequency)
  - Cover (coverage)
- Visualizaciones (bar plots horizontales)
- Exportación a CSV y PNG
- Top 10 features destacados

### 6. Modelo Serializado ✓
- Path: `models/baselines/xgboost_v1.pkl`
- Formato: joblib (compatible con sklearn)
- Metadata: fecha, versión, hiperparámetros
- Checksum MD5 para validación

### 7. Notebook de Análisis ✓
- Notebook: `notebooks/exploratory/08_xgboost_model.ipynb`
- Secciones: carga de datos, EDA rápido, entrenamiento, evaluación, feature importance
- Texto explicativo en español
- Código modular usando funciones de src/utils

---

## Arquitectura Técnica

### Módulos a Crear

#### 1. `src/models/xgboost_trainer.py` (principal)

**Propósito**: Funciones para entrenar y optimizar modelos XGBoost

**Funciones principales**:

```python
def create_xgboost_pipeline(
    model_params: dict = None,
    use_preprocessing: bool = False
) -> Pipeline:
    """
    Create sklearn pipeline with XGBoost model.

    Note: preprocessing is already done in US-012, so by default
    we only include the model. Set use_preprocessing=True only
    for demo/testing purposes.

    Parameters
    ----------
    model_params : dict, optional
        XGBoost hyperparameters
    use_preprocessing : bool, default=False
        Whether to include preprocessing steps (usually False)

    Returns
    -------
    Pipeline
        sklearn pipeline with XGBoost regressor
    """

def train_xgboost_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict,
    cv_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Train XGBoost with cross-validation.

    Returns dictionary with:
    - model: trained model
    - cv_scores: dict with mean/std for each metric
    - fold_scores: list of scores per fold
    """

def optimize_xgboost_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    mlflow_tracking: bool = True
) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Returns:
    - study: optuna study object
    - best_params: best hyperparameters found
    - best_model: model trained with best params
    - trials_df: dataframe with all trials
    """

def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str = "test"
) -> dict:
    """
    Evaluate model on test set.

    Returns dictionary with RMSE, MAE, R2, MAPE.
    """
```

#### 2. `src/utils/mlflow_utils.py` (nuevo)

**Propósito**: Funciones helper para MLflow tracking

**Funciones principales**:

```python
def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5000"
) -> str:
    """
    Setup MLflow experiment and return experiment_id.

    Uses docker-compose MLflow server at http://localhost:5000
    """

def log_model_params(params: dict) -> None:
    """Log model parameters to MLflow."""

def log_model_metrics(
    metrics: dict,
    prefix: str = ""
) -> None:
    """Log metrics to MLflow with optional prefix."""

def log_cv_results(
    cv_scores: dict,
    fold_scores: list
) -> None:
    """Log cross-validation results to MLflow."""

def log_feature_importance(
    model,
    feature_names: list,
    importance_type: str = "gain"
) -> None:
    """Extract and log feature importance."""

def save_and_log_model(
    model: Pipeline,
    model_path: Path,
    artifact_name: str = "model"
) -> None:
    """Save model to disk and log to MLflow."""
```

#### 3. `src/utils/model_evaluation.py` (nuevo)

**Propósito**: Funciones para evaluación de modelos

**Funciones principales**:

```python
def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calculate comprehensive regression metrics.

    Returns:
    - rmse
    - mae
    - r2
    - mape
    - max_error
    """

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Path = None
) -> Figure:
    """Create scatter plot of predictions vs actual values."""

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None
) -> Figure:
    """Create residual plots (histogram + scatter)."""

def plot_feature_importance(
    importance_dict: dict,
    top_n: int = 10,
    save_path: Path = None
) -> Figure:
    """Create horizontal bar plot of feature importance."""

def create_evaluation_report(
    metrics: dict,
    cv_scores: dict,
    feature_importance: dict,
    output_path: Path
) -> None:
    """Generate markdown report with evaluation results."""
```

#### 4. `src/models/train_xgboost.py` (ejecutable)

**Propósito**: Script ejecutable para entrenar XGBoost baseline

**Pipeline de ejecución**:

```python
def main():
    """Main training pipeline."""
    # Step 1: Setup logging and directories
    # Step 2: Setup MLflow experiment (connect to docker container)
    # Step 3: Load preprocessed data (from US-012)
    # Step 4: Define Optuna search space
    # Step 5: Run hyperparameter optimization (100 trials)
    # Step 6: Train final model with best params
    # Step 7: Cross-validation evaluation
    # Step 8: Test set evaluation
    # Step 9: Extract feature importance
    # Step 10: Save model and artifacts
    # Step 11: Log everything to MLflow
    # Step 12: Generate reports
```

---

## Configuración de MLflow

### Setup con Docker Compose

```bash
# Levantar MLflow en segundo plano (ya configurado en docker-compose.yml)
docker-compose up mlflow -d

# Verificar que MLflow está corriendo
curl http://localhost:5000/health

# Ver logs de MLflow
docker-compose logs -f mlflow

# Acceder a UI de MLflow
# http://localhost:5000
```

### Configuración en Código

```python
import mlflow

# MLflow tracking URI (docker-compose service)
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Setup experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("steel_energy_xgboost_baseline")

# Start run
with mlflow.start_run(run_name="xgboost_v1"):
    # Log params, metrics, artifacts
    pass
```

### Storage de Artifacts

Los artifacts se guardarán en:
- **Local**: `./mlflow/artifacts/` (volumen de docker-compose)
- **MLflow DB**: `./mlflow/mlflow.db` (SQLite backend)

---

## Especificación de Hiperparámetros

### Search Space para Optuna

```python
SEARCH_SPACE = {
    # Tree structure
    'max_depth': (3, 10),              # int, prevent overfitting
    'min_child_weight': (1, 10),       # int, minimum samples per leaf
    'gamma': (0.0, 5.0),               # float, regularization

    # Learning rate
    'learning_rate': (0.01, 0.3),      # float, step size
    'n_estimators': (100, 1000),       # int, number of trees

    # Sampling
    'subsample': (0.5, 1.0),           # float, row sampling
    'colsample_bytree': (0.5, 1.0),    # float, column sampling per tree
    'colsample_bylevel': (0.5, 1.0),   # float, column sampling per level

    # Regularization
    'reg_alpha': (0.0, 10.0),          # float, L1 regularization
    'reg_lambda': (0.0, 10.0),         # float, L2 regularization
}
```

### Parámetros Fijos

```python
FIXED_PARAMS = {
    'objective': 'reg:squarederror',   # regression with squared loss
    'eval_metric': 'rmse',             # evaluation metric
    'tree_method': 'hist',             # fast histogram-based algorithm
    'random_state': 42,                # reproducibility
    'n_jobs': -1,                      # use all cores
    'verbosity': 0,                    # silent mode
}
```

---

## Estructura de Archivos a Crear

### Código Fuente

```
src/
├── models/
│   ├── xgboost_trainer.py         # Funciones de entrenamiento (500+ líneas)
│   └── train_xgboost.py           # Script ejecutable (600+ líneas)
├── utils/
│   ├── mlflow_utils.py            # Utilidades MLflow (400+ líneas)
│   └── model_evaluation.py        # Evaluación y plots (500+ líneas)
```

### Tests

```
tests/
└── unit/
    ├── test_xgboost_trainer.py    # Tests para trainer (20+ tests)
    ├── test_mlflow_utils.py       # Tests para MLflow (15+ tests)
    └── test_model_evaluation.py   # Tests para evaluación (15+ tests)
```

### Modelos y Artifacts

```
models/
└── baselines/
    ├── xgboost_v1.pkl             # Modelo principal
    ├── xgboost_v1_metadata.json   # Metadata del modelo
    └── preprocessing_pipeline.pkl  # Pipeline de preprocessing (referencia)

reports/
├── figures/
│   ├── xgboost_feature_importance_gain.png
│   ├── xgboost_feature_importance_weight.png
│   ├── xgboost_predictions_vs_actual.png
│   └── xgboost_residuals.png
├── metrics/
│   ├── xgboost_cv_results.csv
│   ├── xgboost_test_metrics.json
│   └── optuna_trials_history.csv
└── xgboost_evaluation_report.md

mlflow/                             # MLflow storage (docker volume)
├── mlflow.db                       # SQLite backend
└── artifacts/                      # Artifacts storage
    └── {experiment_id}/
        └── {run_id}/
            ├── model/
            ├── feature_importance/
            └── evaluation/
```

### Notebooks

```
notebooks/
└── exploratory/
    └── 08_xgboost_model.ipynb     # Notebook de análisis
```

### Documentación

```
docs/
└── us-resolved/
    └── us-013.md                  # Documento de completion
```

---

## MLflow Tracking Schema

### Run Structure

```yaml
Experiment: "steel_energy_xgboost_baseline"

Run:
  name: "xgboost_baseline_v1"

  tags:
    experiment_type: "baseline"
    model_type: "xgboost"
    model_version: "v1"
    dataset_version: "steel_preprocessed_v1"
    optimization: "optuna"

  params:
    # XGBoost params (10-15 params)
    max_depth: 6
    learning_rate: 0.05
    n_estimators: 500
    # ... all hyperparameters

  metrics:
    # Training metrics
    train_rmse: 0.185
    train_mae: 0.042
    train_r2: 0.92
    train_mape: 8.5

    # Validation metrics
    val_rmse: 0.198
    val_mae: 0.045
    val_r2: 0.90
    val_mape: 9.2

    # Test metrics
    test_rmse: 0.195
    test_mae: 0.044
    test_r2: 0.91
    test_mape: 9.0

    # CV metrics
    cv_rmse_mean: 0.196
    cv_rmse_std: 0.008
    cv_mae_mean: 0.044
    cv_mae_std: 0.002
    cv_r2_mean: 0.91
    cv_r2_std: 0.01

    # Training info
    training_time_seconds: 245.3
    n_trials: 100
    best_trial_number: 67

  artifacts:
    model/
      - model.pkl
      - conda.yaml
      - requirements.txt

    feature_importance/
      - gain.csv
      - weight.csv
      - cover.csv
      - importance_gain.png
      - importance_weight.png

    evaluation/
      - predictions_vs_actual.png
      - residuals.png
      - cv_results.csv
      - test_metrics.json

    optuna/
      - trials_history.csv
      - optimization_history.png
      - parallel_coordinate.png
```

---

## Métricas de Éxito

### Requisitos Mínimos (Must Have)

| Métrica | Target | Justificación |
|---------|--------|---------------|
| Test RMSE | < 0.205 | Superar CUBIST en 15% (0.2410 * 0.85 = 0.205) |
| Test MAE | < 0.046 | Consistente con RMSE objetivo |
| Test R² | > 0.88 | Buen poder explicativo |
| CV RMSE std | < 0.02 | Modelo estable entre folds |
| Training time | < 10 min | Eficiencia computacional |
| MLflow logs | 100% | Reproducibilidad total |

### Requisitos Deseables (Should Have)

| Métrica | Target | Justificación |
|---------|--------|---------------|
| Test RMSE | < 0.19 | Margen de seguridad sobre objetivo |
| Test MAPE | < 10% | Error porcentual aceptable |
| Feature coverage | Top 5 explican >70% | Concentración de importancia |
| Optuna efficiency | Best in first 50 trials | Convergencia rápida |

---

## Pipeline de Entrenamiento (10 Pasos)

### Paso 0: Setup de MLflow (1 min)
```bash
# Levantar MLflow con docker-compose
docker-compose up mlflow -d

# Verificar que está funcionando
curl http://localhost:5000/health
```

### Paso 1: Setup y Configuración (5 min)
```python
# Setup logging
# Create directories (models/baselines/, reports/figures/, reports/metrics/)
# Setup MLflow experiment (connect to http://localhost:5000)
# Load configuration
```

### Paso 2: Carga de Datos (1 min)
```python
# Load steel_preprocessed_train.parquet
# Load steel_preprocessed_val.parquet
# Load steel_preprocessed_test.parquet
# Validate schemas and sizes
```

### Paso 3: Preparación de Features (2 min)
```python
# Separate X and y for train/val/test
# Validate feature names
# Check for nulls and infinities
# Convert Polars DataFrame to numpy arrays
```

### Paso 4: Optuna Optimization (200-300 min)
```python
# Define objective function
# Create Optuna study (minimize RMSE)
# Run 100 trials with pruning
# Extract best parameters
# Save trials history
# Log to MLflow in real-time
```

### Paso 5: Cross-Validation (15-20 min)
```python
# Create 5-fold stratified splits
# Train model on each fold
# Calculate metrics per fold
# Aggregate results (mean, std)
```

### Paso 6: Entrenamiento Final (3-5 min)
```python
# Train on full training set with best params
# Validate on validation set
# Early stopping based on val_rmse
```

### Paso 7: Evaluación en Test (1 min)
```python
# Predict on test set
# Calculate all metrics (RMSE, MAE, R², MAPE)
# Generate prediction plots
# Generate residual plots
```

### Paso 8: Feature Importance (2 min)
```python
# Extract importance (gain, weight, cover)
# Create bar plots for each type
# Export to CSV
# Identify top 10 features
```

### Paso 9: Persistencia de Modelo (2 min)
```python
# Save model with joblib
# Generate metadata JSON
# Calculate MD5 checksum
# Verify model loads correctly
```

### Paso 10: MLflow Logging y Reports (5 min)
```python
# Log all parameters to MLflow
# Log all metrics (train/val/test/cv)
# Log all artifacts (model, plots, CSVs)
# Add tags
# Generate markdown report
```

**Tiempo Total Estimado**: 240-350 minutos (4-6 horas) + setup MLflow

---

## Decisiones de Diseño

### 1. ¿Por qué XGBoost sobre LightGBM para baseline?

**Decisión**: XGBoost
**Razones**:
- Mayor madurez y estabilidad
- Mejor documentación y comunidad
- Integración nativa con MLflow
- Performance comparable para este tamaño de dataset (35k rows)
- Referencia estándar en la industria

### 2. ¿Por qué 100 trials de Optuna?

**Decisión**: 100 trials
**Razones**:
- Balance entre exhaustividad y tiempo de cómputo
- Estudios muestran convergencia alrededor de 50-100 trials
- Permite pruning de trials poco prometedores
- ~3-4 horas de optimización (aceptable para baseline)

### 3. ¿Por qué CV 5-fold en lugar de 10-fold?

**Decisión**: 5-fold
**Razones**:
- Dataset mediano (24k training samples)
- Cada fold tiene ~4,800 samples (suficiente)
- Menor tiempo de cómputo vs 10-fold
- Práctica estándar en competiciones Kaggle
- Menos varianza que 3-fold

### 4. ¿Por qué optimizar RMSE en lugar de MAE?

**Decisión**: RMSE
**Razones**:
- RMSE penaliza más los errores grandes
- Más sensible a outliers (importante en energía)
- Benchmark CUBIST usa RMSE (comparabilidad)
- Métrica estándar en regresión

### 5. ¿Por qué joblib en lugar de pickle?

**Decisión**: joblib
**Razones**:
- Mejor compresión para objetos grandes
- Más rápido para arrays de numpy
- Estándar en sklearn y XGBoost
- Compatible con MLflow

### 6. ¿Por qué NO incluir preprocessing en el pipeline del modelo?

**Decisión**: Modelo sin preprocessing (ya hecho en US-012)
**Razones**:
- US-012 ya generó datos preprocesados (steel_preprocessed_*.parquet)
- Evita duplicación de preprocesamiento
- Más eficiente computacionalmente
- Reutiliza trabajo de US-012 (preprocessing_pipeline.pkl)
- Pipeline del modelo es más simple y especializado

### 7. ¿Por qué usar MLflow en Docker en lugar de local?

**Decisión**: MLflow con docker-compose
**Razones**:
- Ya está configurado en el proyecto
- Consistencia entre todos los miembros del equipo
- Fácil de levantar/bajar (docker-compose up/down)
- Persistencia de datos en volúmenes
- No contamina sistema local

---

## Testing Strategy

### Test Coverage Target: >80%

#### Unit Tests (50 tests estimados)

**test_xgboost_trainer.py** (20 tests):
```python
class TestCreatePipeline:
    def test_pipeline_creation_default_params()
    def test_pipeline_creation_custom_params()
    def test_pipeline_structure()
    def test_pipeline_fit_transform()

class TestTrainWithCV:
    def test_cv_basic_functionality()
    def test_cv_correct_fold_count()
    def test_cv_metrics_calculation()
    def test_cv_reproducibility()

class TestOptunaOptimization:
    def test_optimization_basic()
    def test_optimization_with_pruning()
    def test_best_params_extraction()
    def test_trials_dataframe_export()

class TestEvaluation:
    def test_evaluate_model_metrics()
    def test_evaluate_model_all_datasets()
```

**test_mlflow_utils.py** (15 tests):
```python
class TestMLflowSetup:
    def test_experiment_creation()
    def test_experiment_reuse()
    def test_tracking_uri_connection()

class TestMLflowLogging:
    def test_log_params()
    def test_log_metrics()
    def test_log_cv_results()
    def test_log_feature_importance()
    def test_save_and_log_model()
```

**test_model_evaluation.py** (15 tests):
```python
class TestMetricsCalculation:
    def test_regression_metrics_basic()
    def test_metrics_with_perfect_predictions()
    def test_metrics_with_zero_variance()

class TestPlotting:
    def test_predictions_vs_actual_plot()
    def test_residuals_plot()
    def test_feature_importance_plot()

class TestReporting:
    def test_evaluation_report_generation()
    def test_report_markdown_format()
```

---

## Ejemplo de Uso

### Uso Básico (Script)

```bash
# 1. Levantar MLflow
docker-compose up mlflow -d

# 2. Entrenar modelo baseline
poetry run python src/models/train_xgboost.py

# 3. Ver resultados en MLflow UI
# http://localhost:5000

# 4. Bajar MLflow cuando termines
docker-compose down mlflow
```

### Uso Programático

```python
from src.models.xgboost_trainer import (
    optimize_xgboost_with_optuna,
    train_xgboost_with_cv,
    evaluate_model
)
from src.utils.mlflow_utils import setup_mlflow_experiment
import polars as pl

# Setup MLflow (docker-compose)
setup_mlflow_experiment(
    "xgboost_baseline",
    tracking_uri="http://localhost:5000"
)

# Load preprocessed data (from US-012)
df_train = pl.read_parquet("data/processed/steel_preprocessed_train.parquet")
df_val = pl.read_parquet("data/processed/steel_preprocessed_val.parquet")

# Note: Data is already preprocessed, just convert to numpy
X_train = df_train.drop("Usage_kWh").to_numpy()
y_train = df_train["Usage_kWh"].to_numpy()
X_val = df_val.drop("Usage_kWh").to_numpy()
y_val = df_val["Usage_kWh"].to_numpy()

# Optimize hyperparameters
results = optimize_xgboost_with_optuna(
    X_train, y_train,
    X_val, y_val,
    n_trials=100,
    cv_folds=5
)

# Train final model with CV
cv_results = train_xgboost_with_cv(
    X_train, y_train,
    model_params=results['best_params'],
    cv_folds=5
)

# Evaluate
test_metrics = evaluate_model(
    cv_results['model'],
    X_test, y_test,
    dataset_name="test"
)

print(f"Test RMSE: {test_metrics['rmse']:.4f}")
```

---

## Riesgos y Mitigaciones

### Riesgo 1: Tiempo de Optimización Excesivo

**Probabilidad**: Media
**Impacto**: Medio
**Mitigación**:
- Usar Optuna pruning (MedianPruner)
- Reducir n_trials si excede 4 horas
- Paralelizar trials cuando sea posible

### Riesgo 2: Overfitting en Hyperparameter Tuning

**Probabilidad**: Media
**Impacto**: Alto
**Mitigación**:
- Usar validation set separado para Optuna
- Cross-validation en entrenamiento final
- Early stopping con patience
- Regularización (L1/L2)

### Riesgo 3: No Alcanzar Target RMSE < 0.205

**Probabilidad**: Baja
**Impacto**: Alto
**Mitigación**:
- Feature engineering adicional si es necesario
- Ensemble con otros modelos posteriormente
- Análisis de errores para entender limitaciones

### Riesgo 4: MLflow Docker Container Crashes

**Probabilidad**: Baja
**Impacto**: Medio
**Mitigación**:
- Guardar artefactos localmente como backup
- Verificar MLflow health antes de optimización
- Implementar retry logic en logging
- Documentar cómo reiniciar container

---

## Dependencias y Prerequisitos

### MLflow Setup

```bash
# Levantar MLflow con docker-compose
docker-compose up mlflow -d

# Verificar status
docker-compose ps mlflow

# Ver logs si hay problemas
docker-compose logs mlflow

# Acceder a UI
# http://localhost:5000
```

### Datos Requeridos (de US-012)
- `data/processed/steel_preprocessed_train.parquet` (24,437 rows × 9 cols)
- `data/processed/steel_preprocessed_val.parquet` (5,236 rows × 9 cols)
- `data/processed/steel_preprocessed_test.parquet` (5,237 rows × 9 cols)

### Librerías Python Requeridas

Ya están en `pyproject.toml`, pero para referencia:

```toml
[tool.poetry.dependencies]
xgboost = "^2.0.0"           # XGBoost library
optuna = "^3.4.0"            # Hyperparameter optimization
mlflow = "^2.8.0"            # Experiment tracking
scikit-learn = "^1.3.0"      # ML utilities and pipeline
polars = "^0.19.0"           # Data manipulation
numpy = "^1.25.0"            # Numerical operations
matplotlib = "^3.8.0"        # Plotting
seaborn = "^0.13.0"          # Statistical plots
joblib = "^1.3.0"            # Model serialization
```

### Prerequisitos de Sistema
- Python 3.11+
- Docker + docker-compose (para MLflow)
- 8GB RAM mínimo (16GB recomendado)
- 2GB espacio en disco
- CPU multi-core (para paralelización)

---

## Checklist de Completion

### Infraestructura
- [ ] Docker-compose MLflow corriendo (`docker-compose up mlflow -d`)
- [ ] MLflow UI accesible en http://localhost:5000
- [ ] Directorios creados (models/baselines/, reports/)

### Código
- [ ] `src/models/xgboost_trainer.py` implementado con 4+ funciones
- [ ] `src/models/train_xgboost.py` script ejecutable funcional
- [ ] `src/utils/mlflow_utils.py` con 6+ funciones
- [ ] `src/utils/model_evaluation.py` con 5+ funciones
- [ ] Docstrings estilo Google en todas las funciones
- [ ] Type hints completos
- [ ] Sin warnings de Ruff y Black compliant

### Testing
- [ ] 50+ tests unitarios implementados
- [ ] Coverage >80% en módulos principales
- [ ] Todos los tests pasan (100% success rate)
- [ ] Edge cases cubiertos

### Modelo
- [ ] Modelo entrenado con Optuna (100 trials)
- [ ] Cross-validation 5-fold ejecutado
- [ ] Test RMSE < 0.205 (objetivo alcanzado)
- [ ] Modelo serializado en `models/baselines/xgboost_v1.pkl`
- [ ] Metadata JSON generado

### MLflow
- [ ] Experiment "steel_energy_xgboost_baseline" creado
- [ ] Run con todos los parámetros loggeados
- [ ] Métricas train/val/test/cv loggeadas
- [ ] Artefactos subidos (modelo, plots, CSVs)
- [ ] Tags apropiados asignados
- [ ] Artifacts visibles en MLflow UI

### Feature Importance
- [ ] 3 tipos de importance extraídos (gain, weight, cover)
- [ ] Bar plots generados y guardados
- [ ] CSVs exportados
- [ ] Top 10 features identificados

### Visualizaciones
- [ ] Predictions vs Actual plot (PNG, 300 DPI)
- [ ] Residuals plot (histogram + scatter)
- [ ] Feature importance plots (3 tipos)
- [ ] Todas guardadas en `reports/figures/`

### Reportes
- [ ] `reports/xgboost_evaluation_report.md` generado
- [ ] `reports/metrics/xgboost_test_metrics.json` exportado
- [ ] `reports/metrics/xgboost_cv_results.csv` exportado
- [ ] `reports/metrics/optuna_trials_history.csv` exportado

### Notebook
- [ ] `notebooks/exploratory/08_xgboost_model.ipynb` creado
- [ ] Secciones: setup, carga, EDA, entrenamiento, evaluación, feature importance
- [ ] Texto explicativo en español
- [ ] Código modular (usa funciones de src/)
- [ ] Visualizaciones incluidas

### Documentación
- [ ] `docs/us-resolved/us-013.md` completado
- [ ] README actualizado con sección de modelos
- [ ] Ejemplos de uso documentados
- [ ] Decisiones de diseño explicadas

### Versionado
- [ ] Modelo versionado con DVC
- [ ] `.dvc` file commiteado
- [ ] Git tag creado: `model-xgboost-v1`
- [ ] Commit con conventional commits format

### Validación Final
- [ ] Pipeline end-to-end ejecutable
- [ ] Modelo carga correctamente desde disco
- [ ] Reproducibilidad verificada (mismo random_state)
- [ ] Métricas consistentes entre runs

---

## Timeline Detallado

### Día 1 (4-5 horas)
- **Hora 0.5**: Setup MLflow con docker-compose
- **Hora 1**: Crear módulos base (`xgboost_trainer.py`, `mlflow_utils.py`)
- **Hora 2**: Implementar funciones de entrenamiento y CV
- **Hora 3**: Implementar Optuna optimization
- **Hora 4**: Tests unitarios para funciones principales
- **Hora 5**: Primera ejecución de prueba (debugging)

### Día 2 (3-4 horas)
- **Hora 1**: Ejecutar optimización completa (100 trials)
- **Hora 2**: Implementar evaluación y feature importance
- **Hora 3**: Crear visualizaciones y reportes
- **Hora 4**: Crear notebook de análisis
- **Hora 5**: Testing final y documentación

---

## Métricas de Calidad del Código

### Targets

| Métrica | Target | Herramienta |
|---------|--------|-------------|
| Test Coverage | >80% | pytest-cov |
| Cyclomatic Complexity | <10 per function | radon |
| Maintainability Index | >60 | radon |
| Type Coverage | 100% | mypy |
| Linting | 0 errors | ruff |
| Formatting | compliant | black |
| Docstrings | 100% | interrogate |

---

## Referencias

### Documentación Interna
- [US-011: Temporal Features](../us-resolved/us-011.md)
- [US-012: Preprocessing](../us-resolved/us-012.md)
- [AGENTS.md](../../AGENTS.md) - Guía de buenas prácticas
- [README.md](../../README.md) - Setup y configuración
- [plan_context.md](../../context/plan_context.md) - Contexto del proyecto

### Docker y MLflow
- [docker-compose.yml](../../docker-compose.yml) - Configuración de servicios
- MLflow UI: http://localhost:5000
- MLflow storage: `./mlflow/` (volumen de docker)

### Papers y Recursos
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/abs/1603.02754)
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Benchmarks
- **CUBIST**: RMSE = 0.2410 (baseline a superar)
- **Target**: RMSE < 0.205 (15% improvement)

---

## Preguntas Abiertas para Discusión

1. **¿Deberíamos incluir SHAP values en esta US?**
   - Pros: Mejor interpretabilidad
   - Contras: Aumenta complejidad y tiempo
   - **Propuesta**: Dejar SHAP para US posterior (US-014 o US-015)

2. **¿Qué hacer si no alcanzamos RMSE < 0.205?**
   - Opción A: Más feature engineering
   - Opción B: Ensemble inmediato
   - Opción C: Ajustar target realista
   - **Propuesta**: Analizar errores y decidir basado en gap

3. **¿Cuánto espacio en MLflow para 100 trials?**
   - Estimación: ~500MB artifacts + ~50MB metadata
   - Storage: Volumen de docker `./mlflow/`
   - **Acción**: Monitorear espacio disponible

4. **¿Deberíamos implementar early stopping en Optuna?**
   - **Propuesta**: Sí, usar MedianPruner con patience=5

---

## Aprobación y Sign-off

**Preparado por**: Arthur (MLOps Engineer) + AI Assistant
**Fecha**: 2025-10-27
**Versión**: 1.1 (actualizado con MLflow docker-compose)

**Revisores**:
- [ ] Julian (ML Engineer) - Revisión técnica de modelo
- [ ] Dante (Scrum Master) - Validación de alcance y tiempo
- [ ] Erick (Data Scientist) - Revisión de métricas y evaluación

**Aprobación Final**: ⏳ Pendiente

---

## Próximos Pasos Después de US-013

### US-014: Model Comparison & Ensemble (Sugerido)
- Entrenar LightGBM baseline
- Entrenar Random Forest baseline
- Comparar 3 modelos (XGBoost, LightGBM, RF)
- Crear ensemble si mejora performance

### US-015: Model Explainability (Sugerido)
- SHAP values para interpretabilidad
- Partial Dependence Plots
- Individual Conditional Expectation (ICE) plots
- Global y local explanations

### US-016: Model Deployment Preparation (Sugerido)
- Optimizar modelo para inference
- Crear API endpoint para predicción
- Testing de latencia y throughput
- Documentación de API

---

**Estado**: LISTO PARA REVISIÓN Y APROBACIÓN
**Próxima Acción**: Revisión por equipo → Aprobación → Inicio de implementación

**Comando para comenzar**:
```bash
# 1. Levantar MLflow
docker-compose up mlflow -d

# 2. Verificar MLflow UI
# http://localhost:5000

# 3. Comenzar implementación
# (crear archivos según plan)
```
