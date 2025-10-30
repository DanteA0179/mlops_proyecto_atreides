# Guía para Agentes de IA - Energy Optimization Copilot

Este archivo contiene instrucciones y buenas prácticas para agentes de IA que trabajen en este proyecto.

---

## 🎯 Contexto del Proyecto

**Nombre**: Energy Optimization Copilot  
**Tipo**: Proyecto MLOps - Sistema de Optimización Energética con IA  
**Stack**: Python 3.11+, Poetry, DVC, MLflow, DuckDB, FastAPI, Scikit-Learn  
**Objetivo**: Predicción y análisis de consumo energético en la industria siderúrgica

---

## 📋 Buenas Prácticas Establecidas

### 1. Idioma y Documentación

#### ✅ Notebooks
- **Texto explicativo**: SIEMPRE en español
- **Código**: SIEMPRE en inglés
- **Comentarios en código**: En inglés
- **Markdown cells**: En español

**Ejemplo correcto**:
```python
# Notebook cell (Markdown)
## 1. Análisis Exploratorio de Datos

Este análisis explora los patrones de consumo energético...

# Notebook cell (Code)
# Load data and perform initial exploration
df = pl.read_parquet("data/processed/steel_cleaned.parquet")
summary_stats = df.describe()
```

#### ✅ Código Python
- **Nombres de variables**: En inglés
- **Nombres de funciones**: En inglés
- **Docstrings**: En inglés (estilo Google)
- **Comentarios**: En inglés

#### ✅ Documentación
- **README.md**: En español
- **Documentación técnica**: En español
- **Docstrings en código**: En inglés
- **Comentarios inline**: En inglés

---

### 2. Estructura de Código

#### ✅ Funciones Reutilizables

**SIEMPRE** crear funciones reutilizables en `src/utils/` en lugar de código duplicado en notebooks.

**❌ Incorrecto** (código en notebook):
```python
# En notebook
conn = duckdb.connect("data/steel.duckdb")
df = conn.execute("SELECT * FROM steel_cleaned").pl()
conn.close()
```

**✅ Correcto** (usar función de utils):
```python
# En notebook
from src.utils.duckdb_utils import quick_query
df = quick_query("SELECT * FROM steel_cleaned")
```

#### ✅ Organización de Utilidades

```
src/utils/
├── duckdb_utils.py          # Funciones para DuckDB
├── data_cleaning.py         # Limpieza de datos
├── data_quality.py          # Análisis de calidad
├── outlier_detection.py     # Detección de outliers
├── visualization.py         # Visualizaciones
└── secrets.py               # Manejo de secretos
```

**Regla**: Si una función se usa más de una vez, debe estar en `src/utils/`.

---

### 3. Programación Orientada a Objetos

#### ✅ Transformers de Scikit-Learn

Para feature engineering, SIEMPRE usar clases que hereden de `BaseEstimator` y `TransformerMixin`:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates temporal features from NSM column.
    
    Parameters
    ----------
    nsm_column : str, default='NSM'
        Name of the column containing seconds from midnight
    """
    
    def __init__(self, nsm_column='NSM'):
        self.nsm_column = nsm_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['hour'] = (X[self.nsm_column] // 3600).astype(int)
        return X
```

**Beneficios**:
- ✅ Reutilizable en pipelines de sklearn
- ✅ Compatible con `fit()` y `transform()`
- ✅ Fácil de testear

---

### 4. Pipelines de Scikit-Learn

#### ✅ SIEMPRE usar pipelines

**❌ Incorrecto**:
```python
# Código suelto
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = XGBRegressor()
model.fit(X_scaled, y_train)
```

**✅ Correcto**:
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor())
])
pipeline.fit(X_train, y_train)
```

---

### 5. Manejo de Nombres de Columnas

#### ✅ Columnas con Caracteres Especiales

El dataset tiene columnas con caracteres especiales:
- `CO2(tCO2)` - Paréntesis
- `Lagging_Current_Reactive.Power_kVarh` - Punto

**Las funciones en `src/utils/duckdb_utils.py` manejan esto automáticamente**.

**✅ Correcto**:
```python
# Las funciones manejan automáticamente los caracteres especiales
stats = get_stats_by_column('steel_cleaned', 'Load_Type', 'CO2(tCO2)')
corr = get_correlation('steel_cleaned', 'Usage_kWh', 'CO2(tCO2)')
```

**Para queries SQL personalizados**, usar comillas dobles:
```python
df = quick_query('''
    SELECT "CO2(tCO2)" as CO2
    FROM steel_cleaned
''')
```

---

### 6. Testing

#### ✅ Estructura de Tests

```
tests/
├── unit/              # Tests unitarios
├── integration/       # Tests de integración
└── e2e/              # Tests end-to-end
```

#### ✅ Convenciones de Tests

- **Nombres**: `test_*.py`
- **Clases**: `Test*`
- **Funciones**: `test_*`
- **Coverage mínimo**: >70%

**Ejemplo**:
```python
import pytest

class TestDuckDBUtils:
    """Tests para funciones de DuckDB."""
    
    def test_quick_query_basic(self):
        """Test query básico."""
        df = quick_query("SELECT * FROM steel_cleaned LIMIT 5")
        assert len(df) == 5
```

---

### 7. Versionado de Datos y Modelos

#### ✅ Usar DVC

**NUNCA** commitear archivos grandes a Git. Usar DVC:

```bash
# ✅ Correcto
dvc add data/processed/steel_cleaned.parquet
git add data/processed/steel_cleaned.parquet.dvc
git commit -m "data: add cleaned dataset"

# ❌ Incorrecto
git add data/processed/steel_cleaned.parquet
```

#### ✅ Archivos que van con DVC

- ✅ Datasets (CSV, Parquet, etc.)
- ✅ Modelos entrenados (pkl, pth, h5)
- ✅ Archivos >1MB

#### ✅ Archivos que van con Git

- ✅ Código fuente
- ✅ Configs (<1KB)
- ✅ Documentación
- ✅ Tests

---

### 8. MLflow Experiment Tracking

#### ✅ SIEMPRE loggear experimentos

```python
import mlflow

with mlflow.start_run(run_name="xgboost_baseline"):
    # Log parameters
    mlflow.log_params(model.get_params())
    
    # Train
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

---

### 9. Convenciones de Código

#### ✅ Formateo

- **Formatter**: Black (line-length=100)
- **Linter**: Ruff
- **Type checker**: MyPy (opcional)

```bash
# Antes de commitear
poetry run black .
poetry run ruff check .
```

#### ✅ Docstrings

**Estilo Google** para todas las funciones:

```python
def get_stats_by_column(
    table_name: str,
    group_by_column: str,
    agg_column: str
) -> pl.DataFrame:
    """
    Obtiene estadísticas agregadas agrupadas por una columna.
    
    Parameters
    ----------
    table_name : str
        Nombre de la tabla
    group_by_column : str
        Columna para agrupar
    agg_column : str
        Columna para agregar
        
    Returns
    -------
    pl.DataFrame
        DataFrame con estadísticas (count, avg, min, max, std)
        
    Examples
    --------
    >>> stats = get_stats_by_column('steel_cleaned', 'Load_Type', 'Usage_kWh')
    >>> print(stats)
    """
```

#### ✅ Type Hints

SIEMPRE usar type hints:

```python
# ✅ Correcto
def process_data(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    pass

# ❌ Incorrecto
def process_data(df, threshold=0.5):
    pass
```

---

### 10. Estructura de Notebooks

#### ✅ Orden Estándar

```markdown
# 1. Título y Descripción (en español)

## 2. Imports
import sys
sys.path.append('../..')
from src.utils.duckdb_utils import setup_database

## 3. Configuración
conn = setup_database(...)

## 4. Análisis
### 4.1 Sección 1
### 4.2 Sección 2

## 5. Conclusiones (en español)

## 6. Limpieza
conn.close()
```

#### ✅ Usar Funciones de Utils

**SIEMPRE** preferir funciones de `src/utils/` sobre código inline:

```python
# ✅ Correcto
from src.utils.duckdb_utils import get_stats_by_column
stats = get_stats_by_column('steel_cleaned', 'Load_Type', 'Usage_kWh')

# ❌ Incorrecto
stats = conn.execute("""
    SELECT Load_Type, COUNT(*) as count, AVG(Usage_kWh) as avg
    FROM steel_cleaned
    GROUP BY Load_Type
""").pl()
```

---

### 11. Manejo de Errores

#### ✅ Logging en lugar de prints

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Correcto
logger.info("Processing data...")
logger.error(f"Failed to load file: {e}")

# ❌ Incorrecto
print("Processing data...")
print(f"Error: {e}")
```

#### ✅ Manejo de Excepciones

```python
# ✅ Correcto
try:
    df = load_data(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# ❌ Incorrecto
try:
    df = load_data(path)
except:
    pass
```

---

### 12. Git Commits

#### ✅ Conventional Commits

```bash
# Formato: <type>(<scope>): <description>

# Tipos:
feat:     # Nueva funcionalidad
fix:      # Corrección de bug
docs:     # Cambios en documentación
style:    # Formateo (no afecta código)
refactor: # Refactorización
test:     # Agregar/modificar tests
chore:    # Tareas de mantenimiento

# Ejemplos:
git commit -m "feat(duckdb): add setup_database function"
git commit -m "fix(utils): handle special characters in column names"
git commit -m "docs: update AGENTS.md with best practices"
git commit -m "test(duckdb): add tests for correlation function"
```

---

### 13. Dependencias

#### ✅ Gestión con Poetry

```bash
# Agregar dependencia
poetry add polars

# Agregar dependencia de desarrollo
poetry add --group dev pytest

# Actualizar dependencias
poetry update

# NUNCA editar pyproject.toml manualmente para dependencias
```

---

### 14. Configuración de Notebooks

#### ✅ Setup Inicial

**SIEMPRE** incluir al inicio de notebooks:

```python
# Imports
import sys
sys.path.append('../..')  # Para importar desde src/

# Configuración de visualización
import warnings
warnings.filterwarnings('ignore')

# Polars config
pl.Config.set_tbl_rows(20)
pl.Config.set_fmt_str_lengths(100)
```

---

### 15. Funciones Reutilizables Disponibles

#### ✅ DuckDB (`src/utils/duckdb_utils.py`)

```python
from src.utils.duckdb_utils import (
    setup_database,              # Setup automático de DB
    quick_query,                 # Query rápido
    get_stats_by_column,         # Estadísticas por columna
    get_temporal_stats,          # Análisis temporal
    get_top_n,                   # Top N registros
    get_correlation,             # Correlaciones
    get_weekend_vs_weekday_stats # Comparación fin de semana
)
```

#### ✅ Data Cleaning (`src/utils/data_cleaning.py`)

```python
from src.utils.data_cleaning import (
    convert_data_types,          # Conversión de tipos
    handle_null_values,          # Manejo de nulos
    correct_range_violations,    # Corrección de rangos
    treat_outliers,              # Tratamiento de outliers
    remove_duplicates            # Eliminación de duplicados
)
```

#### ✅ Data Quality (`src/utils/data_quality.py`)

```python
from src.utils.data_quality import (
    compare_schemas,             # Comparar schemas
    analyze_nulls,               # Análisis de nulos
    validate_types               # Validación de tipos
)
```

---

## 🚫 Anti-Patrones (Evitar)

### ❌ Código Duplicado

```python
# ❌ NO hacer esto en múltiples notebooks
conn = duckdb.connect("data/steel.duckdb")
df = conn.execute("SELECT * FROM steel_cleaned").pl()
conn.close()

# ✅ Usar función reutilizable
from src.utils.duckdb_utils import quick_query
df = quick_query("SELECT * FROM steel_cleaned")
```

### ❌ Hardcoded Paths

```python
# ❌ Incorrecto
df = pl.read_csv("C:/Users/arthu/Desktop/data.csv")

# ✅ Correcto
from pathlib import Path
data_dir = Path("data/raw")
df = pl.read_csv(data_dir / "data.csv")
```

### ❌ Magic Numbers

```python
# ❌ Incorrecto
df = df.filter(pl.col('value') > 0.5)

# ✅ Correcto
THRESHOLD = 0.5
df = df.filter(pl.col('value') > THRESHOLD)
```

### ❌ Commits de Archivos Grandes

```python
# ❌ NUNCA hacer esto
git add data/large_dataset.csv
git commit -m "add dataset"

# ✅ Usar DVC
dvc add data/large_dataset.csv
git add data/large_dataset.csv.dvc
git commit -m "data: add large dataset with DVC"
```

---

## 📚 Referencias Rápidas

### Estructura del Proyecto

```
mlops_proyecto_atreides/
├── src/
│   ├── data/           # Scripts de procesamiento
│   ├── features/       # Feature engineering (POO)
│   ├── models/         # Training y pipelines
│   ├── api/            # FastAPI backend
│   └── utils/          # Funciones reutilizables ⭐
├── notebooks/
│   ├── exploratory/    # EDA (texto en español)
│   └── experimental/   # Experimentos
├── tests/
│   ├── unit/          # Tests unitarios
│   ├── integration/   # Tests de integración
│   └── e2e/           # Tests end-to-end
├── data/              # Datos (versionados con DVC)
├── models/            # Modelos (versionados con DVC)
└── docs/              # Documentación
```

### Comandos Útiles

```bash
# Tests
poetry run pytest tests/ -v
poetry run pytest --cov=src --cov-report=html

# Formateo
poetry run black .
poetry run ruff check .

# DVC
dvc add data/file.csv
dvc push
dvc pull

# MLflow
poetry run mlflow ui

# Git
git status
git add .
git commit -m "type(scope): description"
git push
```

---

## ✅ Checklist para Nuevas Funcionalidades

Antes de considerar una tarea completada:

- [ ] Código en inglés, documentación en español
- [ ] Funciones reutilizables en `src/utils/` si aplica
- [ ] Docstrings estilo Google
- [ ] Type hints en todas las funciones
- [ ] Tests unitarios (coverage >70%)
- [ ] Formateado con Black
- [ ] Sin warnings de Ruff
- [ ] Datos/modelos versionados con DVC
- [ ] Experimentos loggeados en MLflow
- [ ] Commit con Conventional Commits
- [ ] Documentación actualizada

---

## 🎓 Principios de Diseño

1. **DRY (Don't Repeat Yourself)**: Si lo usas dos veces, crea una función
2. **KISS (Keep It Simple, Stupid)**: Código simple es mejor que código complejo
3. **YAGNI (You Aren't Gonna Need It)**: No agregues funcionalidad que no necesitas ahora
4. **Separation of Concerns**: Cada módulo tiene una responsabilidad clara
5. **Code Reusability**: Funciones en utils, no código duplicado en notebooks

---

## 🚨 Código Natural - Evitar Marcadores de IA

### ❌ NO usar elementos que delatan código generado por IA

**Evitar estos patrones típicos de IA**:

```python
# ❌ NO usar emojis en comentarios
# 🔍 Load the data
# ✅ Process the results
# 🚀 Train the model

# ❌ NO usar separadores decorativos
print("="*70)
print("-"*50)
print("*"*80)

# ❌ NO usar comentarios excesivamente descriptivos
# Step 1: Initialize the connection
# Step 2: Load the data
# Step 3: Process the results

# ❌ NO usar banners ASCII
# ============================================
# SECTION: DATA LOADING
# ============================================
```

**✅ Usar código natural y profesional**:

```python
# Load and validate data
df = pl.read_parquet("data/processed/steel_cleaned.parquet")

# Calculate summary statistics
summary = df.describe()

# Train model with cross-validation
model.fit(X_train, y_train)
```

### ✅ Estilo de Comentarios Profesional

```python
# ✅ Correcto - Comentarios concisos y técnicos
# Filter outliers using IQR method
df_clean = df.filter((pl.col('value') > q1) & (pl.col('value') < q3))

# Calculate rolling average for smoothing
df = df.with_columns(pl.col('usage').rolling_mean(window_size=7))

# ❌ Incorrecto - Comentarios obvios o decorativos
# Now we will filter the outliers from our dataset using the IQR method
# This is an important step in our data cleaning process
df_clean = df.filter((pl.col('value') > q1) & (pl.col('value') < q3))
```

### ✅ Output y Logging Natural

```python
# ✅ Correcto - Mensajes informativos simples
logger.info(f"Loaded {len(df)} records")
logger.info(f"Training completed in {elapsed:.2f}s")

# ❌ Incorrecto - Mensajes decorativos
logger.info("="*70)
logger.info("🎯 Starting training process...")
logger.info("="*70)
```

### ✅ Separación de Secciones en Notebooks

```python
# ✅ Correcto - Usar markdown cells para secciones
# En Markdown cell:
## 2. Data Loading

# En Code cell:
df = pl.read_parquet("data/processed/steel_cleaned.parquet")

# ❌ Incorrecto - Separadores en código
print("\n" + "="*70)
print("SECTION 2: DATA LOADING")
print("="*70 + "\n")
df = pl.read_parquet("data/processed/steel_cleaned.parquet")
```

### Reglas Generales

1. **Sin emojis** en código o comentarios
2. **Sin separadores decorativos** (=, -, *)
3. **Comentarios concisos** y técnicos, no narrativos
4. **Mensajes de log simples** y directos
5. **Usar markdown cells** para estructura en notebooks, no prints decorativos

---

**Versión**: 1.0  
**Última actualización**: 2025-10-16  
**Mantenido por**: MLOps Team - Proyecto Atreides

---

## 📞 Contacto

Si tienes dudas sobre estas prácticas, consulta:
- `docs/` - Documentación del proyecto
- `README.md` - Guía de inicio
- `STRUCTURE.md` - Estructura del proyecto
