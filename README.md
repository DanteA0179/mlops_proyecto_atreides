# Energy Optimization Copilot ⚡

**Sistema de Optimización Energética con IA para la Industria Siderúrgica**

Un copiloto inteligente que combina Foundation Models de series temporales con IA generativa para predicción y análisis conversacional de consumo energético.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-Poetry-blue)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-Ruff-red)](https://github.com/astral-sh/ruff)
[![DVC](https://img.shields.io/badge/data-DVC-orange)](https://dvc.org/)

## 🎯 Objetivo del Proyecto

Desarrollar un sistema MLOps completo que:
- **Predice** consumo energético con RMSE < 0.205 (15% mejor que benchmark CUBIST)
- **Explica** drivers de consumo mediante análisis conversacional
- **Optimiza** operaciones industriales a través de simulaciones "what-if"

## 👥 Equipo

| Rol | Nombre | Responsabilidades |
|-----|--------|-------------------|
| Data Engineer | Juan | Pipeline de datos, DVC, data quality |
| Data Scientist | Erick | EDA, feature engineering, análisis |
| ML Engineer | Julian | Training, Foundation Models, optimización |
| Software Engineer & Scrum Master | Dante | API, frontend, gestión de proyecto |
| MLOps/SRE Engineer | Arthur | Infraestructura, CI/CD, deployment |

## 📊 Dataset

- **Fuente:** [UCI ML Repository - Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
- **Registros:** 35,040 mediciones (2018)
- **Frecuencia:** 15 minutos
- **Variable objetivo:** `Usage_kWh` (consumo energético)

## 🏗️ Arquitectura

```
├── data/                   # Datos (versionados con DVC)
│   ├── raw/               # Datos originales (sucio)
│   ├── processed/         # Datos limpios
│   └── external/          # Datos externos
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # EDA
│   └── experimental/      # Experimentos
├── src/                   # Código fuente
│   ├── data/             # Scripts de procesamiento
│   ├── features/         # Feature engineering
│   ├── models/           # Entrenamiento y evaluación
│   ├── api/              # FastAPI backend
│   ├── monitoring/       # Drift detection (Evidently)
│   └── utils/            # Utilidades
├── models/                # Modelos entrenados (versionados con DVC)
│   ├── baselines/        # XGBoost, LightGBM
│   └── foundation/       # Chronos, TimesFM
├── tests/                 # Tests (pytest)
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── reports/               # Reportes y visualizaciones
│   ├── figures/
│   └── metrics/
├── .github/workflows/     # CI/CD (GitHub Actions)
└── config/                # Configuraciones
```

## 🚀 Quickstart

### Prerrequisitos

- Python 3.11 o superior
- Poetry 1.8+
- Git 2.40+
- DVC 3.48+
- Docker 24.x (opcional, para deployment)

### 1. Clonar el Repositorio

```bash
git https://github.com/DanteA0179/mlops_proyecto_atreides.git
cd mlops_proyecto_atreides
```

### 2. Configurar Entorno con Poetry

```bash
# Instalar Poetry (si no lo tienes)
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependencias
poetry install

# Activar entorno virtual
poetry shell
```

### 3. Configurar Credenciales con Google Secret Manager

```bash
# Autenticarse en GCP
gcloud auth login
gcloud auth application-default login
gcloud config set project mlops-atreides-2025

# Cargar secretos del proyecto (DVC, MLflow, API keys, etc.)
python scripts/setup_secrets.py load --project-id mlops-atreides-2025

# Ver guía completa: docs/SECRET_MANAGER_SETUP.md
```

### 4. Configurar DVC con Google Cloud Storage

```bash
# Opción 1: Script automático (recomendado)
# Windows:
.\scripts\setup_dvc_credentials.ps1

# Linux/Mac:
./scripts/setup_dvc_credentials.sh

# Opción 2: Manual
gcloud auth application-default login
dvc remote modify --local gcs-remote credentialpath "$env:APPDATA\gcloud\application_default_credentials.json"

# Verificar configuración
dvc remote list

# Descargar datos y modelos desde remote
dvc pull
```

**📖 Ver guía completa:** [docs/DVC_SETUP.md](docs/DVC_SETUP.md)

### 5. Ejecutar la API con Docker (Recomendado)

```bash
# Opción 1: Docker Compose (más fácil)
docker-compose up -d

# Verificar que la API está corriendo
curl http://localhost:8000/health

# Ver logs
docker-compose logs -f api

# Opción 2: Docker build manual
docker build -t energy-optimization-api:latest .
docker run -d -p 8000:8000 --name energy-api energy-optimization-api:latest
```

**📖 Ver guía completa:** [docs/DOCKER_SETUP.md](docs/DOCKER_SETUP.md)

### 6. Cargar Datos a DuckDB (Opcional pero Recomendado)

```bash
# Cargar datos limpios a DuckDB para exploración SQL interactiva
poetry run python src/data/load_to_duckdb.py

# Esto crea: data/steel.duckdb con tabla 'steel_cleaned'
```

**Uso en Python:**
```python
from src.data.load_to_duckdb import create_database, query_to_dataframe

# Conectar a DB
conn = create_database("data/steel.duckdb")

# Query interactivo
df = query_to_dataframe(conn, """
    SELECT Load_Type, AVG(Usage_kWh) as avg_usage
    FROM steel_cleaned
    GROUP BY Load_Type
""")
print(df)
```

### 7. Ejecutar la API Localmente (Desarrollo)

```bash
# Opción 1: Uvicorn con hot reload
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Opción 2: Ejecutar el main.py directamente
poetry run python src/api/main.py

# Acceder a la documentación interactiva
# http://localhost:8000/docs
```

### 8. Verificar Instalación

```bash
# Ejecutar script de verificación US-003
poetry run python scripts/verify_us003.py

# Ejecutar tests
poetry run pytest

# Ejecutar tests específicos de DuckDB
poetry run pytest tests/unit/test_load_to_duckdb.py -v

# Verificar linting
poetry run ruff check .
poetry run black --check .

# Verificar mypy (type checking)
poetry run mypy src/
```

## 🔍 Feature Importance Analysis

El proyecto incluye un módulo completo para análisis de importancia de features que ayuda a identificar las variables más predictivas del consumo energético.

### Uso Rápido

```python
import polars as pl
from src.utils.feature_importance import (
    calculate_mutual_information,
    calculate_pearson_correlation,
    plot_feature_importance,
    compare_importance_methods
)

# Cargar datos
df = pl.read_parquet("data/processed/steel_cleaned.parquet")

# Calcular mutual information (captura relaciones no lineales)
mi_scores = calculate_mutual_information(df, target_column="Usage_kWh")
print(mi_scores.head(10))

# Calcular correlaciones de Pearson (relaciones lineales)
correlations = calculate_pearson_correlation(df, target_column="Usage_kWh")
print(correlations.head(10))

# Visualizar top 10 features
fig = plot_feature_importance(
    mi_scores,
    score_column='mi_score',
    method_name='Mutual Information',
    top_n=10,
    output_path='reports/figures/mutual_information_top10.png'
)

# Comparar ambos métodos
comparison = compare_importance_methods(mi_scores, correlations, top_n=10)
print(f"Overlap: {comparison['overlap_percentage']:.1f}%")
print(f"Common features: {comparison['common_features']}")
```

### Notebooks de Análisis

- **`notebooks/exploratory/05_feature_importance_analysis.ipynb`** - Análisis completo con visualizaciones y conclusiones en español

### Funciones Disponibles

| Función | Descripción |
|---------|-------------|
| `calculate_mutual_information()` | Calcula mutual information scores (relaciones no lineales) |
| `calculate_pearson_correlation()` | Calcula correlaciones de Pearson (relaciones lineales) |
| `plot_feature_importance()` | Genera gráficos de barras horizontales de top N features |
| `get_top_features()` | Extrae lista de nombres de top N features |
| `compare_importance_methods()` | Compara rankings de MI vs correlación |

### Resultados Clave

**Top 5 Features por Mutual Information:**
1. CO2(tCO2) - 1.214
2. Lagging_Current_Power_Factor - 1.204
3. Lagging_Current_Reactive.Power_kVarh - 0.823
4. NSM - 0.450
5. Leading_Current_Power_Factor - 0.413

**Top 5 Features por Correlación de Pearson:**
1. CO2(tCO2) - 0.894
2. Lagging_Current_Reactive.Power_kVarh - 0.782
3. Lagging_Current_Power_Factor - 0.358
4. Leading_Current_Power_Factor - 0.328
5. Leading_Current_Reactive_Power_kVarh - -0.295

**Overlap:** 100% de las top 5 features aparecen en ambos métodos, indicando consenso robusto.

## �️ Feature Engineering

El proyecto incluye un sistema completo de ingeniería de features temporales que transforma datos de tiempo en representaciones óptimas para modelos de Machine Learning.

### Features Temporales (US-011)

Se crearon **7 features temporales** a partir del dataset limpio:

#### Features Directos
1. **`hour`** (0-23): Hora del día extraída de NSM (Number of Seconds from Midnight)
2. **`day_of_week`** (0-6): Día de la semana numérico (Lunes=0, Domingo=6)
3. **`is_weekend`** (bool): Indicador de fin de semana (Sábado/Domingo)

#### Features Cíclicos (Cyclical Encoding)
4. **`cyclical_hour_sin`**: sin(2π × hour / 24)
5. **`cyclical_hour_cos`**: cos(2π × hour / 24)
6. **`cyclical_day_sin`**: sin(2π × day / 7)
7. **`cyclical_day_cos`**: cos(2π × day / 7)

### ¿Por Qué Codificación Cíclica?

**Problema**: Los features de tiempo son cíclicos, pero los valores numéricos directos los tratan como lineales:
- Hora 23 está cerca de Hora 0 (1 hora de diferencia)
- Pero numéricamente: |23 - 0| = 23 ❌

**Solución**: Mapear features temporales al círculo unitario usando trigonometría:
- Preserva periodicidad: hora 23 ≈ hora 0
- Mantiene ortogonalidad: sin²(θ) + cos²(θ) = 1
- Permite a modelos ML entender proximidad temporal correctamente

### Uso con Sklearn Transformers (POO)

```python
from src.features import TemporalFeatureEngineer

# Opción 1: Transformer completo (todos los features)
engineer = TemporalFeatureEngineer(
    nsm_col='NSM',
    day_name_col='Day_of_week'
)

df_featured = engineer.fit_transform(df)
# Output: 7 nuevas columnas agregadas

# Opción 2: Uso en sklearn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ('temporal', TemporalFeatureEngineer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Uso con Funciones de Utilidad

```python
from src.features import (
    create_all_temporal_features,
    validate_temporal_features
)

# Crear todos los features de una vez
df_featured = create_all_temporal_features(
    df,
    nsm_col='NSM',
    day_name_col='Day_of_week'
)

# Validar que todos los features fueron creados correctamente
validation = validate_temporal_features(df_featured)

if validation['valid']:
    print("✅ Todos los features temporales son válidos")
else:
    print(f"❌ Problemas encontrados: {validation}")
```

### Transformers Individuales

```python
from src.features import (
    HourExtractor,
    DayOfWeekEncoder,
    WeekendIndicator,
    CyclicalEncoder
)

# Extraer solo hora
hour_extractor = HourExtractor(nsm_col='NSM')
df = hour_extractor.fit_transform(df)

# Codificación cíclica genérica (reutilizable!)
# Ejemplo 1: Hora
hour_encoder = CyclicalEncoder(column='hour', period=24)
df = hour_encoder.fit_transform(df)

# Ejemplo 2: Mes (¡funciona para cualquier feature cíclico!)
month_encoder = CyclicalEncoder(column='month', period=12)
df = month_encoder.fit_transform(df)

# Ejemplo 3: Dirección del viento
wind_encoder = CyclicalEncoder(column='wind_direction', period=360)
df = wind_encoder.fit_transform(df)
```

### Notebooks de Análisis

- **`notebooks/exploratory/06_temporal_feature_engineering.ipynb`** - Análisis completo con:
  - Explicación matemática de codificación cíclica
  - Visualizaciones (gráficos polares, mapas de calor)
  - Análisis de correlación con variable objetivo
  - Patrones de consumo por hora y día

### Resultados Clave

**Correlaciones con `Usage_kWh`:**
- `cyclical_hour_cos`: **-0.44** (correlación más fuerte)
- `cyclical_hour_sin`: -0.24
- `day_of_week`: -0.22
- `hour`: +0.22
- `cyclical_day_sin`: +0.21
- `cyclical_day_cos`: -0.07

**Insight**: Las features cíclicas (`cyclical_hour_cos`) capturan patrones temporales mejor que features lineales (`hour`), con correlaciones casi 2× más fuertes.

### Dataset Generado

- **Input**: `data/processed/steel_cleaned.parquet` (34,910 × 11)
- **Output**: `data/processed/steel_featured.parquet` (34,910 × 18)
- **Tamaño**: 0.72 MB (compresión Snappy)
- **Validación**: ✅ 8 checks automáticos (rangos, tipos, ortogonalidad)

### Scripts Ejecutables

```bash
# Generar dataset con features temporales
poetry run python src/features/build_features.py

# Outputs generados:
# - data/processed/steel_featured.parquet
# - reports/feature_engineering_report.md
# - reports/feature_engineering_log.json
```

### Testing

```bash
# Tests de transformers (98.51% coverage)
poetry run pytest tests/unit/test_temporal_transformers.py -v

# Tests de funciones utilitarias (89.32% coverage)
poetry run pytest tests/unit/test_temporal_features.py -v

# Todos los tests de features temporales
poetry run pytest tests/unit/test_temporal* -v
```

## �📝 Desarrollo

### Flujo de Trabajo

1. **Crear branch desde develop:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/EP-XXX-descripcion
   ```

2. **Hacer cambios y commitear:**
   ```bash
   # Pre-commit hooks se ejecutan automáticamente
   git add .
   git commit -m "feat: descripción del cambio"
   ```

3. **Ejecutar tests:**
   ```bash
   poetry run pytest tests/ -v
   poetry run pytest --cov=src --cov-report=html
   ```

4. **Crear Pull Request:**
   ```bash
   git push origin feature/EP-XXX-descripcion
   # Abrir PR en GitHub hacia develop
   ```

### Notebooks Jupyter

```bash
# Iniciar JupyterLab
poetry run jupyter lab

# O usar notebooks existentes en notebooks/
```

### Experimentos MLflow

```bash
# Iniciar MLflow UI
poetry run mlflow ui --port 5000

# Acceder a http://localhost:5000
```

### Pipeline de Entrenamiento (Prefect)

```bash
# Iniciar Prefect server (local)
poetry run prefect server start

# Ejecutar flow de entrenamiento
poetry run python src/models/train_pipeline.py
```

## 🧪 Testing

```bash
# Todos los tests
poetry run pytest

# Tests con coverage
poetry run pytest --cov=src --cov-report=term-missing

# Tests específicos
poetry run pytest tests/unit/test_data_cleaning.py -v

# Tests con markers
poetry run pytest -m "not slow"
```

## 🔧 Configuración

### Variables de Entorno y Secretos

**🔐 Este proyecto usa Google Secret Manager para gestionar credenciales de forma segura.**

✅ **NO** se usan archivos `.env` para credenciales
✅ Todo el equipo accede a las mismas configuraciones
✅ Cero riesgo de commits accidentales de credenciales

Ver [docs/SECRET_MANAGER_SETUP.md](docs/SECRET_MANAGER_SETUP.md) para configuración completa.

#### Quickstart para Nuevos Miembros

```bash
# 1. Autenticarse en GCP
gcloud auth login
gcloud auth application-default login
gcloud config set project mlops-atreides-2025

# 2. Cargar secretos
python scripts/setup_secrets.py load --project-id mlops-atreides-2025

# 3. Verificar
echo $GCP_PROJECT_ID
echo $GCS_BUCKET_NAME
```

### Configuración Local (Solo para Desarrollo sin GCP)

**Nota**: Si tienes acceso a GCP, usa Secret Manager en su lugar.

Para desarrollo local sin acceso a GCP, crear archivo `.env`:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# DVC Remote (GCS)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
DVC_REMOTE=gcs

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ollama (LLM local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### Pre-commit Hooks

```bash
# Instalar hooks
poetry run pre-commit install

# Ejecutar manualmente
poetry run pre-commit run --all-files
```

## 🐳 Docker

### Desarrollo Local

```bash
# Build
docker build -t energy-opt-api:latest .

# Run
docker run -p 8000:8000 energy-opt-api:latest

# Con docker-compose
docker-compose up -d
```

### Deployment a Cloud Run (GCP)

```bash
# Autenticar
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build & Push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/energy-opt-api

# Deploy
gcloud run deploy energy-opt-api \
  --image gcr.io/YOUR_PROJECT_ID/energy-opt-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2
```

## 📈 Métricas del Proyecto

### Objetivos de Performance

| Métrica | Benchmark (CUBIST) | Meta | Status |
|---------|-------------------|------|--------|
| RMSE | 0.2410 | < 0.205 | 🔄 En progreso |
| MAE | 0.0547 | < 0.046 | 🔄 En progreso |
| CV (%) | 0.8770 | < 0.75 | 🔄 En progreso |
| Latencia API | N/A | < 500ms p95 | 🔄 En progreso |
| Test Coverage | N/A | > 70% | 🔄 En progreso |

### Stack Tecnológico

**Data & ML:**
- Polars, Pandas, NumPy, DuckDB
- Scikit-learn, XGBoost, LightGBM
- Chronos (Amazon), TimesFM (Google)
- PyTorch, Transformers, Accelerate

**MLOps:**
- DVC (data versioning)
- MLflow (experiment tracking)
- Prefect (orchestration)
- Evidently (monitoring)

**Backend:**
- FastAPI, Pydantic, Uvicorn
- Docker, Cloud Run (GCP)

**LLM & AI:**
- Ollama (local inference)
- Llama 3.2 (3B)
- LangChain

## 📚 Documentación

### Guías de Configuración
- **[Configuración de DVC con GCS](docs/DVC_SETUP.md)** - Setup rápido de DVC para el equipo
- **[Google Secret Manager Setup](docs/SECRET_MANAGER_SETUP.md)** - Gestión segura de credenciales

### Documentación del Proyecto
- [Plan de Proyecto](context/PlaneacionProyecto.md)
- [ML Canvas](docs/ml_canvas.md) (pendiente)
- [API Documentation](http://localhost:8000/docs) (Swagger UI)
- [Architecture Decision Records](docs/adr/) (pendiente)

## 🤝 Contribución

1. Hacer fork del repositorio
2. Crear feature branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'feat: agregar nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Abrir Pull Request

### Convenciones de Commit

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Cambios en documentación
- `style:` Formateo, sin cambios de código
- `refactor:` Refactoring de código
- `test:` Agregar o modificar tests
- `chore:` Tareas de mantenimiento

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- UCI Machine Learning Repository por el dataset
- Amazon Science (Chronos)
- Google Research (TimesFM)
- Comunidad open-source de Python/ML

---

**Proyecto desarrollado como parte del curso de MLOps**

*Última actualización: Octubre 2025*
