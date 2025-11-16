# Project Structure Documentation

Este documento describe la estructura del proyecto Energy Optimization Copilot, basada en **Cookiecutter Data Science** con adaptaciones específicas para MLOps moderno.

## 🏛️ Cookiecutter Data Science - Adaptaciones

Este proyecto sigue las convenciones de [Cookiecutter Data Science v2](https://drivendata.github.io/cookiecutter-data-science/), el estándar de facto para proyectos de ciencia de datos, con las siguientes adaptaciones para MLOps:

### ✅ Estructura Base de Cookiecutter (Conservada)
- `data/` - Organización de datos (raw, processed, external)
- `notebooks/` - Jupyter notebooks para exploración y experimentación
- `src/` - Código fuente del proyecto como paquete Python
- `models/` - Modelos entrenados y serializados
- `reports/` - Análisis y visualizaciones generadas
- `README.md` - Documentación principal del proyecto

### 🔧 Adaptaciones para MLOps

#### 1. **Versionado de Datos y Modelos**
- **Agregado**: DVC (Data Version Control) para tracking de datos y modelos
- **Razón**: Reproducibilidad y colaboración en artefactos grandes
- **Estructura**: Archivos `.dvc` en `data/` y `models/`

#### 2. **Orquestación y Workflow**
- **Agregado**: Prefect para pipelines de ML
- **Razón**: Automatización de entrenamiento y deployment
- **Ubicación**: Flujos en `src/pipelines/` (futuro)

#### 3. **Experiment Tracking**
- **Agregado**: MLflow para tracking de experimentos
- **Razón**: Comparación de modelos y reproducibilidad
- **Nota**: Los artefactos `mlruns/` están en `.gitignore`

#### 4. **CI/CD**
- **Agregado**: `.github/workflows/` para GitHub Actions
- **Razón**: Automatización de tests, linting y deployment
- **Cookiecutter estándar**: No incluye CI/CD por defecto

#### 5. **Configuración y Secretos**
- **Agregado**: `config/` para configuraciones YAML
- **Agregado**: `.env.example` para variables de entorno
- **Razón**: Separación de código y configuración (12-factor app)

#### 6. **API y Deployment**
- **Agregado**: `src/api/` para FastAPI backend
- **Razón**: Servir modelos en producción
- **Cookiecutter estándar**: Solo incluye `src/models/predict_model.py`

#### 7. **Monitoring**
- **Agregado**: `src/monitoring/` para drift detection
- **Razón**: Observabilidad en producción con Evidently AI
- **Cookiecutter estándar**: No contempla monitoring

#### 8. **Testing Mejorado**
- **Agregado**: `tests/` con subdivisión (unit, integration, e2e)
- **Razón**: Testing comprehensivo para producción
- **Cookiecutter estándar**: Solo `tests/` plano

### 📊 Comparación: Cookiecutter vs Nuestro Proyecto

| Aspecto | Cookiecutter Estándar | Nuestro Proyecto |
|---------|----------------------|------------------|
| **Versionado de datos** | No incluido | ✅ DVC |
| **Experiment tracking** | No incluido | ✅ MLflow |
| **Orquestación** | No incluido | ✅ Prefect |
| **API** | No incluido | ✅ FastAPI |
| **Monitoring** | No incluido | ✅ Evidently AI |
| **CI/CD** | No incluido | ✅ GitHub Actions |
| **Containerización** | Opcional | ✅ Docker + Docker Compose |
| **Tests** | Básico | ✅ Unit + Integration + E2E |
| **Pre-commit hooks** | No incluido | ✅ Black, Ruff, Bandit |

### 🎯 Principios Mantenidos de Cookiecutter

1. **Data is immutable** - `data/raw/` nunca se modifica
2. **Notebooks are for exploration** - Código de producción va en `src/`
3. **Build from the environment up** - Poetry para gestión de dependencias
4. **Keep secrets out of version control** - `.env` en `.gitignore`
5. **Analysis is a DAG** - Pipeline de datos es direccional

## 📁 Estructura de Directorios

```
energy-optimization-copilot/
│
├── .dvc/                          # Configuración DVC
│   ├── config                     # Remote storage config (GCS)
│   └── .gitignore
│
├── .github/                       # GitHub workflows
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration
│       ├── cd.yml                 # Continuous Deployment
│       └── tests.yml              # Automated testing
│
├── config/                        # Archivos de configuración
│   ├── logging.yaml              # Logging configuration
│   ├── model_config.yaml         # Model hyperparameters
│   └── data_schema.yaml          # Data validation schemas
│
├── context/                       # Documentación del proyecto
│   ├── PlaneacionProyecto.md     # Plan completo del proyecto
│   └── sprint_reports/           # Reportes de sprints
│
├── data/                          # Datos (versionados con DVC)
│   ├── raw/                      # Datos originales inmutables
│   │   ├── .gitkeep
│   │   └── steel_dirty.csv.dvc   # Versionado con DVC
│   │
│   ├── processed/                # Datos procesados intermedios
│   │   ├── .gitkeep
│   │   ├── steel_cleaned.parquet.dvc
│   │   └── steel_featured.parquet.dvc
│   │
│   └── external/                 # Datos de fuentes externas
│       └── .gitkeep
│
├── models/                        # Modelos entrenados (versionados con DVC)
│   ├── baselines/                # Modelos clásicos
│   │   ├── .gitkeep
│   │   ├── xgboost_v1.pkl.dvc
│   │   └── lightgbm_v1.pkl.dvc
│   │
│   └── foundation/               # Foundation Models
│       ├── .gitkeep
│       └── chronos_finetuned_v1.pth.dvc
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploratory/              # EDA inicial
│   │   ├── 00_data_profiling.ipynb
│   │   ├── 01_EDA.ipynb
│   │   └── 02_time_series_analysis.ipynb
│   │
│   └── experimental/             # Experimentos de modelos
│       ├── 10_baseline_models.ipynb
│       └── 11_chronos_finetuning.ipynb
│
├── reports/                       # Reportes generados
│   ├── figures/                  # Gráficos y visualizaciones
│   │   ├── .gitkeep
│   │   ├── eda_distributions.png
│   │   └── model_performance.png
│   │
│   └── metrics/                  # Métricas de modelos
│       ├── .gitkeep
│       └── model_comparison.csv
│
├── src/                           # Código fuente del proyecto
│   ├── __init__.py
│   │
│   ├── data/                     # Scripts de procesamiento de datos
│   │   ├── __init__.py
│   │   ├── clean_data.py        # Limpieza de datos
│   │   ├── load_to_duckdb.py    # Carga a DuckDB
│   │   └── validate.py          # Validación de esquemas
│   │
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── build_features.py    # Crear features temporales
│   │   └── transformers.py      # Transformaciones custom
│   │
│   ├── models/                   # Entrenamiento y evaluación
│   │   ├── __init__.py
│   │   ├── train_baseline.py    # XGBoost/LightGBM
│   │   ├── train_chronos.py     # Fine-tuning Chronos
│   │   ├── evaluate.py          # Métricas de evaluación
│   │   └── predict.py           # Inferencia
│   │
│   ├── api/                      # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py              # App principal
│   │   ├── routes/              # Endpoints
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py    # /predict
│   │   │   └── copilot.py       # /copilot/chat
│   │   ├── schemas.py           # Pydantic models
│   │   └── dependencies.py      # Dependency injection
│   │
│   ├── monitoring/               # Drift detection
│   │   ├── __init__.py
│   │   ├── evidently_monitor.py # Configuración Evidently
│   │   └── alerts.py            # Sistema de alertas
│   │
│   └── utils/                    # Utilidades
│       ├── __init__.py
│       ├── logging.py           # Configuración de logging
│       ├── config.py            # Carga de configuración
│       └── metrics.py           # Métricas custom
│
├── tests/                         # Tests (pytest)
│   ├── __init__.py
│   │
│   ├── unit/                     # Tests unitarios
│   │   ├── __init__.py
│   │   ├── test_data_cleaning.py
│   │   └── test_features.py
│   │
│   ├── integration/              # Tests de integración
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   │
│   └── e2e/                      # Tests end-to-end
│       ├── __init__.py
│       └── test_api.py
│
├── scripts/                       # Scripts de automatización
│   ├── download_data.sh         # Descargar dataset
│   ├── setup_gcs.sh             # Configurar GCS remote
│   └── deploy.sh                # Script de deployment
│
├── .dvcignore                     # Archivos ignorados por DVC
├── .env.example                   # Ejemplo de variables de entorno
├── .gitignore                     # Archivos ignorados por Git
├── .pre-commit-config.yaml       # Configuración pre-commit hooks
├── docker-compose.yml            # Orquestación local
├── Dockerfile                     # Imagen Docker
├── LICENSE                        # Licencia MIT
├── poetry.lock                    # Lockfile de Poetry
├── pyproject.toml                # Configuración Poetry y tools
├── README.md                      # Documentación principal
└── STRUCTURE.md                  # Este archivo
```

## 📋 Convenciones

### Nomenclatura de Archivos

- **Python scripts**: `snake_case.py`
- **Notebooks**: `##_descripcion.ipynb` (números secuenciales)
- **Modelos DVC**: `nombre_version.ext.dvc`
- **Configs**: `nombre_config.yaml`

### Branches Git

```
main                    # Producción
├── develop            # Integración
    ├── feature/EP-XXX-descripcion
    ├── bugfix/EP-XXX-descripcion
    └── hotfix/descripcion
```

### Versionado DVC

```bash
# Agregar datos
dvc add data/raw/archivo.csv
git add data/raw/archivo.csv.dvc .gitignore
git commit -m "data: add raw steel dataset"

# Push a remote
dvc push

# Tag de versión
git tag -a data-v1.0 -m "Clean dataset version 1.0"
git push origin data-v1.0
```

## 🔧 Configuración por Entorno

### Desarrollo Local

```bash
# .env (local)
ENVIRONMENT=development
DEBUG=true
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Staging

```bash
# .env.staging
ENVIRONMENT=staging
DEBUG=false
MLFLOW_TRACKING_URI=https://mlflow-staging.example.com
```

### Producción

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
MLFLOW_TRACKING_URI=https://mlflow-prod.example.com
CLOUD_RUN_MIN_INSTANCES=1
```

## 📊 Flujo de Datos

```
1. Raw Data (CSV)
   ↓ [src/data/clean_data.py]
2. Cleaned Data (Parquet)
   ↓ [src/features/build_features.py]
3. Featured Data (Parquet)
   ↓ [src/models/train_*.py]
4. Trained Model (.pth/.pkl)
   ↓ [src/models/predict.py]
5. Predictions (JSON/Parquet)
   ↓ [src/monitoring/evidently_monitor.py]
6. Drift Reports (HTML)
```

## 🧪 Testing Strategy

### Unit Tests (tests/unit/)
- Funciones individuales
- Transformaciones de datos
- Utilidades

### Integration Tests (tests/integration/)
- Pipeline completo de datos
- Entrenamiento + evaluación
- API endpoints

### E2E Tests (tests/e2e/)
- Flujo completo: request → response
- Simulación de usuario
- Performance testing

## 📦 Artifacts Management

### DVC Tracked
- `data/raw/*.csv`
- `data/processed/*.parquet`
- `models/**/*.{pkl,pth,h5}`

### Git Tracked
- Código fuente (src/)
- Configuraciones (config/)
- Tests (tests/)
- Documentación (*.md)
- Notebooks (notebooks/)

### Not Tracked
- `mlruns/` (MLflow artifacts)
- `.env` (variables de entorno)
- `*.duckdb` (bases de datos locales)
- `__pycache__/` (bytecode)

## 🚀 Deployment Artifacts

### Docker

#### Archivos de Docker
- `Dockerfile.api` - Multi-stage Dockerfile optimizado para producción
- `docker-compose.yml` - Orquestación local (API + MLflow + DuckDB)
- `.dockerignore` - Exclusiones de build (data/, notebooks/, tests/)
- `docker/README.md` - Guía completa de Docker

#### Características de la Imagen
- **Base**: `python:3.11-slim`
- **Tamaño**: ~800MB-1GB (< 1.5GB target)
- **Build time**: < 5 min completo, < 30 seg rebuild
- **Arquitectura**: Multi-stage (builder + runtime)
- **Seguridad**: Non-root user (appuser)
- **Producción**: Gunicorn + Uvicorn workers
- **Modelos**: ONNX embebidos (~2-3MB)

#### Scripts de Build
- `scripts/docker_build.sh` - Build y validación (Linux/Mac)
- `scripts/docker_build.ps1` - Build y validación (Windows)

#### Estrategia de Modelos
1. **Embebidos** (default): Modelos ONNX en `/app/models/onnx`
2. **Externos** (opcional): Volúmenes en `/app/models/external`
3. **Fallback**: Env var `MODEL_PATH` configurable

### Cloud Run
- Artifact: `gcr.io/PROJECT_ID/energy-api:latest`
- Memory: 2Gi, CPU: 2, Timeout: 300s
- Auto-scaling: 0-10 instancias
- Deployment: GitHub Actions workflow

### Documentación de Deployment
- `docs/deployment/docker-deployment.md` - Guía técnica completa
- `docker/README.md` - Inicio rápido y troubleshooting

---

**Última actualización:** Noviembre 2025
**Mantenedor:** Equipo MLOps - Proyecto Atreides
