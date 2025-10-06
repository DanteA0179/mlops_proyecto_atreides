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
git clone https://github.com/your-org/energy-optimization-copilot.git
cd energy-optimization-copilot
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

### 3. Configurar DVC

```bash
# Inicializar DVC (ya está configurado)
dvc pull  # Descargar datos y modelos desde remote

# Si es primera vez, configurar remote (GCS)
dvc remote add -d gcs gs://energy-opt-dvc-remote
dvc remote modify gcs credentialpath path/to/service-account.json
```

### 4. Verificar Instalación

```bash
# Ejecutar tests
poetry run pytest

# Verificar linting
poetry run ruff check .
poetry run black --check .

# Verificar mypy (type checking)
poetry run mypy src/
```

## 📝 Desarrollo

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

### Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

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

**Proyecto desarrollado como parte del curso de MLOps - Maestría en Ciencia de Datos**

*Última actualización: Octubre 2025*
