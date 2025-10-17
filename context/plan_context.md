# Plan de Proyecto - Contexto Resumido

## Objetivo General

Desarrollar un **Copiloto de IA para Optimización Energética** en la industria siderúrgica que combine modelos de pronóstico de series temporales (Foundation Models) con una interfaz conversacional de IA generativa, permitiendo a operadores industriales tomar decisiones informadas basadas en predicciones precisas y análisis de escenarios interactivos.

**Meta Principal:** Superar el benchmark CUBIST (RMSE: 0.2410) en al menos 15%, logrando RMSE < 0.205.

---

## Dataset: Steel Industry Energy Consumption

**Fuente:** UCI Machine Learning Repository  
**Registros:** 35,040 (año 2018, planta en Corea del Sur)  
**Frecuencia:** 15 minutos

### Variables Principales

**Target:**
- `Usage_kWh`: Consumo de energía

**Features:**
- `Lagging_Current_Reactive_Power_kVarh`: Potencia reactiva en atraso
- `Leading_Current_Reactive_Power_kVarh`: Potencia reactiva en adelanto
- `CO2(tCO2)`: Emisiones de dióxido de carbono
- `Lagging_Current_Power_Factor`: Factor de potencia en atraso
- `Leading_Current_Power_Factor`: Factor de potencia en adelanto
- `NSM` (Number of Seconds from Midnight): Marca temporal
- `WeekStatus`: Día de semana/fin de semana
- `Day_of_week`: Día específico
- `Load_Type`: Tipo de carga (Light, Medium, Maximum)

**Nota:** Dataset "sucio" con errores introducidos intencionalmente (nulos, outliers, inconsistencias).

---

## Stack Tecnológico

### Desarrollo y Datos
- **Python:** 3.11.x
- **Gestión de dependencias:** Poetry
- **Procesamiento:** Polars (principal), Pandas (compatibilidad)
- **Base de datos:** DuckDB (analytics local)
- **Formato:** Parquet (columnar comprimido)

### Machine Learning
- **Clásicos:** XGBoost, LightGBM, Scikit-learn
- **Foundation Models:** Chronos-T5 (Amazon), TimesFM (Google)
- **Framework:** PyTorch 2.2+
- **Hyperparameter tuning:** Optuna

### MLOps
- **Versionado:** DVC + Git
- **Tracking:** MLflow
- **Orquestación:** Prefect
- **Testing:** pytest (>70% coverage)
- **CI/CD:** GitHub Actions
- **Monitoring:** Evidently AI
- **Containerización:** Docker

### Backend y API
- **Framework:** FastAPI
- **Validación:** Pydantic
- **Server:** Uvicorn (dev), Gunicorn (prod)

### IA Generativa
- **Runtime:** Ollama
- **Modelo:** Llama 3.2 3B (local en GPU 4070)
- **Framework:** LangChain

### Frontend
- **UI:** Streamlit o Gradio

### Cloud (GCP)
- **Deployment:** Cloud Run (scale-to-zero)
- **Storage:** Cloud Storage (DVC remote)
- **Registry:** Container Registry
- **Presupuesto:** ≤ $50 USD total

### Visualización
- Matplotlib, Seaborn, Plotly

---

## Estructura de Sprints

### Sprint 1: Foundation & Baseline (Semanas 1-2)

**Objetivo:** Pipeline completo de datos a modelo, superando benchmark CUBIST

**Entregables clave:**
- Dataset limpio y versionado (DVC)
- EDA exhaustivo con visualizaciones
- Feature engineering (temporal, cíclico, lags)
- Modelos entrenados:
  - XGBoost (baseline)
  - LightGBM (baseline)
  - Chronos-T5 (zero-shot + fine-tuned)
- ML Canvas completo
- Presentación y video Sprint 1

**Criterio de éxito:** RMSE < 0.205 (15% mejor que CUBIST: 0.2410)

---

### Sprint 2: MLOps Automation (Semanas 3-4)

**Objetivo:** Automatizar pipeline, deployar API, implementar monitoreo

**Entregables clave:**
- Pipeline de entrenamiento automatizado (Prefect)
- API RESTful completa (FastAPI):
  - `POST /predict` - predicción individual
  - `POST /predict/batch` - predicción batch
  - `GET /health`, `/model/info`, `/model/metrics`
- Tests unitarios (>70% coverage)
- Dockerfile optimizado
- CI/CD con GitHub Actions
- Deployment en Cloud Run
- Monitoreo con Evidently AI (data drift, model drift)
- Documentación API (Swagger)

**Criterios de éxito:**
- API deployada con latencia p95 < 1 segundo
- Costos GCP < $15 en el sprint

---

### Sprint 3: Copilot & Deployment (Semanas 5-6)

**Objetivo:** Interfaz conversacional con LLM, deployment completo

**Entregables clave:**
- Ollama + Llama 3.2 3B configurado (GPU 4070)
- Endpoint `/copilot/chat` en FastAPI
- Integración LLM con backend:
  - Parsing de consultas en lenguaje natural
  - Orquestación de predicciones
  - Síntesis de respuestas
- Frontend Streamlit con 3 páginas:
  - Home/Introducción
  - Predicción Simple
  - Copiloto Conversacional
- Deployment frontend
- Documentación completa (README, informe técnico)
- Presentación final y video demo (10-15 min)

**Criterios de éxito:**
- Copiloto responde correctamente a ≥5 casos de uso
- Frontend deployado y accesible
- Proyecto cumple criterios de rúbrica (≥90/100)

---

## Equipo y Roles

- **Juan** - Data Engineer (limpieza, ETL, DuckDB)
- **Erick** - Data Scientist (EDA, feature engineering, análisis)
- **Julian** - Machine Learning Engineer (modelos, Chronos, LLM)
- **Dante** - Software Engineer & Scrum Master (FastAPI, Streamlit, gestión)
- **Arthur** - MLOps/SRE Engineer (infra, CI/CD, Cloud, monitoring)

---

## Métricas de Éxito

| Criterio | Meta |
|----------|------|
| RMSE en test set | < 0.205 |
| MAE en test set | < 0.046 |
| CV (%) | < 0.75% |
| Reproducibilidad | 100% con DVC |
| Code coverage | > 70% |
| Latencia API (p95) | < 500ms |
| Presupuesto GCP | ≤ $50 USD |
| Commits por persona | ≥ 15 |
| Calificación Sprint 1 | ≥ 90/100 |

---

## Arquitectura Simplificada

```
Frontend (Streamlit)
    ↓
FastAPI Backend
    ├── /predict → Chronos Model
    └── /copilot/chat → Ollama (Llama 3.2)
         ↓
    Evidently Monitoring
         ↓
    DuckDB ← Parquet (DVC)
         ↓
    GCS (Cloud Storage)
```

**Deployment:**
- Entrenamiento: Local (GPU 4070)
- API: Cloud Run (GCP)
- Frontend: Streamlit Cloud o Cloud Run
- Storage: Google Cloud Storage

---

## Metodología

**Scrum adaptado para MLOps:**
- 3 sprints de 2 semanas
- 12 hrs/semana por integrante
- Ceremonias: Planning, Daily Standups (async), Review, Retrospective
- Herramientas: Linear (gestión), GitHub (código), MLflow (tracking)

---

## Casos de Uso del Copiloto

1. **Predicción:** "¿Cuánto consumiremos mañana a las 10am con carga Medium?"
2. **What-if:** "¿Cuánto ahorraré si cambio la producción de 10am a 2am?"
3. **Explicación:** "¿Por qué hubo un pico de consumo el viernes pasado?"
4. **Optimización:** "¿Cómo puedo reducir el consumo en el turno nocturno?"
5. **Análisis:** "¿Qué tipo de carga consume más energía?"

---

**Duración total:** 6 semanas  
**Presupuesto:** $50 USD  
**Metodología:** Scrum + MLOps  
**Versión:** 1.0
