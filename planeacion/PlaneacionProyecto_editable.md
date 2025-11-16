# Plan de Proyecto: Sistema de Optimización Energética con IA para la Industria Siderúrgica

**Copiloto Inteligente para Predicción y Análisis de Consumo Energético**

---

**Equipo de Desarrollo:**
- **Juan** - Data Engineer (DE)
- **Erick** - Data Scientist (DS)
- **Julian** - Machine Learning Engineer (MLE)
- **Dante** - Software Engineer & Scrum Master (SE)
- **Arthur** - MLOps/SRE Engineer (MLOps)

**Detalles del Proyecto:**
- **Duración:** 6 semanas (3 sprints de 2 semanas)
- **Dedicación:** 12 horas/semana por integrante
- **Metodología:** Scrum adaptado para MLOps
- **Presupuesto GCP:** $50 USD total
- **Versión del Documento:** 1.0

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Introducción y Contexto](#2-introducción-y-contexto)
3. [Objetivos del Proyecto](#3-objetivos-del-proyecto)
4. [Análisis del Problema y Propuesta de Valor](#4-análisis-del-problema-y-propuesta-de-valor)
5. [Stack Tecnológico](#5-stack-tecnológico)
6. [Arquitectura de la Solución](#6-arquitectura-de-la-solución)
7. [Metodología: Scrum Adaptado para MLOps](#7-metodología-scrum-adaptado-para-mlops)
8. [Plan de Ejecución: Sprint 1](#8-plan-de-ejecución-sprint-1)
9. [Plan de Ejecución: Sprint 2](#9-plan-de-ejecución-sprint-2)
10. [Plan de Ejecución: Sprint 3](#10-plan-de-ejecución-sprint-3)
11. [Roles y Responsabilidades](#11-roles-y-responsabilidades)
12. [Gestión de Riesgos](#12-gestión-de-riesgos)
13. [Plan de Entregables](#13-plan-de-entregables)
14. [Conclusiones](#14-conclusiones)
15. [Referencias](#15-referencias)

---

## 1. Resumen Ejecutivo

Este proyecto desarrolla un **Copiloto de Inteligencia Artificial para la Optimización Energética** en la industria siderúrgica, utilizando el dataset "Steel Industry Energy Consumption" de UCI Machine Learning Repository. La solución combina modelos de Machine Learning de última generación (Foundation Models para series temporales) con una interfaz conversacional basada en IA generativa, permitiendo a operadores de planta realizar análisis predictivos y exploraciones de escenarios "what-if" en lenguaje natural.

### Innovación Principal

Mientras que los enfoques tradicionales se limitan a generar predicciones numéricas, nuestro sistema transforma datos predictivos en **inteligencia operativa accionable**, cerrando la brecha entre "saber qué consumo habrá" y "comprender cómo optimizarlo".

### Diferenciadores Técnicos

- Evaluación comparativa de Foundation Models SOTA (Chronos-T5, TimesFM) vs. métodos clásicos
- Arquitectura MLOps híbrida (local + cloud) optimizada para FinOps ($50 USD presupuesto total)
- Pipeline completamente reproducible con DVC + Git + MLflow
- Deployment dual: demo local + producción en GCP Cloud Run

### Resultados Esperados

**Meta Principal:** Superar el benchmark establecido por el modelo CUBIST (RMSE: 0.2410) en al menos un 15%, demostrando la superioridad de los Foundation Models en pronóstico de series temporales industriales.

| Criterio | Métrica | Meta |
|----------|---------|------|
| Precisión del Modelo | RMSE en test set | < 0.205 (15% mejor que CUBIST) |
| Precisión del Modelo | MAE en test set | < 0.046 |
| Precisión del Modelo | CV (%) | < 0.75% |
| Reproducibilidad | Capacidad de recrear resultados | 100% reproducible con DVC |
| Cobertura de Código | Tests unitarios | > 70% |
| Latencia de Inferencia | Tiempo de respuesta API | < 500ms p95 |
| Presupuesto | Gasto total en GCP | ≤ $50 USD |
| Colaboración | Commits por integrante | ≥ 15 commits/persona |
| Documentación | Completitud ML Canvas | 9/9 secciones completas |
| Calidad Académica | Evaluación rúbrica Sprint 1 | ≥ 90/100 puntos |

---

## 2. Introducción y Contexto

### 2.1 Contexto Industrial

La industria siderúrgica global enfrenta una crisis de sostenibilidad energética caracterizada por:

1. **Volatilidad Económica:** Los costos energéticos representan entre 20-40% de los costos operativos totales en plantas siderúrgicas, con fluctuaciones impredecibles en los precios de la electricidad.

2. **Presión Regulatoria:** Las regulaciones de emisiones de carbono (Carbon Border Adjustment Mechanism de la UE, normativas EPA en EE.UU.) imponen penalizaciones significativas por consumo energético no optimizado.

3. **Complejidad Operativa:** El consumo energético en plantas siderúrgicas depende de múltiples variables interdependientes (tipo de carga, factor de potencia, ciclos de producción) que los sistemas de gestión tradicionales no pueden optimizar eficientemente.

### 2.2 El Dataset: Steel Industry Energy Consumption

El dataset proviene de una planta siderúrgica en Corea del Sur, capturando datos de consumo energético entre enero y diciembre de 2018. Contiene **35,040 registros** con las siguientes características:

**Variables Principales:**
- `Usage_kWh`: Consumo de energía (variable objetivo)
- `Lagging_Current_Reactive_Power_kVarh`: Potencia reactiva en atraso
- `Leading_Current_Reactive_Power_kVarh`: Potencia reactiva en adelanto
- `CO2(tCO2)`: Emisiones de dióxido de carbono
- `Lagging_Current_Power_Factor`: Factor de potencia en atraso
- `Leading_Current_Power_Factor`: Factor de potencia en adelanto
- `NSM` (Number of Seconds from Midnight): Marca temporal
- `WeekStatus`: Día de semana/fin de semana
- `Day_of_week`: Día específico
- `Load_Type`: Tipo de carga (Light, Medium, Maximum)

**Problemática del Dataset "Sucio":**

Se nos proporcionó intencionalmente un dataset con errores introducidos para simular condiciones reales:
- Valores nulos y faltantes
- Outliers no realistas
- Inconsistencias en tipos de datos
- Registros duplicados
- Errores de codificación en variables categóricas

### 2.3 Estado del Arte

**Benchmark Establecido (2021):**

El paper de referencia evaluó 15 algoritmos de Machine Learning clásicos, identificando al modelo **CUBIST** como óptimo con:
- RMSE: 0.2410
- MAE: 0.0547
- CV: 0.8770%

**Evolución Tecnológica (2023-2025):**

La aparición de Foundation Models pre-entrenados para series temporales (Chronos de Amazon Science, TimesFM de Google Research) ha revolucionado el campo, ofreciendo capacidades zero-shot que superan significativamente a los modelos clásicos en benchmarks recientes.

### 2.4 La Brecha: Del Pronóstico a la Acción

El problema crítico de las soluciones predictivas actuales es el **"gap de la última milla"**: un modelo que predice "el consumo mañana será 850 kWh" no proporciona:
- ¿Por qué será ese valor?
- ¿Qué variables lo están impulsando?
- ¿Cómo puedo reducirlo mediante acciones operativas?
- ¿Qué escenarios alternativos existen?

**Nuestro proyecto cierra esta brecha mediante un Copiloto Conversacional de IA.**

---

## 3. Objetivos del Proyecto

### 3.1 Objetivo General

Desarrollar un sistema inteligente de optimización energética que combine modelos de pronóstico de series temporales de última generación con una interfaz conversacional de IA generativa, permitiendo a operadores industriales tomar decisiones informadas basadas en predicciones precisas y análisis de escenarios interactivos.

### 3.2 Objetivos Específicos

**Técnicos:**

1. Implementar un pipeline completo de ingeniería de datos que transforme el dataset "sucio" en datos limpios y analizables, documentando cada paso del proceso.
2. Realizar un Análisis Exploratorio de Datos (EDA) exhaustivo que identifique patrones temporales, correlaciones y comportamientos estacionales en el consumo energético.
3. Evaluar comparativamente al menos 4 algoritmos de Machine Learning (2 clásicos + 2 Foundation Models) para seleccionar el modelo óptimo.
4. **Superar el benchmark CUBIST en al menos 15%** en las métricas principales (RMSE < 0.205, MAE < 0.046).
5. Construir una API RESTful con FastAPI que exponga el modelo como servicio, cumpliendo estándares de producción.
6. Implementar un sistema de versionado completo (datos + código + modelos) usando DVC + Git + MLflow.
7. Desarrollar una interfaz conversacional usando un LLM open-source (Llama 3.2) que permita consultas en lenguaje natural.

**De Proceso (MLOps):**

8. Establecer un pipeline de CI/CD automatizado para entrenamiento, validación y despliegue de modelos.
9. Implementar monitoreo de model drift y data drift usando Evidently AI.
10. Mantener el presupuesto de GCP bajo $50 USD mediante estrategias de FinOps.

**Académicos:**

11. Documentar los requerimientos del proyecto usando el framework ML Canvas.
12. Demostrar trabajo colaborativo efectivo con evidencia de contribuciones de todos los integrantes en GitHub.
13. Producir documentación de grado de maestría que sirva como referencia para futuros proyectos MLOps.

### 3.3 Criterios de Éxito

El proyecto será considerado exitoso si cumple con:

| Criterio | Métrica | Meta |
|----------|---------|------|
| **Precisión del Modelo** | RMSE en test set | < 0.205 (15% mejor que CUBIST) |
| **Precisión del Modelo** | MAE en test set | < 0.046 |
| **Precisión del Modelo** | CV (%) | < 0.75% |
| **Reproducibilidad** | Capacidad de recrear resultados | 100% reproducible con DVC |
| **Cobertura de Código** | Tests unitarios | > 70% |
| **Latencia de Inferencia** | Tiempo de respuesta API | < 500ms p95 |
| **Presupuesto** | Gasto total en GCP | ≤ $50 USD |
| **Colaboración** | Commits por integrante | ≥ 15 commits/persona |
| **Documentación** | Completitud ML Canvas | 9/9 secciones completas |
| **Calidad Académica** | Evaluación rúbrica Sprint 1 | ≥ 90/100 puntos |

---

## 4. Análisis del Problema y Propuesta de Valor

### 4.1 Definición del Problema

**Problema Principal:**

Los operadores de plantas siderúrgicas carecen de herramientas que traduzcan datos de consumo energético en decisiones operativas accionables, resultando en:
- Sobre-consumo energético evitable (5-15% según estudios de la industria)
- Incapacidad para planificar cargas de trabajo en horarios de menor costo energético
- Falta de visibilidad sobre drivers de consumo en tiempo real
- Emisiones de CO2 excesivas por operación ineficiente

**Problema Secundario:**

Los modelos predictivos tradicionales generan números pero no explican:
- Causalidad (¿por qué aumentará el consumo?)
- Accionabilidad (¿qué acciones puedo tomar?)
- Exploración de escenarios (¿qué pasa si cambio X variable?)

### 4.2 Propuesta de Valor

Desarrollamos un **"Copiloto de IA para Optimización Energética"** que:

**Para Operadores de Planta:**
- Permite preguntar en lenguaje natural: *"¿Cuánto ahorraré si muevo la producción de bobinas de acero de 10am a 2am?"*
- Proporciona explicaciones causales: *"El pico de consumo del viernes se debió a una combinación de carga máxima + bajo factor de potencia"*
- Sugiere optimizaciones: *"Puedes reducir 8% del consumo ajustando el factor de potencia en el turno nocturno"*

**Para Gerentes de Energía:**
- Predicciones precisas (RMSE < 0.20) para planificación de demanda
- Cuantificación de impacto de carbono por escenario
- Reportes ejecutivos automatizados

**Para Ingenieros de Mantenimiento:**
- Detección de anomalías en patrones de consumo
- Alertas tempranas de drift en variables críticas

### 4.3 ML Canvas

A continuación, documentamos los requerimientos del proyecto usando el framework ML Canvas:

#### **1. Propuesta de Valor**
Reducir costos operativos energéticos y emisiones de CO2 en plantas siderúrgicas mediante un asistente de IA que combina predicciones precisas con análisis conversacional de escenarios, permitiendo optimización proactiva de cargas de trabajo.

#### **2. Tarea de Machine Learning**
**Tipo:** Regresión supervisada de series temporales multivariadas  
**Objetivo:** Predecir `Usage_kWh` (consumo de energía) con horizonte de 1 hora a 24 horas  
**Contexto:** Predicción continua en un entorno de producción industrial con alta variabilidad temporal

#### **3. Fuentes de Datos**

**Primaria:**
- Dataset UCI Steel Industry Energy Consumption (35,040 registros, 2018)
- Frecuencia: 15 minutos
- Formato: CSV (versión sucia para ingeniería de datos)

**Futuras (post-proyecto):**
- API de sensores IoT en tiempo real
- Datos de precios de electricidad del mercado spot
- Información meteorológica (temperatura, humedad)

#### **4. Características (Features)**

**Features Originales:**
- Potencia reactiva (lagging/leading) - kVarh
- Factor de potencia (lagging/leading) - ratio
- Emisiones CO2 - tCO2
- Tipo de carga - categórica (Light/Medium/Maximum)

**Features Ingenierizadas:**
- `NSM` (Number of Seconds from Midnight) - temporal
- `day_of_week` - categórica (Lunes-Domingo)
- `is_weekend` - binaria
- `hour` - temporal
- `cyclical_hour_sin`, `cyclical_hour_cos` - encoding cíclico
- `lag_features` - consumo en t-1, t-2, t-24 (autoregresivos)
- `rolling_mean_24h` - media móvil
- Interacciones: `load_type × hour`, `power_factor × reactive_power`

#### **5. Modelos a Evaluar**

**Baseline Clásicos:**
1. XGBoost con hyperparameter tuning
2. LightGBM con early stopping

**Foundation Models (Innovación):**
3. **Chronos-T5-Small** (Amazon Science, 250M parámetros)
   - Zero-shot evaluation
   - Fine-tuning en dataset específico
4. **TimesFM** (Google Research, 200M parámetros)
   - Zero-shot evaluation

**Meta-modelo:**
5. Ensemble ponderado de top-2 modelos

#### **6. Métricas de Evaluación**

**Primarias:**
- **RMSE** (Root Mean Squared Error): Penaliza errores grandes, crítico para peaks
- **MAE** (Mean Absolute Error): Error promedio absoluto, interpretable
- **CV(%)** (Coefficient of Variation): Error relativo normalizado

**Benchmark a Superar:**
- CUBIST (2021): RMSE 0.2410, MAE 0.0547, CV 0.8770%
- **Meta del equipo:** RMSE < 0.205, MAE < 0.046, CV < 0.75%

**Secundarias:**
- Latencia de inferencia (< 500ms p95)
- Tamaño del modelo (< 2GB para deployment eficiente)
- MAPE (Mean Absolute Percentage Error) para interpretabilidad de negocio

#### **7. Usuarios e Integración**

**Usuarios Primarios:**
- Operadores de planta (turno 24/7)
- Gerentes de energía (planificación semanal)
- Ingenieros de proceso (optimización continua)

**Integración:**
- **Sprint 1-2:** Demo local (Docker + FastAPI)
- **Sprint 3:** Aplicación web con interfaz conversacional
  - Frontend: Streamlit o Gradio
  - Backend: FastAPI + Ollama (Llama 3.2)
  - Deployment: Cloud Run (GCP)

**Casos de Uso:**
1. Predicción de consumo próximas 24 horas
2. Simulación "what-if": cambio de tipo de carga
3. Análisis retrospectivo: explicación de picos históricos
4. Recomendaciones: sugerencias de optimización

#### **8. Evaluación Offline**

**Estrategia de Validación:**
- **Train/Validation/Test Split:** 60% / 15% / 25% (respetando orden temporal)
- **Cross-validation:** Time Series Split con 5 folds
- **Backtesting:** Rolling window de 7 días para validar estabilidad

**Conjunto de Test:**
- Últimos 2 meses del dataset (8,760 registros)
- Incluye diferentes estacionalidades y tipos de carga
- Sin data leakage del conjunto de entrenamiento

#### **9. Monitoreo en Producción**

**Herramienta:** Evidently AI (open-source)

**Métricas a Monitorear:**
- **Data Drift:** Distribuciones de features (KS test, PSI)
- **Target Drift:** Distribución de `Usage_kWh` real
- **Model Performance:** RMSE/MAE calculados en ventanas móviles de 7 días
- **Prediction Drift:** Distribución de predicciones del modelo

**Alertas:**
- Drift severo en cualquier feature (PSI > 0.2) → Email a equipo MLOps
- Degradación de performance (RMSE > 0.25) → Trigger reentrenamiento automático
- Anomalías en distribución de target → Investigación manual

---

## 5. Stack Tecnológico

Esta sección detalla todas las herramientas, librerías y servicios que utilizaremos, organizados por categoría funcional. Cada elección está justificada técnica y económicamente.

### 5.1 Gestión de Proyecto y Colaboración

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Linear** | Web | Gestión de sprints, historias de usuario, tracking | Interfaz moderna, integración nativa con GitHub, superior a Jira para equipos pequeños | $0 (Plan Free) |
| **GitHub** | Cloud | Control de versiones, CI/CD, colaboración en código | Estándar de industria, GitHub Actions incluido, ilimitado para repos públicos | $0 |
| **GitHub Projects** | Cloud | Kanban board alternativo/backup | Integración perfecta con issues y PRs | $0 |
| **Notion/Google Docs** | Cloud | Documentación colaborativa, minutas de reuniones | Accesibilidad, comentarios en tiempo real | $0 |

### 5.2 Entorno de Desarrollo

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Python** | 3.11.x | Lenguaje principal | Mejor soporte de type hints, performance mejorado vs 3.10, compatible con todas las librerías ML | $0 |
| **Poetry** | 1.8+ | Gestión de dependencias y entornos virtuales | Resuelve dependencias determinísticamente, genera lock files, superior a pip/conda para reproducibilidad | $0 |
| **Visual Studio Code** | Latest | IDE principal | IntelliSense, debugging integrado, extensiones de Python/ML, colaboración Live Share | $0 |
| **Jupyter Lab** | 4.x | Notebooks para EDA y experimentación | Interfaz moderna, kernels múltiples, extensiones (debugger, git) | $0 |
| **Git** | 2.40+ | Control de versiones local | Base para DVC, hooks pre-commit | $0 |
| **Docker** | 24.x | Containerización | Ambientes reproducibles, deployment consistente | $0 |
| **Docker Compose** | 2.x | Orquestación local de servicios | Testing de arquitectura multi-contenedor | $0 |

### 5.3 Procesamiento y Análisis de Datos

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Polars** | 0.20+ | Data processing principal | 5-10x más rápido que Pandas en operaciones típicas, sintaxis expresiva, backend Rust paralelo | $0 |
| **Pandas** | 2.2+ | Compatibilidad con librerías legacy | Algunas librerías ML solo aceptan Pandas DataFrames | $0 |
| **NumPy** | 1.26+ | Operaciones numéricas vectorizadas | Base de todo el stack científico de Python | $0 |
| **DuckDB** | 0.10+ | Data warehouse local/SQL analytics | Performance de BigQuery sin costos, queries SQL sobre Parquet, perfecto para EDA | $0 |
| **PyArrow** | 15+ | I/O eficiente de Parquet | Formato columnar comprimido, interoperabilidad Polars/Pandas | $0 |

### 5.4 Visualización y EDA

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Matplotlib** | 3.8+ | Gráficos base, customización avanzada | Control total sobre estética, publicaciones académicas | $0 |
| **Seaborn** | 0.13+ | Visualizaciones estadísticas | Syntax sugar sobre Matplotlib, paletas profesionales | $0 |
| **Plotly** | 5.19+ | Gráficos interactivos | Dashboards interactivos, time series con zoom, exportable a HTML | $0 |
| **Plotly Express** | Incluido | Gráficos rápidos de alta calidad | Sintaxis simple, ideal para EDA | $0 |

### 5.5 Machine Learning - Clásico

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Scikit-learn** | 1.4+ | Preprocessing, baselines, métricas | Estándar de industria, pipelines robustos, amplia documentación | $0 |
| **XGBoost** | 2.0+ | Modelo baseline gradient boosting | SOTA en tabular data, eficiente, soporte GPU | $0 |
| **LightGBM** | 4.3+ | Modelo baseline alternativo | Más rápido que XGBoost en datasets grandes, menor uso de memoria | $0 |
| **Optuna** | 3.6+ | Hyperparameter optimization | Algoritmos modernos (TPE), integración MLflow, visualizaciones | $0 |

### 5.6 Machine Learning - Foundation Models

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Chronos (HuggingFace)** | Latest | Foundation model principal para time series | SOTA 2024-2025, zero-shot + fine-tuning, Amazon Science research | $0 |
| **TimesFM (Google)** | Latest | Foundation model alternativo | Google Research, arquitectura Transformer optimizada | $0 |
| **Transformers (HuggingFace)** | 4.38+ | Librería base para Foundation Models | Manejo de modelos pre-entrenados, tokenizers | $0 |
| **PyTorch** | 2.2+ | Framework de deep learning | Backend de Chronos/TimesFM, soporte CUDA para 4070 | $0 |
| **Accelerate (HuggingFace)** | 0.27+ | Multi-GPU training, mixed precision | Optimiza uso de GPU 4070, reduce memoria | $0 |

### 5.7 MLOps - Tracking y Versionado

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **DVC** | 3.48+ | Versionado de datos y modelos | Git para data science, remote storage en GCS, reproducibilidad | $0 |
| **MLflow** | 2.11+ | Experiment tracking, model registry | Logging automático de métricas/params, UI local, comparación de runs | $0 |
| **Weights & Biases** | Latest | Tracking alternativo (opcional) | Mejor UI que MLflow, pero requiere cuenta cloud (plan free limitado) | $0 (Free tier) |

### 5.8 MLOps - Orquestación y CI/CD

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Prefect** | 2.16+ | Workflow orchestration | Open-source, UI moderna, más simple que Airflow, triggers event-driven | $0 (Self-hosted) |
| **GitHub Actions** | Cloud | CI/CD pipelines | Incluido con GitHub, testing automático, linting, build de Docker | $0 |
| **Pre-commit** | 3.6+ | Git hooks para calidad de código | Enforce linting, formateo, tests antes de commit | $0 |
| **pytest** | 8.0+ | Testing framework | Estándar Python, fixtures, coverage reports | $0 |
| **Black** | 24+ | Code formatter | Opinionated, elimina debates de estilo | $0 |
| **Ruff** | 0.3+ | Linter ultrarrápido | Reemplaza Flake8/isort/pylint, 10-100x más rápido (Rust) | $0 |

### 5.9 MLOps - Monitoring

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Evidently AI** | 0.4+ | Data drift, model drift, performance monitoring | Open-source, reportes HTML interactivos, detección automática de drift | $0 |
| **Prometheus** | 2.50+ | Métricas de sistema (opcional) | Monitoring de APIs, recursos, alerting | $0 |
| **Grafana** | 10+ | Dashboards (opcional) | Visualización de métricas Prometheus | $0 |

### 5.10 Backend y API

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **FastAPI** | 0.110+ | Framework web para API REST | Async/await nativo, auto-generación de docs (Swagger), validación Pydantic, SOTA performance | $0 |
| **Pydantic** | 2.6+ | Validación de datos | Type safety, serialización JSON automática | $0 |
| **Uvicorn** | 0.27+ | ASGI server | Servidor async de alto performance para FastAPI | $0 |
| **Gunicorn** | 21+ | WSGI server (producción) | Multi-worker process manager para deployment | $0 |

### 5.11 IA Generativa - LLM Local

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Ollama** | 0.1+ | Runtime para LLMs locales | Corre modelos en la 4070 (Llama, Mistral), API compatible con OpenAI | $0 |
| **Llama 3.2 (3B)** | Latest | LLM para copiloto conversacional | 3B cabe en 4070 (16GB VRAM), Meta license permite uso comercial | $0 |
| **LangChain** | 0.1+ | Framework para LLM applications | Chains, agents, prompt templates, integración con Ollama | $0 |
| **LlamaIndex** | 0.10+ | RAG framework (opcional) | Si necesitamos retrieval-augmented generation | $0 |

### 5.12 Frontend (Sprint 3)

| Herramienta | Versión | Propósito | Justificación | Costo |
|-------------|---------|-----------|---------------|-------|
| **Streamlit** | 1.32+ | UI web para demo | Rápido desarrollo, widgets interactivos, Python puro | $0 |
| **Gradio** | 4.20+ | UI alternativa | Mejor para demos de ML, interfaz chat nativa | $0 |

### 5.13 Cloud - Google Cloud Platform

| Servicio | Propósito | Configuración FinOps | Costo Estimado |
|----------|-----------|----------------------|----------------|
| **Cloud Storage** | DVC remote, artefactos | Standard storage, lifecycle policy (30 días) | $0.50/mes |
| **Cloud Run** | API deployment | Scale-to-zero, min instances=0, max=2, concurrency=80 | $8-15/mes |
| **Container Registry** | Docker images | Comprimir images, cleanup automático | $0.10/mes |
| **Cloud Functions** | Triggers opcionales | 2M invocations/mes free tier | $0-2/mes |
| **Secret Manager** | API keys, credentials | Solo secrets críticos | $0.06/mes |
| **Cloud Logging** | Logs (limitado) | Retención 7 días, filtros agresivos | $0 (free tier) |
| **Cloud Monitoring** | Alertas básicas | Solo métricas críticas | $0 (free tier) |

**Total GCP Proyectado:** $10-20/mes × 1.5 meses = **$15-30 total** (buffer $50)

### 5.14 Herramientas de Productividad

| Herramienta | Propósito | Costo |
|-------------|-----------|-------|
| **Mermaid.js** | Diagramas de arquitectura en Markdown | $0 |
| **Draw.io** | Diagramas complejos | $0 |
| **Excalidraw** | Sketches rápidos | $0 |
| **Markdown** | Documentación técnica | $0 |
| **LaTeX (Overleaf)** | Documentos académicos formales (opcional) | $0 |

### 5.15 Resumen de Costos

| Categoría | Costo Total |
|-----------|-------------|
| Desarrollo Local | $0 |
| Herramientas SaaS | $0 (planes free) |
| Google Cloud Platform | $15-30 |
| Hardware (GPU ya existente) | $0 |
| **TOTAL PROYECTO** | **$15-30 USD** |

**Margen de seguridad:** $50 - $30 = **$20 USD buffer** para experimentación o overages.

---

## 6. Arquitectura de la Solución

### 6.1 Visión General

La arquitectura del proyecto sigue el principio de **"Hybrid MLOps"**: desarrollo y entrenamiento local (aprovechando la GPU 4070) con deployment opcional en cloud para demostración. Esto maximiza la eficiencia de costos mientras mantiene capacidades profesionales de producción.

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAPA DE PRESENTACIÓN                        │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  Streamlit UI    │◄────────┤   Gradio Chat    │            │
│  │ (Interactive Web)│         │  (Conversational)│            │
│  └────────┬─────────┘         └────────┬─────────┘            │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────────┐
│                    CAPA DE APLICACIÓN                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              FastAPI Backend                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │    │
│  │  │  Prediction  │  │  Copilot     │  │  Monitoring │ │    │
│  │  │  Endpoint    │  │  Endpoint    │  │  Endpoint   │ │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │    │
│  └─────────┼──────────────────┼──────────────────┼────────┘    │
└────────────┼──────────────────┼──────────────────┼─────────────┘
             │                  │                  │
             │                  │                  │
┌────────────▼──────────────────▼──────────────────▼─────────────┐
│                     CAPA DE MODELOS                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Chronos Model  │  │   Ollama Server  │  │  Evidently   │  │
│  │  (Time Series)  │  │  (Llama 3.2 3B)  │  │  (Drift Det.)│  │
│  │                 │  │                  │  │              │  │
│  │  Inference:     │  │  LLM Reasoning:  │  │  Monitoring: │  │
│  │  - Load model   │  │  - Parse query   │  │  - Drift     │  │
│  │  - Preprocess   │  │  - Orchestrate   │  │  - Alerts    │  │
│  │  - Predict      │  │  - Synthesize    │  │              │  │
│  └────────┬────────┘  └─────────┬────────┘  └──────┬───────┘  │
└───────────┼────────────────────┼──────────────────┼───────────┘
            │                    │                  │
┌───────────▼────────────────────▼──────────────────▼───────────┐
│                      CAPA DE DATOS                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   DuckDB     │  │  Parquet     │  │  MLflow Tracking   │  │
│  │  (Analytics) │  │  (Storage)   │  │    (Experiments)   │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              DVC + Git (Version Control)                 │ │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │ │
│  │   │  Data    │  │  Models  │  │  Code    │             │ │
│  │   │ Versions │  │ Registry │  │ Versions │             │ │
│  │   └──────────┘  └──────────┘  └──────────┘             │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                           │
                           │ DVC Push/Pull
                           ▼
              ┌─────────────────────────┐
              │   Google Cloud Storage  │
              │    (DVC Remote Store)   │
              └─────────────────────────┘
```

### 6.2 Componentes Detallados

#### **6.2.1 Plano de Datos**

**A. Almacenamiento Local**
- **Formato principal:** Parquet (columnar, comprimido, 80% menor que CSV)
- **Base de datos analítica:** DuckDB (queries SQL sobre Parquet, performance similar a BigQuery)
- **Versionado:** DVC con remote en Google Cloud Storage

**B. Pipeline de Datos**
```
Raw CSV (Sucio)
    ↓ [Juan: cleaning_pipeline.py]
Cleaned Parquet v1
    ↓ [DVC add + commit]
DVC Versioned (tag: data-v1.0)
    ↓ [Erick: feature_engineering.py]
Engineered Parquet v2
    ↓ [DVC add + commit]
DVC Versioned (tag: data-v2.0)
    ↓ [Train/Val/Test split]
3 Parquet files
    ↓ [DVC push]
GCS Backup
```

**C. Versionado con DVC**
```bash
# Estructura de archivos versionados
data/
  raw/
    steel_dirty.csv         # Original (no versionado)
    steel_clean.csv         # Referencia (no versionado)
  processed/
    steel_cleaned.parquet.dvc    # ← Versionado con DVC
    steel_featured.parquet.dvc   # ← Versionado con DVC
    train.parquet.dvc
    val.parquet.dvc
    test.parquet.dvc
models/
  baselines/
    xgboost_v1.pkl.dvc
    lightgbm_v1.pkl.dvc
  foundation/
    chronos_finetuned_v1.pth.dvc    # ← Modelo principal
```

#### **6.2.2 Plano de Entrenamiento**

**Entrenamiento Local (GPU 4070 - 16GB VRAM)**

Hardware specs:
- GPU: NVIDIA RTX 4070 Laptop (16GB GDDR6)
- RAM: 64GB DDR5
- Storage: NVMe SSD

Configuración de entrenamiento:
```python
# Chronos fine-tuning config
training_args = {
    "batch_size": 32,  # Ajustado para 16GB VRAM
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",  # Reduce memoria 50%
    "max_epochs": 50,
    "early_stopping_patience": 10,
    "learning_rate": 1e-4,
    "warmup_steps": 500
}
```

**Pipeline de Entrenamiento Automatizado (Prefect)**
```python
from prefect import flow, task

@task
def load_data():
    # Carga desde DuckDB
    pass

@task
def train_model(data, model_type):
    # Entrenamiento con MLflow logging
    pass

@task
def evaluate_model(model, test_data):
    # Métricas: RMSE, MAE, CV
    pass

@task
def register_model(model, metrics):
    # Si metrics > threshold, registrar en MLflow
    pass

@flow
def training_pipeline():
    data = load_data()
    model = train_model(data, "chronos")
    metrics = evaluate_model(model, test_data)
    register_model(model, metrics)
```

#### **6.2.3 Plano de Inferencia**

**A. API FastAPI**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI(title="Energy Optimization API")

class PredictionRequest(BaseModel):
    lagging_reactive_power: float
    leading_reactive_power: float
    co2: float
    lagging_power_factor: float
    leading_power_factor: float
    nsm: int
    day_of_week: str
    load_type: str

class PredictionResponse(BaseModel):
    predicted_usage_kwh: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # 1. Validar entrada
    # 2. Preprocesar features
    # 3. Cargar modelo Chronos
    # 4. Inferencia
    # 5. Post-procesamiento
    # 6. Logging en Evidently
    return response

@app.post("/copilot/chat")
async def copilot_chat(query: str):
    # 1. Enviar query a Ollama (Llama 3.2)
    # 2. LLM parsea intención
    # 3. Si requiere predicción, llamar a /predict
    # 4. LLM sintetiza respuesta en lenguaje natural
    return {"response": llm_answer}
```

**B. Deployment con Docker**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-root

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["poetry", "run", "gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
```

**C. Cloud Run Deployment (FinOps Optimizado)**
```yaml
# cloud-run-config.yaml
service: energy-optimization-api
region: us-central1
platform: managed

scaling:
  minInstances: 0          # Scale-to-zero cuando no hay tráfico
  maxInstances: 2          # Límite para evitar costos inesperados
  concurrency: 80          # Requests por instancia

resources:
  cpu: 1                   # 1 vCPU suficiente
  memory: 2Gi              # Modelo + runtime

timeout: 300s              # 5 min timeout

env:
  - name: MODEL_PATH
    value: gs://our-bucket/models/chronos_v1.pth
  - name: EVIDENTLY_ENABLED
    value: "true"
```

#### **6.2.4 Plano de Monitoreo**

**Evidently AI - Data Drift Detection**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

# Generar reporte de drift semanal
report = Report(metrics=[
    DataDriftPreset(),
    RegressionPreset()
])

report.run(
    reference_data=train_data,  # Datos de entrenamiento
    current_data=production_data  # Últimos 7 días de producción
)

report.save_html("reports/drift_report_week_12.html")

# Alerting
if report.as_dict()["metrics"][0]["result"]["dataset_drift"]:
    send_alert("Data drift detected! Review report.")
```

### 6.3 Flujo End-to-End

**Caso de Uso: Predicción de Consumo**

1. **Usuario:** Operador abre UI de Streamlit
2. **UI → Backend:** POST request a `/predict` con features
3. **Backend → Modelo:** Carga Chronos desde disco (cache en memoria)
4. **Modelo:** Inferencia en CPU (~50ms)
5. **Backend → UI:** Response con predicción + intervalo de confianza
6. **Background:** Log de predicción a Evidently para drift monitoring

**Caso de Uso: Consulta Conversacional**

1. **Usuario:** "¿Cuánto ahorraré si cambio carga a Maximum en horario nocturno?"
2. **UI → Backend:** POST a `/copilot/chat`
3. **Backend → Ollama:** Query procesada por Llama 3.2
4. **LLM:** Identifica necesidad de 2 predicciones (escenario actual vs. modificado)
5. **LLM → Backend:** Solicita 2 llamadas a `/predict` internamente
6. **Backend → LLM:** Retorna 2 predicciones
7. **LLM:** Calcula diferencia, traduce a respuesta en español
8. **Backend → UI:** "Cambiar a carga Maximum en horario nocturno (00:00-06:00) ahorraría aproximadamente 127 kWh (15% de reducción), equivalente a 0.08 tCO2 menos de emisiones."

---

## 7. Metodología: Scrum Adaptado para MLOps

### 7.1 Principios Scrum en Contexto MLOps

Adoptamos Scrum con adaptaciones específicas para proyectos de Machine Learning, reconociendo que:

1. **La incertidumbre es inherente:** Los experimentos ML pueden fallar; el backlog debe ser flexible.
2. **La "Definition of Done" es compleja:** Un modelo "terminado" requiere métricas, reproducibilidad y documentación.
3. **Las dependencias técnicas son fuertes:** Data engineering debe preceder a model training.

### 7.2 Estructura del Proyecto

**Duración total:** 6 semanas (1.5 meses)  
**Sprints:** 3 sprints de 2 semanas cada uno  
**Esfuerzo por sprint:** 12 hrs/semana × 5 personas = 60 hrs/persona × 2 semanas = **120 person-hours por sprint**

### 7.3 Ceremonias Scrum

#### **Sprint Planning (Lunes, inicio de sprint - 2 horas)**
- **Participantes:** Todo el equipo
- **Agenda:**
  1. Review del objetivo del sprint (20 min)
  2. Product Owner (Dante) presenta backlog priorizado (30 min)
  3. Equipo estima historias de usuario (Planning Poker - 40 min)
  4. Compromisos finales y assignment (30 min)
- **Resultado:** Sprint backlog en Linear con tareas asignadas

#### **Daily Standups (Martes-Viernes, 15 minutos - asíncrono en Discord/Slack)**
- Cada miembro responde:
  1. ¿Qué hice ayer?
  2. ¿Qué haré hoy?
  3. ¿Tengo bloqueadores?
- **Formato:** Mensaje de texto (timezone-friendly)
- **Escalation:** Bloqueadores críticos → reunión sync de 30 min

#### **Sprint Review (Último viernes del sprint - 1.5 horas)**
- **Agenda:**
  1. Demo de funcionalidades completadas (40 min)
  2. Review de métricas del sprint (20 min)
  3. Feedback y ajustes al backlog (30 min)
- **Artefacto:** Video de demo para stakeholders

#### **Sprint Retrospective (Último viernes del sprint - 1 hora)**
- **Framework:** Start/Stop/Continue
- **Temas:**
  - ¿Qué funcionó bien técnicamente?
  - ¿Qué procesos mejorar?
  - ¿Cómo mejorar la colaboración?
- **Artefacto:** Action items para el próximo sprint

### 7.4 Herramientas de Gestión

#### **Linear - Configuración**

**Estructura de proyectos:**
```
Proyecto: Energy Optimization Copilot
  ├── Milestone 1: Sprint 1 - Foundation & Baseline
  ├── Milestone 2: Sprint 2 - MLOps Automation
  └── Milestone 3: Sprint 3 - Copilot & Deployment

Estados:
  - Backlog
  - Todo (Sprint actual)
  - In Progress
  - In Review (PR abierto)
  - Done
  - Blocked

Prioridades:
  - P0: Crítico (bloquea otros tasks)
  - P1: Alto (requerido para sprint goal)
  - P2: Medio (nice-to-have)
  - P3: Bajo (backlog futuro)

Labels:
  - type:data-engineering
  - type:model-training
  - type:mlops
  - type:documentation
  - type:bug
  - sprint:1, sprint:2, sprint:3
```

#### **GitHub - Workflow**

**Branch strategy:**
```
main (producción)
  ├── develop (integración continua)
      ├── feature/EP-001-project-setup
      ├── feature/EP-002-data-cleaning
      ├── feature/EP-005-chronos-model
      └── hotfix/fix-parquet-schema
```

**Pull Request Template:**
```markdown
## Description
[Descripción clara del cambio]

## Related Issue
Closes #EP-XXX

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows style guidelines (Black, Ruff)
- [ ] Self-review completed
- [ ] Tests added/updated (pytest)
- [ ] Documentation updated
- [ ] No new warnings
- [ ] DVC files updated if data/model changed

## Testing
[Cómo testeaste los cambios]

## Screenshots (if applicable)
```

**CI/CD con GitHub Actions:**
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  pull_request:
    branches: [develop, main]
  push:
    branches: [develop, main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Poetry
        run: pip install poetry
      - name: Install dependencies
        run: poetry install
      - name: Run Ruff
        run: poetry run ruff check .
      - name: Run Black
        run: poetry run black --check .
  
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Install dependencies
        run: poetry install
      - name: Run pytest
        run: poetry run pytest --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 7.5 Definition of Done (DoD)

Una historia de usuario está "Done" cuando:

**Para Data Engineering:**
- [ ] Script ejecuta sin errores
- [ ] Output validado contra dataset de referencia (si aplica)
- [ ] Archivo Parquet creado y versionado con DVC
- [ ] Documentación en docstring (Google style)
- [ ] Notebook de EDA (si aplica) con conclusiones escritas

**Para Model Training:**
- [ ] Modelo entrena exitosamente
- [ ] Métricas loggeadas en MLflow
- [ ] Modelo serializado y versionado con DVC
- [ ] Comparación con baseline documentada
- [ ] Hyperparameters registrados

**Para MLOps:**
- [ ] Pipeline ejecuta end-to-end
- [ ] Tests unitarios con >70% coverage
- [ ] Dockerfile funcional (si aplica)
- [ ] Documentación de deployment

**Para Documentación:**
- [ ] Markdown formateado correctamente
- [ ] Diagramas claros (Mermaid/Draw.io)
- [ ] Review de al menos 1 compañero
- [ ] Merged a develop branch

---

## 8. Plan de Ejecución: Sprint 1

**Duración:** Semanas 1-2  
**Objetivo:** Establecer pipeline completo de datos a modelo, superando benchmark CUBIST  
**Sprint Goal Statement:** "Al final del Sprint 1, tendremos un modelo Chronos fine-tuned que supera CUBIST en 15%, con pipeline reproducible y documentado en ML Canvas."

### 8.1 Objetivos Específicos del Sprint 1

1. ✅ Configurar infraestructura MLOps (Poetry, DVC, MLflow, Prefect)
2. ✅ Limpiar dataset "sucio" y validar contra referencia limpia (>99% match)
3. ✅ Realizar EDA exhaustivo con visualizaciones de calidad publicable
4. ✅ Entrenar 4 modelos: XGBoost, LightGBM, Chronos (zero-shot), Chronos (fine-tuned)
5. ✅ Lograr RMSE < 0.205 con el mejor modelo
6. ✅ Documentar ML Canvas completo
7. ✅ Producir presentación ejecutiva y video del equipo

### 8.2 Backlog del Sprint 1

#### **Epic 1: Project Setup & Infrastructure (Arthur - 12 hrs)**

**US-001: Configurar Poetry y estructura del proyecto**
- **Como** miembro del equipo
- **Quiero** un entorno de desarrollo reproducible con Poetry
- **Para que** todos trabajemos con las mismas versiones de librerías
- **Criterios de Aceptación:**
  - [x] Repositorio GitHub creado con estructura de carpetas
  - [x] `pyproject.toml` configurado con dependencias principales
  - [x] `poetry.lock` generado
  - [x] README.md con instrucciones de setup
  - [x] `.gitignore` configurado
- **Estimación:** 2 story points (3 hrs)
- **Tareas técnicas:**
  1. `git init` y crear repo en GitHub
  2. `poetry init` con Python 3.11
  3. Agregar deps: `polars`, `scikit-learn`, `xgboost`, `mlflow`, `dvc`, `fastapi`
  4. Crear estructura:
     ```
     energy-optimization/
     ├── data/
     │   ├── raw/
     │   ├── processed/
     │   └── external/
     ├── notebooks/
     ├── src/
     │   ├── data/
     │   ├── models/
     │   ├── features/
     │   └── api/
     ├── tests/
     ├── models/
     ├── reports/
     ├── .dvc/
     ├── pyproject.toml
     └── README.md
     ```
---

**US-001b: Aplicar Cookiecutter Data Science**
- Como: Miembro del equipo
- Quiero: Estructura estandarizada con Cookiecutter
- Para que: El proyecto siga convenciones de industria
- Criterios:
  - [x] `cookiecutter data-science` aplicado
  - [x] Adaptaciones documentadas en STRUCTURE.md
  - [x] Carpetas estándar: data/, models/, notebooks/, src/, tests/
- Estimación: 1 story point (1 hr)

**US-002: Configurar DVC con Google Cloud Storage**
- **Como** data scientist
- **Quiero** versionar datasets y modelos en la nube
- **Para que** podamos reproducir experimentos
- **Criterios de Aceptación:**
  - [x] DVC inicializado (`dvc init`)
  - [x] Remote storage configurado en GCS bucket
  - [x] Primer archivo testeado: `dvc add`, `dvc push`, `dvc pull`
  - [x] Documentación en README de cómo usar DVC
- **Estimación:** 3 story points (4 hrs)
- **Tareas técnicas:**
  1. Crear GCS bucket: `gsutil mb gs://energy-opt-dvc-remote`
  2. `dvc remote add -d gcs gs://energy-opt-dvc-remote`
  3. Configurar service account key
  4. Test workflow completo

**US-003: Setup Docker y FastAPI boilerplate**
- **Como** MLOps engineer
- **Quiero** un contenedor base para la API
- **Para que** podamos deployar consistentemente
- **Criterios de Aceptación:**
  - [x] `Dockerfile` multi-stage funcional
  - [x] `docker-compose.yml` para desarrollo local
  - [x] FastAPI app responde en `/health`
  - [x] Build exitoso < 2 min
- **Estimación:** 2 story points (3 hrs)

**US-004: Configurar MLflow y Prefect local**
- **Como** ML engineer
- **Quiero** tracking de experimentos y orquestación
- **Para que** podamos comparar modelos sistemáticamente
- **Criterios de Aceptación:**
  - [x] MLflow server corriendo local (`mlflow ui` accesible)
  - [x] Prefect Cloud free tier configurado O Prefect server local
  - [x] Primer flow de ejemplo ejecutando
  - [x] Logging de métricas dummy funcionando
- **Estimación:** 2 story points (2 hrs)

---

#### **Epic 2: Data Engineering (Juan - 12 hrs)**

**US-005: Análisis del dataset sucio**
- **Como** data engineer
- **Quiero** identificar todos los problemas de calidad de datos
- **Para que** planifiquemos la limpieza correctamente
- **Criterios de Aceptación:**
  - [x] Notebook `00_data_profiling.ipynb` completado
  - [x] Reporte `data_quality_report.md` con:
    - Conteo de nulos por columna
    - Outliers detectados (método IQR y Z-score)
    - Inconsistencias de tipos de datos
    - Duplicados
    - Rangos anómalos
  - [x] Comparación con dataset de referencia limpio
- **Estimación:** 3 story points (4 hrs)
- **Ejemplo de reporte esperado:**
  ```markdown
  ## Data Quality Issues
  
  ### 1. Missing Values
  | Column | Nulls | % of Total |
  |--------|-------|------------|
  | Lagging_Current_Reactive_Power_kVarh | 342 | 0.98% |
  | Leading_Current_Power_Factor | 127 | 0.36% |
  
  ### 2. Outliers Detected
  - Usage_kWh: 47 valores > 3 std (método Z-score)
  - CO2: 12 valores negativos (imposibles físicamente)
  
  ### 3. Type Inconsistencies
  - Load_Type: 15 registros con valores "Médium" (typo de "Medium")
  - NSM: 8 registros con valores no numéricos
  ```

**US-006: Pipeline de limpieza de datos**
- **Como** data engineer
- **Quiero** un script reproducible que limpie el dataset
- **Para que** obtengamos un dataset listo para EDA
- **Criterios de Aceptación:**
  - [x] Script `src/data/clean_data.py` completado
  - [x] Output: `data/processed/steel_cleaned.parquet`
  - [x] Validación: >99% match con dataset de referencia
  - [x] Versionado con DVC (tag: `data-v1.0`)
  - [x] Tests unitarios para funciones de limpieza
- **Estimación:** 5 story points (6 hrs)
- **Código esperado (estructura):**
  ```python
  import polars as pl
  
  def load_raw_data(path: str) -> pl.DataFrame:
      """Carga CSV sucio."""
      return pl.read_csv(path)
  
  def handle_missing_values(df: pl.DataFrame) -> pl.DataFrame:
      """Imputa nulos con mediana para numéricos."""
      numeric_cols = df.select(pl.col(pl.Float64)).columns
      for col in numeric_cols:
          median = df[col].median()
          df = df.with_columns(
              pl.col(col).fill_null(median)
          )
      return df
  
  def remove_outliers(df: pl.DataFrame, columns: list, method='iqr') -> pl.DataFrame:
      """Remueve outliers usando IQR."""
      # Implementación IQR
      pass
  
  def fix_categorical_errors(df: pl.DataFrame) -> pl.DataFrame:
      """Corrige typos en Load_Type."""
      return df.with_columns(
          pl.col('Load_Type').str.replace('Médium', 'Medium')
      )
  
  def clean_pipeline(raw_path: str, output_path: str):
      df = load_raw_data(raw_path)
      df = handle_missing_values(df)
      df = remove_outliers(df, ['Usage_kWh', 'CO2'])
      df = fix_categorical_errors(df)
      df.write_parquet(output_path)
  ```

**US-007: Carga a DuckDB**
- **Como** data scientist
- **Quiero** datos en DuckDB para queries SQL
- **Para que** pueda explorar interactivamente
- **Criterios de Aceptación:**
  - [x] Script `src/data/load_to_duckdb.py`
  - [x] Base de datos `data/steel.duckdb` creada
  - [x] Queries de ejemplo documentadas
- **Estimación:** 1 story point (2 hrs)

---

#### **Epic 3: Exploratory Data Analysis (Erick - 12 hrs)**

**US-008: EDA exhaustivo con visualizaciones**
- **Como** data scientist
- **Quiero** entender profundamente los datos
- **Para que** pueda diseñar mejores features
- **Criterios de Aceptación:**
  - [x] Notebook `01_EDA.ipynb` con:
    - Estadísticas descriptivas completas
    - Distribución de cada variable (histogramas + KDE)
    - Correlation matrix + heatmap
    - Pairplots de variables clave
    - Boxplots por Load_Type
  - [x] Sección de conclusiones escritas
  - [x] Exportar figuras a `reports/figures/`
- **Estimación:** 5 story points (6 hrs)
- **Visualizaciones requeridas:**
  1. Distribución de `Usage_kWh` (target)
  2. Correlation heatmap (todas las variables)
  3. Consumo por hora del día (line plot)
  4. Consumo por día de semana (box plot)
  5. Scatter: CO2 vs Usage_kWh
  6. Scatter matrix: top 5 features correlacionadas

**US-009: Análisis de series temporales**
- **Como** data scientist
- **Quiero** descomponer la serie temporal
- **Para que** entienda tendencias y estacionalidad
- **Criterios de Aceptación:**
  - [x] Decomposición STL (Seasonal-Trend-Loess)
  - [x] ACF/PACF plots
  - [x] Análisis de estacionalidad por Load_Type
  - [x] Documentación de hallazgos
- **Estimación:** 3 story points (4 hrs)
- **Código esperado:**
  ```python
  from statsmodels.tsa.seasonal import STL
  import matplotlib.pyplot as plt
  
  # Preparar serie temporal
  ts = df.sort('NSM').select(['NSM', 'Usage_kWh']).to_pandas()
  ts.set_index('NSM', inplace=True)
  
  # Decomposición STL
  stl = STL(ts['Usage_kWh'], seasonal=13)  # 13 = periodicidad diaria
  result = stl.fit()
  
  # Plot
  fig, axes = plt.subplots(4, 1, figsize=(12, 10))
  result.observed.plot(ax=axes[0], title='Original')
  result.trend.plot(ax=axes[1], title='Trend')
  result.seasonal.plot(ax=axes[2], title='Seasonal')
  result.resid.plot(ax=axes[3], title='Residual')
  plt.tight_layout()
  plt.savefig('reports/figures/stl_decomposition.png')
  ```

**US-010: Feature importance preliminar**
- **Como** data scientist
- **Quiero** saber qué features son más predictivas
- **Para que** priorice en feature engineering
- **Criterios de Aceptación:**
  - [x] Mutual information scores calculados
  - [x] Gráfico de barras con top 10 features
  - [x] Correlación Pearson con target
- **Estimación:** 2 story points (2 hrs)

---

#### **Epic 4: Feature Engineering (Erick + Julian - 6 hrs c/u)**

**US-011: Crear features temporales**
- **Como** ML engineer
- **Quiero** features que capturen patrones temporales
- **Para que** el modelo aprenda estacionalidad
- **Criterios de Aceptación:**
  - [x] Script `src/features/build_features.py`
  - [x] Features creados:
    - `hour` (0-23)
    - `day_of_week` (0-6)
    - `is_weekend` (boolean)
    - `cyclical_hour_sin`, `cyclical_hour_cos`
    - `cyclical_day_sin`, `cyclical_day_cos`
  - [x] Output: `data/processed/steel_featured.parquet`
  - [x] Versionado DVC (tag: `data-v2.0`)
- **Estimación:** 2 story points (3 hrs)
- **Código esperado:**
  ```python
  import polars as pl
  import numpy as np
  
  def create_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
      # NSM to datetime
      df = df.with_columns(
          (pl.col('NSM').cast(pl.Float64) / 60).alias('minutes_from_midnight')
      )
      
      # Cyclical encoding
      df = df.with_columns([
          pl.col('hour').mul(2 * np.pi / 24).sin().alias('hour_sin'),
          pl.col('hour').mul(2 * np.pi / 24).cos().alias('hour_cos'),
          pl.col('day_of_week').mul(2 * np.pi / 7).sin().alias('day_sin'),
          pl.col('day_of_week').mul(2 * np.pi / 7).cos().alias('day_cos')
      ])
      
      return df
  ```

**US-012: Scaling y encoding**
- **Como** ML engineer
- **Quiero** features normalizadas y categóricas codificadas
- **Para que** los modelos entrenen correctamente
- **Criterios de Aceptación:**
  - [x] StandardScaler para features numéricos
  - [x] OneHotEncoder para `Load_Type`
  - [x] Pipeline sklearn serializado
  - [x] Transformación aplicada a train/val/test
- **Estimación:** 2 story points (3 hrs)

---

#### **Epic 5: Model Training & Evaluation (Julian - 12 hrs)**

**US-013: Baseline XGBoost**
- **Como** ML engineer
- **Quiero** entrenar un modelo baseline robusto
- **Para que** tengamos un punto de comparación interno
- **Criterios de Aceptación:**
  - [x] Modelo XGBoost entrenado
  - [x] Hyperparameter tuning con Optuna (100 trials)
  - [x] Cross-validation 5-fold
  - [x] Métricas loggeadas en MLflow
  - [x] Feature importance exportado
  - [x] Modelo serializado: `models/baselines/xgboost_v1.pkl`
- **Estimación:** 3 story points (4 hrs)
- **Código esperado:**
  ```python
  import xgboost as xgb
  import optuna
  import mlflow
  
  def objective(trial):
      params = {
          'max_depth': trial.suggest_int('max_depth', 3, 10),
          'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
          'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
          'subsample': trial.suggest_float('subsample', 0.6, 1.0),
          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
      }
      
      model = xgb.XGBRegressor(**params, random_state=42)
      cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_root_mean_squared_error')
      return -cv_scores.mean()
  
  # Optimization
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=100)
  
  # Train final model
  best_params = study.best_params
  with mlflow.start_run(run_name="xgboost_baseline"):
      mlflow.log_params(best_params)
      
      model = xgb.XGBRegressor(**best_params, random_state=42)
      model.fit(X_train, y_train)
      
      y_pred = model.predict(X_test)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      mae = mean_absolute_error(y_test, y_pred)
      cv = (rmse / y_test.mean()) * 100
      
      mlflow.log_metrics({"rmse": rmse, "mae": mae, "cv_percent": cv})
      mlflow.sklearn.log_model(model, "model")
  ```

**US-014: Chronos-T5 - Foundation Model**
- **Como** ML engineer
- **Quiero** implementar Chronos para aprovechar SOTA
- **Para que** superemos el benchmark CUBIST
- **Criterios de Aceptación:**
  - [x] Modelo Chronos-T5-Small cargado desde HuggingFace
  - [x] **Zero-shot evaluation** primero (sin fine-tuning)
  - [x] **Fine-tuning** en dataset usando 4070 GPU
  - [x] Comparación zero-shot vs fine-tuned
  - [x] Métricas loggeadas en MLflow (ambas versiones)
  - [x] Checkpoint guardado: `models/foundation/chronos_finetuned_v1.pth`
  - [x] **RMSE < 0.205** (criterio de éxito)
- **Estimación:** 8 story points (8 hrs)
- **Implementación esperada:**
  ```python
  from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
  import torch
  
  # Cargar modelo pre-entrenado
  model_name = "amazon/chronos-t5-small"
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  # PASO 1: Zero-shot evaluation
  with mlflow.start_run(run_name="chronos_zeroshot"):
      model.eval()
      predictions = []
      for batch in test_loader:
          with torch.no_grad():
              outputs = model.generate(batch['input_ids'])
          predictions.extend(outputs)
      
      rmse_zeroshot = compute_rmse(predictions, y_test)
      mlflow.log_metric("rmse", rmse_zeroshot)
  
  # PASO 2: Fine-tuning
  training_args = TrainingArguments(
      output_dir="./chronos_finetuned",
      num_train_epochs=50,
      per_device_train_batch_size=32,
      per_device_eval_batch_size=64,
      gradient_accumulation_steps=4,
      fp16=True,  # Mixed precision para 4070
      learning_rate=1e-4,
      warmup_steps=500,
      logging_steps=100,
      evaluation_strategy="steps",
      eval_steps=500,
      save_strategy="steps",
      save_steps=1000,
      load_best_model_at_end=True,
      metric_for_best_model="eval_loss",
      early_stopping_patience=10
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      tokenizer=tokenizer
  )
  
  with mlflow.start_run(run_name="chronos_finetuned"):
      trainer.train()
      
      # Evaluation
      predictions = trainer.predict(test_dataset)
      rmse_finetuned = compute_rmse(predictions.predictions, y_test)
      mae = compute_mae(predictions.predictions, y_test)
      cv = (rmse_finetuned / y_test.mean()) * 100
      
      mlflow.log_metrics({
          "rmse": rmse_finetuned,
          "mae": mae,
          "cv_percent": cv,
          "improvement_over_cubist": ((0.2410 - rmse_finetuned) / 0.2410) * 100
      })
      
      # Save model
      model.save_pretrained("models/foundation/chronos_finetuned_v1")
  ```

---

#### **Epic 6: Documentation & Presentation (Dante + Erick - 6 hrs c/u)**

**US-015: ML Canvas completo**
- **Como** project lead
- **Quiero** documentar requerimientos en ML Canvas
- **Para que** tengamos un framework compartido
- **Criterios de Aceptación:**
  - [x] Documento `docs/ML_Canvas.md` con 9 secciones completas
  - [x] Validado por todo el equipo
  - [x] Exportado a PDF
- **Estimación:** 3 story points (4 hrs)
- **Template a seguir:** (ver sección 4.3 de este documento)

**US-016: Presentación ejecutiva Sprint 1**
- **Como** project lead
- **Quiero** comunicar resultados a stakeholders
- **Para que** demuestren nuestro progreso
- **Criterios de Aceptación:**
  - [x] PDF con 10-12 slides:
    1. Portada + equipo
    2. Contexto del problema
    3. Dataset y proceso de limpieza
    4. EDA - hallazgos clave (3 slides)
    5. Arquitectura técnica
    6. Comparación de modelos (tabla)
    7. Resultados vs CUBIST
    8. Próximos pasos
  - [x] Diseño profesional (template Canva/PowerPoint)
  - [x] Gráficos exportados de notebooks
- **Estimación:** 3 story points (4 hrs)

**US-017: Video explicativo del equipo**
- **Como** equipo
- **Quiero** grabar un video de 7-10 min
- **Para que** presentemos nuestro trabajo
- **Criterios de Aceptación:**
  - [x] Guión escrito y ensayado
  - [x] Video grabado (Zoom/Google Meet)
  - [x] Estructura:
    - Intro (Dante, 1 min)
    - Data engineering (Juan, 2 min)
    - EDA (Erick, 2 min)
    - Models (Julian, 3 min)
    - Infra (Arthur, 1 min)
    - Conclusiones (Todos, 1 min)
  - [x] Editado (cortes básicos)
  - [x] Subido a YouTube/Drive
  - [x] Link en PDF de presentación
- **Estimación:** 5 story points (6 hrs - incluye preparación y edición)

---

### 8.3 Estimación Total del Sprint 1

| Epic | Person-Hours | % del Sprint |
|------|--------------|--------------|
| EP-1: Infrastructure | 12 | 10% |
| EP-2: Data Engineering | 12 | 10% |
| EP-3: EDA | 12 | 10% |
| EP-4: Feature Engineering | 12 | 10% |
| EP-5: Model Training | 12 | 10% |
| EP-6: Documentation | 12 | 10% |
| **Total** | **72** | **60%** |

**Buffer:** 48 person-hours (40%) para:
- Debugging inesperado
- Re-entrenamiento si métricas no se cumplen
- Reuniones de sincronización
- Code reviews

### 8.4 Riesgos del Sprint 1

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Chronos no supera CUBIST en primera iteración | Media | Alto | Plan B: Ensemble XGBoost+TimesFM, más feature engineering |
| GPU 4070 insuficiente para fine-tuning | Baja | Medio | Reducir batch size, usar gradient accumulation |
| Dataset de referencia no disponible para validación | Baja | Medio | Validar con métricas estadísticas (distribuciones) |
| Conflictos de merge en GitHub | Media | Bajo | PRs pequeños, reviews diarias |
| Miembro del equipo no disponible temporalmente | Media | Medio | Pair programming, documentación clara |

### 8.5 Métricas de Éxito del Sprint 1

**Cuantitativas:**
- [ ] RMSE del mejor modelo < 0.205 ✅
- [ ] MAE del mejor modelo < 0.046 ✅
- [ ] Match con dataset referencia > 99% ✅
- [ ] 100% de historias de usuario completadas
- [ ] Velocity: ≥ 30 story points

**Cualitativas:**
- [ ] ML Canvas aprobado por equipo
- [ ] Código pasa CI/CD (linting, tests)
- [ ] Documentación clara en README
- [ ] Video recibido positivamente

### 8.6 Tabla de Comparación Esperada

| Modelo | RMSE | MAE | CV (%) | Training Time | Inference (ms/sample) | Superó CUBIST? |
|--------|------|-----|--------|---------------|----------------------|----------------|
| CUBIST (benchmark) | 0.2410 | 0.0547 | 0.8770 | N/A | N/A | - |
| XGBoost (baseline) | ~0.22 | ~0.050 | ~0.82 | 15 min | 2 | ⚠️ Marginal |
| Chronos-T5 (zero-shot) | ~0.21 | ~0.048 | ~0.78 | 0 | 50 | ✅ SÍ |
| **Chronos-T5 (fine-tuned)** | **~0.18** | **~0.042** | **~0.67** | 2 hrs (4070) | 50 | ✅✅ SUPERADO |
| TimesFM (zero-shot) | ~0.20 | ~0.046 | ~0.74 | 0 | 40 | ✅ SÍ |

---

## 9. Plan de Ejecución: Sprint 2

**Duración:** Semanas 3-4  
**Objetivo:** Automatizar pipeline MLOps, deployar API en producción, implementar monitoreo  
**Sprint Goal Statement:** "Al final del Sprint 2, tendremos una API RESTful deployada en Cloud Run con CI/CD automatizado y monitoreo de drift operando."

### 9.1 Objetivos Específicos del Sprint 2

1. ✅ Construir pipeline de entrenamiento automatizado con Prefect
2. ✅ Desarrollar API FastAPI completa con endpoints de predicción
3. ✅ Implementar tests unitarios y de integración (>70% coverage)
4. ✅ Deployar en Cloud Run con configuración FinOps optimizada
5. ✅ Configurar CI/CD con GitHub Actions
6. ✅ Implementar monitoreo con Evidently AI
7. ✅ Documentar API con OpenAPI/Swagger

### 9.2 Backlog del Sprint 2

#### **Epic 7: Automated ML Pipeline (Arthur + Julian - 8 hrs c/u)**

**US-018: Pipeline de entrenamiento con Prefect**
- **Como** MLOps engineer
- **Quiero** automatizar todo el proceso de entrenamiento
- **Para que** pueda re-entrenar modelos con un comando
- **Criterios de Aceptación:**
  - [x] Flow de Prefect que ejecuta:
    1. Load data desde DuckDB
    2. Preprocessing (scaling, encoding)
    3. Training (model configurable)
    4. Evaluation en val set
    5. Si performance > threshold → registrar en MLflow
    6. Save artifacts con DVC
  - [x] Parametrizable (model_type, hyperparams)
  - [x] Logs detallados
  - [x] Manejo de errores robusto
- **Estimación:** 5 story points (6 hrs)
- **Código esperado:**
  ```python
  from prefect import flow, task
  from prefect.task_runners import SequentialTaskRunner
  import mlflow
  
  @task
  def load_data_task():
      import duckdb
      conn = duckdb.connect('data/steel.duckdb')
      df = conn.execute("SELECT * FROM steel_featured").fetchdf()
      return train_test_split(df, test_size=0.25, random_state=42)
  
  @task
  def preprocess_task(X_train, X_test):
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      return X_train_scaled, X_test_scaled, scaler
  
  @task
  def train_model_task(X_train, y_train, model_type='xgboost', params=None):
      if model_type == 'xgboost':
          model = xgb.XGBRegressor(**params)
      elif model_type == 'chronos':
          model = load_chronos_model()
      
      model.fit(X_train, y_train)
      return model
  
  @task
  def evaluate_model_task(model, X_test, y_test):
      y_pred = model.predict(X_test)
      metrics = {
          'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
          'mae': mean_absolute_error(y_test, y_pred),
          'r2': r2_score(y_test, y_pred)
      }
      return metrics
  
  @task
  def register_model_task(model, metrics, threshold_rmse=0.21):
      if metrics['rmse'] < threshold_rmse:
          with mlflow.start_run():
              mlflow.log_metrics(metrics)
              mlflow.sklearn.log_model(model, "model")
              # DVC add
              os.system("dvc add models/production/best_model.pkl")
              os.system("dvc push")
          return True
      return False
  
  @flow(name="Training Pipeline", task_runner=SequentialTaskRunner())
  def training_pipeline(model_type='xgboost', params=None):
      # Load
      X_train, X_test, y_train, y_test = load_data_task()
      
      # Preprocess
      X_train_scaled, X_test_scaled, scaler = preprocess_task(X_train, X_test)
      
      # Train
      model = train_model_task(X_train_scaled, y_train, model_type, params)
      
      # Evaluate
      metrics = evaluate_model_task(model, X_test_scaled, y_test)
      
      # Register if good
      registered = register_model_task(model, metrics)
      
      if registered:
          print(f"✅ Model registered! RMSE: {metrics['rmse']:.4f}")
      else:
          print(f"❌ Model below threshold. RMSE: {metrics['rmse']:.4f}")
      
      return metrics
  ```

**US-019: Scheduled retraining (opcional)**
- **Como** MLOps engineer
- **Quiero** re-entrenamiento automático semanal
- **Para que** el modelo se mantenga actualizado
- **Criterios de Aceptación:**
  - [x] Deployment de Prefect en Prefect Cloud (free tier)
  - [x] Schedule configurado: cada domingo 3am
  - [x] Notificaciones por email si falla
- **Estimación:** 2 story points (2 hrs)

---

#### **Epic 8: API Development (Dante + Julian - 8 hrs c/u)**

**US-020: FastAPI endpoints principales**
- **Como** desarrollador
- **Quiero** una API RESTful completa
- **Para que** podamos consumir el modelo desde frontend
- **Criterios de Aceptación:**
  - [x] Endpoints implementados:
    - `POST /predict` - predicción individual
    - `POST /predict/batch` - predicción batch
    - `GET /health` - health check
    - `GET /model/info` - metadata del modelo
    - `GET /model/metrics` - métricas actuales
  - [x] Validación con Pydantic
  - [x] Manejo de errores HTTP
  - [x] Logging estructurado
- **Estimación:** 5 story points (6 hrs)
- **Código esperado:**
  ```python
  from fastapi import FastAPI, HTTPException
  from pydantic import BaseModel, Field, validator
  from typing import List, Optional
  import joblib
  import numpy as np
  from datetime import datetime
  
  app = FastAPI(
      title="Energy Optimization API",
      description="Predict energy consumption for steel industry",
      version="1.0.0"
  )
  
  # Load model at startup
  model = None
  scaler = None
  
  @app.on_event("startup")
  async def load_model():
      global model, scaler
      model = joblib.load("models/production/chronos_v1.pkl")
      scaler = joblib.load("models/production/scaler.pkl")
  
  class PredictionRequest(BaseModel):
      lagging_reactive_power: float = Field(..., ge=0)
      leading_reactive_power: float = Field(..., ge=0)
      co2: float = Field(..., ge=0)
      lagging_power_factor: float = Field(..., ge=0, le=1)
      leading_power_factor: float = Field(..., ge=0, le=1)
      nsm: int = Field(..., ge=0, le=86400)
      day_of_week: int = Field(..., ge=0, le=6)
      load_type: str
      
      @validator('load_type')
      def validate_load_type(cls, v):
          if v not in ['Light', 'Medium', 'Maximum']:
              raise ValueError('load_type must be Light, Medium, or Maximum')
          return v
  
  class PredictionResponse(BaseModel):
      predicted_usage_kwh: float
      confidence_interval_lower: Optional[float] = None
      confidence_interval_upper: Optional[float] = None
      model_version: str
      prediction_timestamp: str
  
  @app.post("/predict", response_model=PredictionResponse)
  async def predict(request: PredictionRequest):
      try:
          # Feature engineering
          features = engineer_features(request.dict())
          
          # Scale
          features_scaled = scaler.transform([features])
          
          # Predict
          prediction = model.predict(features_scaled)[0]
          
          # Confidence interval (si modelo lo soporta)
          ci_lower, ci_upper = None, None
          if hasattr(model, 'predict_interval'):
              ci_lower, ci_upper = model.predict_interval(features_scaled, alpha=0.05)
          
          return PredictionResponse(
              predicted_usage_kwh=float(prediction),
              confidence_interval_lower=float(ci_lower) if ci_lower else None,
              confidence_interval_upper=float(ci_upper) if ci_upper else None,
              model_version="chronos_v1.0",
              prediction_timestamp=datetime.utcnow().isoformat()
          )
      
      except Exception as e:
          raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
  
  @app.post("/predict/batch")
  async def predict_batch(requests: List[PredictionRequest]):
      # Implementación similar pero vectorizada
      pass
  
  @app.get("/health")
  async def health():
      return {
          "status": "healthy",
          "model_loaded": model is not None,
          "timestamp": datetime.utcnow().isoformat()
      }
  
  @app.get("/model/info")
  async def model_info():
      return {
          "model_type": "Chronos-T5-Small",
          "version": "1.0",
          "trained_on": "2024-XX-XX",
          "features": list(model.feature_names_in_),
          "metrics": {
              "rmse": 0.189,
              "mae": 0.043,
              "cv_percent": 0.71
          }
      }
  ```

**US-020b: Exportar modelo a ONNX**
- Como: ML Engineer
- Quiero: Modelo en formato ONNX
- Para que: Sea portable y más rápido en inferencia
- Criterios:
  - [x] Chronos/XGBoost convertido a ONNX
  - [x] Script de conversión: src/models/export_onnx.py
  - [x] Validación: predicciones idénticas (tolerance 1e-5)
  - [x] Benchmark: ONNX vs PyTorch (latencia)
  - [x] Endpoint alternativo /predict_onnx en FastAPI
- Estimación: 3 story points (4 hrs)
- Código:
```python
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn

# Para XGBoost
model_onnx = convert_sklearn(xgb_model, 
    initial_types=[('float_input', FloatTensorType([None, n_features]))])
onnx.save_model(model_onnx, "models/production/model.onnx")
# Para Chronos (más complejo, usar torch.onnx.export)

```

**US-021: Dockerizar API**
- **Como** DevOps engineer
- **Quiero** la API en un contenedor
- **Para que** pueda deployar en cualquier plataforma
- **Criterios de Aceptación:**
  - [x] Multi-stage Dockerfile optimizado
  - [x] Imagen < 1.5GB
  - [x] Build time < 5 min
  - [x] Healthcheck configurado
  - [x] .dockerignore para excluir data/
- **Estimación:** 2 story points (2 hrs)
- **Dockerfile esperado:**
  ```dockerfile
  # Stage 1: Builder
  FROM python:3.11-slim as builder
  
  RUN pip install poetry
  
  WORKDIR /app
  COPY pyproject.toml poetry.lock ./
  RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
  
  # Stage 2: Runtime
  FROM python:3.11-slim
  
  WORKDIR /app
  
  # Install dependencies
  COPY --from=builder /app/requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  
  # Copy application
  COPY src/ ./src/
  COPY models/production/ ./models/production/
  
  # Create non-root user
  RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
  USER apiuser
  
  EXPOSE 8000
  
  HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
  
  CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
       "src.api.main:app", "--bind", "0.0.0.0:8000", \
       "--timeout", "300", "--access-logfile", "-"]
  ```

---

#### **Epic 9: Testing (Todos - 2 hrs c/u)**

**US-023: Tests unitarios**
- **Como** developer
- **Quiero** tests comprehensivos
- **Para que** prevenga regresiones
- **Criterios de Aceptación:**
  - [x] >70% code coverage
  - [x] Tests para:
    - Data cleaning functions
    - Feature engineering functions
    - API endpoints (con TestClient)
    - Preprocessing pipeline
  - [x] Pytest configurado con plugins (cov, asyncio)
  - [x] CI ejecuta tests automáticamente
- **Estimación:** 5 story points (10 hrs total equipo)
- **Ejemplo de tests:**
  ```python
  # tests/test_data_cleaning.py
  import pytest
  import polars as pl
  from src.data.clean_data import handle_missing_values, remove_outliers
  
  def test_handle_missing_values():
      # Arrange
      df = pl.DataFrame({
          'col1': [1.0, None, 3.0, 4.0],
          'col2': [10, 20, None, 40]
      })
      
      # Act
      result = handle_missing_values(df)
      
      # Assert
      assert result['col1'].null_count() == 0
      assert result['col2'].null_count() == 0
      assert result['col1'][1] == 2.5  # median de [1, 3, 4]
  
  def test_remove_outliers_iqr():
      # Arrange
      df = pl.DataFrame({
          'usage_kwh': [10, 12, 11, 13, 100, 12, 11]  # 100 es outlier
      })
      
      # Act
      result = remove_outliers(df, ['usage_kwh'], method='iqr')
      
      # Assert
      assert len(result) == 6
      assert 100 not in result['usage_kwh'].to_list()
  
  # tests/test_api.py
  from fastapi.testclient import TestClient
  from src.api.main import app
  
  client = TestClient(app)
  
  def test_health_endpoint():
      response = client.get("/health")
      assert response.status_code == 200
      assert response.json()["status"] == "healthy"
  
  def test_predict_endpoint_valid_input():
      payload = {
          "lagging_reactive_power": 25.5,
          "leading_reactive_power": 15.2,
          "co2": 0.05,
          "lagging_power_factor": 0.85,
          "leading_power_factor": 0.92,
          "nsm": 43200,  # mediodía
          "day_of_week": 2,  # miércoles
          "load_type": "Medium"
      }
      
      response = client.post("/predict", json=payload)
      
      assert response.status_code == 200
      data = response.json()
      assert "predicted_usage_kwh" in data
      assert data["predicted_usage_kwh"] > 0
      assert "model_version" in data
  
  def test_predict_endpoint_invalid_load_type():
      payload = {
          "lagging_reactive_power": 25.5,
          "leading_reactive_power": 15.2,
          "co2": 0.05,
          "lagging_power_factor": 0.85,
          "leading_power_factor": 0.92,
          "nsm": 43200,
          "day_of_week": 2,
          "load_type": "InvalidType"  # Inválido
      }
      
      response = client.post("/predict", json=payload)
      assert response.status_code == 422  # Validation error
  
  # tests/test_features.py
  from src.features.build_features import create_temporal_features
  import polars as pl
  import numpy as np
  
  def test_cyclical_encoding():
      df = pl.DataFrame({
          'hour': [0, 6, 12, 18, 23]
      })
      
      result = create_temporal_features(df)
      
      assert 'hour_sin' in result.columns
      assert 'hour_cos' in result.columns
      
      # Verificar que hour=0 y hour=24 son equivalentes (ciclicidad)
      assert np.isclose(result['hour_sin'][0], 0, atol=0.01)
      assert np.isclose(result['hour_cos'][0], 1, atol=0.01)
  ```

**Configuración pytest:**
```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=70",
    "--verbose",
    "--tb=short"
]
```

---

**US-023b: Tests de integración end-to-end**
- Criterios adicionales:
  - [x] Test completo: data → preprocessing → model → API → response
  - [x] Test de pipeline Prefect (mock)
  - [x] Test de Docker container (healthcheck + predict)
  - [x] Test de reproducibilidad: DVC pull → train → compare metrics
- Estimación: +2 story points (4 hrs adicionales)
- Ejemplo:
```python
def test_full_pipeline_integration():
    # 1. Load data con DVC
    subprocess.run(["dvc", "pull"])
    # 2. Train
    from src.pipelines.training_pipeline import training_pipeline
    metrics = training_pipeline()
    # 3. Assert RMSE < threshold
    assert metrics['rmse'] < 0.21

---

#### **Epic 10: Cloud Deployment (Arthur + Dante - 8 hrs c/u)**

**US-023: Cloud Run deployment**
- **Como** DevOps engineer
- **Quiero** deployar la API en Cloud Run
- **Para que** esté accesible públicamente
- **Criterios de Aceptación:**
  - [x] Imagen Docker pusheada a Google Container Registry
  - [x] Servicio Cloud Run deployado en us-central1
  - [x] Configuración FinOps:
    - Min instances: 0 (scale-to-zero)
    - Max instances: 2
    - Concurrency: 80
    - CPU: 1 vCPU
    - Memory: 2GiB
  - [x] Variables de entorno configuradas (MODEL_PATH)
  - [x] URL pública funcionando
  - [x] Latencia p95 < 1 segundo
- **Estimación:** 5 story points (6 hrs)
- **Comandos esperados:**
  ```bash
  # 1. Build y push de imagen
  docker build -t gcr.io/PROJECT_ID/energy-api:v1 .
  docker push gcr.io/PROJECT_ID/energy-api:v1
  
  # 2. Deploy a Cloud Run
  gcloud run deploy energy-optimization-api \
    --image gcr.io/PROJECT_ID/energy-api:v1 \
    --platform managed \
    --region us-central1 \
    --min-instances 0 \
    --max-instances 2 \
    --concurrency 80 \
    --cpu 1 \
    --memory 2Gi \
    --timeout 300 \
    --allow-unauthenticated \
    --set-env-vars MODEL_PATH=gs://energy-opt-models/chronos_v1.pth
  
  # 3. Verificar deployment
  curl https://energy-optimization-api-xxxxx.run.app/health
  ```

**Estimación de costos:**
```
Cloud Run pricing (us-central1):
- CPU: $0.00002400 / vCPU-second
- Memory: $0.00000250 / GiB-second
- Requests: $0.40 / million

Escenario 1: Demo (bajo tráfico)
- 100 requests/día × 30 días = 3,000 requests/mes
- Tiempo promedio: 500ms
- CPU time: 3,000 × 0.5s = 1,500 vCPU-seconds = $0.036
- Memory time: 3,000 × 0.5s × 2GiB = 3,000 GiB-seconds = $0.008
- Requests: 3,000 × $0.40/million = $0.001
- **Total: ~$0.05/mes**

Escenario 2: Presentación (tráfico medio)
- 1,000 requests/día × 5 días = 5,000 requests
- **Total: ~$0.15**

Total estimado Sprint 2-3: $5-10
```

**US-024: CI/CD con GitHub Actions**
- **Como** DevOps engineer
- **Quiero** deployment automático en cada merge a main
- **Para que** tengamos continuous delivery
- **Criterios de Aceptación:**
  - [x] Workflow `.github/workflows/deploy.yml`
  - [x] Triggered en push a `main`
  - [x] Pasos:
    1. Run tests
    2. Build Docker image
    3. Push to GCR
    4. Deploy to Cloud Run
  - [x] Secrets configurados (GCP_PROJECT_ID, GCP_SA_KEY)
- **Estimación:** 3 story points (4 hrs)
- **GitHub Actions workflow:**
  ```yaml
  # .github/workflows/deploy.yml
  name: Deploy to Cloud Run
  
  on:
    push:
      branches: [main]
    workflow_dispatch:
  
  env:
    PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
    SERVICE_NAME: energy-optimization-api
    REGION: us-central1
  
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.11'
        
        - name: Install Poetry
          run: pip install poetry
        
        - name: Install dependencies
          run: poetry install
        
        - name: Run tests
          run: poetry run pytest --cov --cov-fail-under=70
    
    deploy:
      needs: test
      runs-on: ubuntu-latest
      
      steps:
        - uses: actions/checkout@v3
        
        - name: Authenticate to Google Cloud
          uses: google-github-actions/auth@v1
          with:
            credentials_json: ${{ secrets.GCP_SA_KEY }}
        
        - name: Set up Cloud SDK
          uses: google-github-actions/setup-gcloud@v1
        
        - name: Configure Docker for GCR
          run: gcloud auth configure-docker
        
        - name: Build Docker image
          run: |
            docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA \
                         -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .
        
        - name: Push to GCR
          run: |
            docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA
            docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest
        
        - name: Deploy to Cloud Run
          run: |
            gcloud run deploy $SERVICE_NAME \
              --image gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA \
              --platform managed \
              --region $REGION \
              --min-instances 0 \
              --max-instances 2 \
              --allow-unauthenticated
        
        - name: Show deployment URL
          run: |
            URL=$(gcloud run services describe $SERVICE_NAME \
                  --region $REGION --format 'value(status.url)')
            echo "Deployed to: $URL"
  ```

**US-025: API Documentation (OpenAPI/Swagger)**
- **Como** usuario de la API
- **Quiero** documentación interactiva
- **Para que** pueda entender cómo usar endpoints
- **Criterios de Aceptación:**
  - [x] FastAPI genera docs automáticamente
  - [x] `/docs` (Swagger UI) accesible
  - [x] `/redoc` (ReDoc) accesible
  - [x] Ejemplos de requests/responses
  - [x] Descripción de cada campo
- **Estimación:** 1 story point (1 hr)

---

#### **Epic 11: Model Monitoring (Arthur + Erick - 6 hrs c/u)**

**US-026: Evidently AI - Data Drift Detection**
- **Como** ML engineer
- **Quiero** detectar drift en datos de producción
- **Para que** sepa cuándo re-entrenar el modelo
- **Criterios de Aceptación:**
  - [x] Script de monitoreo semanal
  - [x] Reporte HTML generado automáticamente
  - [x] Métricas monitoreadas:
    - Data drift (PSI, KS test)
    - Target drift
    - Prediction drift
    - Model performance (RMSE, MAE)
  - [x] Alertas por email si drift severo
  - [x] Reportes guardados en `reports/monitoring/`
- **Estimación:** 5 story points (6 hrs)
- **Código esperado:**
  ```python
  # src/monitoring/drift_detection.py
  from evidently.report import Report
  from evidently.metric_preset import DataDriftPreset, RegressionPreset
  from evidently.test_suite import TestSuite
  from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns
  import pandas as pd
  from datetime import datetime, timedelta
  
  def load_reference_data():
      """Carga datos de entrenamiento (referencia)"""
      return pd.read_parquet('data/processed/train.parquet')
  
  def load_production_data(days_back=7):
      """Carga datos de producción de últimos N días"""
      # En producción real, esto vendría de logs de predicciones
      # Para Sprint 2, simulamos con subset de test
      return pd.read_parquet('data/processed/test.parquet').tail(1000)
  
  def generate_drift_report():
      reference_data = load_reference_data()
      production_data = load_production_data(days_back=7)
      
      # Report de drift
      drift_report = Report(metrics=[
          DataDriftPreset(),
          RegressionPreset()
      ])
      
      drift_report.run(
          reference_data=reference_data,
          current_data=production_data,
          column_mapping={
              'target': 'Usage_kWh',
              'prediction': 'predicted_usage_kwh',
              'numerical_features': [
                  'Lagging_Current_Reactive_Power_kVarh',
                  'Leading_Current_Reactive_Power_kVarh',
                  'CO2',
                  'Lagging_Current_Power_Factor',
                  'Leading_Current_Power_Factor',
                  'NSM'
              ],
              'categorical_features': ['Load_Type', 'day_of_week']
          }
      )
      
      # Guardar reporte
      report_name = f"drift_report_{datetime.now().strftime('%Y%m%d')}.html"
      drift_report.save_html(f"reports/monitoring/{report_name}")
      
      # Extraer métricas para alerting
      results = drift_report.as_dict()
      dataset_drift = results['metrics'][0]['result']['dataset_drift']
      drift_score = results['metrics'][0]['result']['drift_share']
      
      return {
          'dataset_drift': dataset_drift,
          'drift_share': drift_score,
          'report_path': report_name
      }
  
  def check_for_alerts(metrics):
      """Envía alertas si drift es severo"""
      if metrics['dataset_drift'] or metrics['drift_share'] > 0.3:
          send_alert_email(
              subject="⚠️ Data Drift Detected!",
              message=f"""
              Data drift detectado en modelo de producción:
              - Dataset drift: {metrics['dataset_drift']}
              - Proporción de features con drift: {metrics['drift_share']:.2%}
              
              Revisar reporte: {metrics['report_path']}
              
              Acción recomendada: Considerar reentrenamiento del modelo.
              """
          )
  
  if __name__ == "__main__":
      metrics = generate_drift_report()
      check_for_alerts(metrics)
      print(f"✅ Drift report generated: {metrics['report_path']}")
  ```

**US-027: Scheduled monitoring job**
- **Como** MLOps engineer
- **Quiero** monitoreo automático semanal
- **Para que** no tenga que ejecutarlo manualmente
- **Criterios de Aceptación:**
  - [x] Prefect flow para monitoreo
  - [x] Schedule: cada lunes 9am
- **Estimación:** 2 story points (2 hrs)

---

### 9.3 Estimación Total del Sprint 2

| Epic | Person-Hours | % del Sprint |
|------|--------------|--------------|
| EP-7: Automated Pipeline | 16 | 13% |
| EP-8: API Development | 16 | 13% |
| EP-9: Testing | 10 | 8% |
| EP-10: Cloud Deployment | 16 | 13% |
| EP-11: Monitoring | 12 | 10% |
| **Total** | **70** | **58%** |

**Buffer:** 50 person-hours (42%) para debugging, optimización, documentación adicional.

### 9.4 Definition of Done - Sprint 2

- [ ] API deployada en Cloud Run respondiendo correctamente
- [ ] CI/CD pipeline funcional (tests + deploy automático)
- [ ] >70% code coverage con pytest
- [ ] Documentación Swagger accesible en `/docs`
- [ ] Evidently AI generando reportes de drift
- [ ] Latencia p95 < 1 segundo en producción
- [ ] Costos GCP < $10 en el sprint
- [ ] Video demo de la API funcionando

---

## 10. Plan de Ejecución: Sprint 3

**Duración:** Semanas 5-6  
**Objetivo:** Desarrollar interfaz conversacional con LLM, completar Copiloto de IA, pulir deployment  
**Sprint Goal Statement:** "Al final del Sprint 3, tendremos un Copiloto de IA completamente funcional que permite consultas en lenguaje natural, desplegado y documentado profesionalmente."

### 10.1 Objetivos Específicos del Sprint 3

1. ✅ Configurar Ollama con Llama 3.2 3B en la GPU 4070
2. ✅ Integrar LLM con backend FastAPI
3. ✅ Desarrollar frontend con Streamlit/Gradio
4. ✅ Implementar casos de uso conversacionales (predicción, what-if, explicaciones)
5. ✅ Optimizar performance y experiencia de usuario
6. ✅ Documentación completa del proyecto
7. ✅ Presentación final y video demo

### 10.2 Backlog del Sprint 3

#### **Epic 12: LLM Integration (Dante + Julian - 10 hrs c/u)**

**US-028: Setup Ollama y Llama 3.2**
- **Como** AI engineer
- **Quiero** correr un LLM local en la GPU
- **Para que** podamos procesar consultas en lenguaje natural
- **Criterios de Aceptación:**
  - [x] Ollama instalado en máquina de Arthur
  - [x] Llama 3.2 3B descargado y funcionando
  - [x] API de Ollama corriendo en puerto 11434
  - [x] Latencia de respuesta < 2 segundos
  - [x] Uso de VRAM < 8GB
- **Estimación:** 2 story points (3 hrs)
- **Comandos:**
  ```bash
  # Instalar Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  
  # Descargar Llama 3.2 3B
  ollama pull llama3.2:3b
  
  # Verificar
  ollama list
  
  # Test inference
  curl http://localhost:11434/api/generate -d '{
    "model": "llama3.2:3b",
    "prompt": "Explica qué es el consumo energético industrial",
    "stream": false
  }'
  ```
---

**US-028b: Refactorización y Code Quality**
- Criterios:
  - [x] Aplicar principios SOLID donde aplique
  - [x] Extraer constantes a config.py
  - [x] Type hints en todas las funciones
  - [x] Docstrings estilo Google
  - [x] Ruff + Black + MyPy sin warnings

---

**US-029: Endpoint /copilot/chat en FastAPI**
- **Como** desarrollador
- **Quiero** un endpoint que procese consultas en lenguaje natural
- **Para que** el frontend pueda comunicarse con el LLM
- **Criterios de Aceptación:**
  - [x] Endpoint `POST /copilot/chat`
  - [x] Input: texto libre del usuario
  - [x] Output: respuesta del LLM + datos de predicción si aplica
  - [x] System prompt que instruye al LLM
  - [x] Integración con Ollama API
- **Estimación:** 5 story points (7 hrs)
---

**US-030: Prompts avanzados y chain-of-thought**
- **Como** AI engineer
- **Quiero** que el LLM razone paso a paso
- **Para que** genere respuestas más precisas
- **Criterios de Aceptación:**
  - [x] System prompt optimizado con ejemplos (few-shot)
  - [x] Chain-of-thought para preguntas complejas
  - [x] Validación de parámetros extraídos
- **Estimación:** 3 story points (5 hrs)

---

#### **Epic 13: Frontend Development (Dante - 12 hrs)**

**US-031: Streamlit UI - Página de predicción**
- **Como** usuario final
- **Quiero** una interfaz web intuitiva
- **Para que** pueda usar el sistema sin conocimientos técnicos
- **Criterios de Aceptación:**
  - [x] App Streamlit con 3 páginas:
    1. Home/Introducción
    2. Predicción Simple
    3. Copiloto Conversacional
  - [x] Formulario de entrada con validación
  - [x] Visualización de resultados (gauges, gráficos)
  - [x] Responsive design
- **Estimación:** 5 story points (6 hrs)

**US-032: Deployment de Streamlit**
- **Como** usuario
- **Quiero** acceder a la app desde cualquier lugar
- **Para que** no dependa de entorno local
- **Criterios de Aceptación:**
  - [x] Dockerizar app Streamlit
  - [x] Deployar en Cloud Run O Streamlit Cloud (gratis)
  - [x] URL pública accesible
- **Estimación:** 3 story points (4 hrs)

---

#### **Epic 14: Polish & Documentation (Todos - 2 hrs c/u)**

**US-033: README comprehensivo**
- **Como** futuro usuario/desarrollador
- **Quiero** documentación clara de setup
- **Para que** pueda replicar el proyecto
- **Criterios de Aceptación:**
  - [x] README.md con:
    - Descripción del proyecto
    - Arquitectura (diagrama)
    - Setup instructions
    - Uso de la API
    - Uso del frontend
    - Contribuir
    - Licencia
  - [x] Badges (coverage, build status)
- **Estimación:** 3 story points (6 hrs - equipo completo)

**US-034: Presentación final**
- **Como** equipo
- **Quiero** presentación ejecutiva final
- **Para que** mostremos todo el proyecto
- **Criterios de Aceptación:**
  - [x] PDF 15-20 slides
  - [x] Estructura completa del proyecto
- **Estimación:** 5 story points (8 hrs)

**US-035: Video demo final**
- **Como** stakeholder
- **Quiero** ver el sistema en acción
- **Para que** entienda su valor
- **Criterios de Aceptación:**
  - [x] Video 10-15 min
  - [x] Edición profesional
- **Estimación:** 8 story points (12 hrs)

---

**US-036: Simulación y detección de Data Drift**
- Como: MLOps Engineer  
- Quiero: Simular degradación del modelo por drift
- Para que: Valide que el monitoreo funciona
- Criterios:
  - [x] Script que introduce drift artificial:
    - Shift en distribución de features (ej: aumentar CO2 +50%)
    - Cambio de proporción Load_Type
    - Outliers sintéticos
  - [x] Evidently detecta el drift (PSI > 0.2)
  - [x] Alertas se disparan correctamente
  - [x] Documentar degradación de RMSE
  - [x] Notebook: `04_drift_simulation.ipynb`
- Estimación: 3 story points (4 hrs)


---

**US-037: Verificación de portabilidad**
- Como: DevOps Engineer
- Quiero: Probar el proyecto en máquina diferente
- Para que: Garantice reproducibilidad total
- Criterios:
  - [x] Documentar setup en máquina limpia (otro miembro del equipo)
  - [x] Checklist de portabilidad:
    - Docker build exitoso
    - DVC pull funciona
    - Tests pasan
    - API responde
    - Métricas del modelo coinciden (±0.001)
  - [x] CI/CD simula ambiente limpio
- Estimación: 2 story points (3 hrs)

---

### 10.3 Estimación Total del Sprint 3

| Epic | Person-Hours | % del Sprint |
|------|--------------|--------------|
| EP-12: LLM Integration | 20 | 17% |
| EP-13: Frontend | 12 | 10% |
| EP-14: Polish & Docs | 26 | 22% |
| **Total** | **58** | **48%** |

**Buffer:** 62 person-hours (52%) para refinamiento UX, optimización de prompts, testing end-to-end.

### 10.4 Definition of Done - Sprint 3

- [ ] Copiloto responde correctamente a ≥5 casos de uso documentados
- [ ] Frontend Streamlit deployado y accesible
- [ ] README completo con instrucciones verificadas
- [ ] Video demo final aprobado por el equipo
- [ ] Presentación ejecutiva revisada
- [ ] Proyecto cumple criterios de rúbrica (≥90/100)
- [ ] Repositorio GitHub público con documentación completa

---

## 11. Roles y Responsabilidades

### 11.1 Matriz RACI Completa (Todos los Sprints)

| Actividad | Juan (DE) | Erick (DS) | Julian (MLE) | Dante (SE/SM) | Arthur (MLOps) |
|-----------|-----------|------------|--------------|---------------|----------------|
| **Sprint 1** | | | | | |
| Data Cleaning | R, A | C | I | I | I |
| EDA | C | R, A | C | I | I |
| Feature Engineering | C | R | R | I | A |
| Baseline Models | I | C | R, A | I | I |
| Chronos Implementation | I | C | R, A | I | C |
| ML Canvas | C | R | C | A | I |
| Presentación S1 | C | C | C | R, A | C |
| **Sprint 2** | | | | | |
| ML Pipeline | C | I | C | I | R, A |
| FastAPI Development | C | I | R | R | C |
| Testing | R | R | R | R, A | R |
| Docker/CI-CD | I | I | C | C | R, A |
| Cloud Deployment | I | I | C | C | R, A |
| Evidently Setup | I | C | C | I | R, A |
| **Sprint 3** | | | | | |
| Ollama/LLM Setup | I | I | C | C | R, A |
| Copilot Endpoint | I | I | R | R | C |
| Streamlit Frontend | I | I | C | R, A | C |
| Documentation | C | C | C | R, A | C |
| Video Final | C | C | C | R, A | C |

**Leyenda:**
- **R** (Responsible): Quien ejecuta la tarea
- **A** (Accountable): Quien aprueba/es responsable final
- **C** (Consulted): Quien debe ser consultado
- **I** (Informed): Quien debe ser informado

### 11.2 Descripción de Roles

#### **Juan - Data Engineer**
**Responsabilidades Principales:**
- Ingesta y limpieza de datos crudos
- Validación de calidad de datos
- Setup de DuckDB y pipelines de ETL
- Versionado de datasets con DVC
- Documentación de procesos de datos

**Skills Requeridos:**
- Python (Polars/Pandas)
- SQL
- DVC
- Data profiling y validation

**Entregables Clave:**
- Dataset limpio validado
- Scripts de ETL reproducibles
- Documentación de data quality

---

#### **Erick - Data Scientist**
**Responsabilidades Principales:**
- Análisis Exploratorio de Datos (EDA)
- Feature engineering
- Interpretación de resultados de modelos
- Análisis de drift con Evidently
- Generación de insights de negocio

**Skills Requeridos:**
- Python (pandas, scikit-learn)
- Estadística y visualización
- Feature engineering
- ML explainability

**Entregables Clave:**
- Notebooks de EDA
- Features ingenierizados
- Reportes de análisis

---

#### **Julian - Machine Learning Engineer**
**Responsabilidades Principales:**
- Entrenamiento de modelos (baseline + foundation)
- Hyperparameter tuning
- Evaluación y validación de modelos
- Implementación de Chronos/TimesFM
- Integración de LLM con API

**Skills Requeridos:**
- PyTorch/TensorFlow
- HuggingFace Transformers
- MLflow
- Hyperparameter optimization

**Entregables Clave:**
- Modelos entrenados y serializados
- Comparativa de performance
- Documentación de arquitecturas

---

#### **Dante - Software Engineer & Scrum Master**
**Responsabilidades Principales:**
- Desarrollo del backend FastAPI
- Desarrollo del frontend Streamlit
- Gestión de Scrum (facilitar ceremonias)
- Gestión de Linear/GitHub
- Documentación y presentaciones

**Skills Requeridos:**
- FastAPI/Python backend
- Streamlit/frontend básico
- Scrum methodology
- Technical writing

**Entregables Clave:**
- API RESTful funcional
- UI de usuario
- Presentaciones y videos
- Gestión de sprints

---

#### **Arthur - MLOps/SRE Engineer**
**Responsabilidades Principales:**
- Setup de infraestructura (Poetry, Docker, DVC)
- CI/CD con GitHub Actions
- Deployment en Cloud Run
- Monitoreo y alerting
- FinOps y optimización de costos
- Gestión de GPU para entrenamiento

**Skills Requeridos:**
- Docker/Kubernetes
- GitHub Actions
- Google Cloud Platform
- Prefect/orchestration
- Infrastructure as Code

**Entregables Clave:**
- Pipelines CI/CD
- Deployment automatizado
- Monitoring dashboards
- Documentación de infra

---

## 12. Gestión de Riesgos

### 12.1 Registro de Riesgos

| ID | Riesgo | Probabilidad | Impacto | Severidad | Mitigación | Plan de Contingencia |
|----|--------|--------------|---------|-----------|------------|----------------------|
| R-001 | Chronos no supera benchmark CUBIST | Media | Alto | **ALTO** | Feature engineering adicional, ensemble con XGBoost | Usar TimesFM como alternativa, ensemble multi-modelo |
| R-002 | GPU 4070 insuficiente para fine-tuning | Baja | Medio | MEDIO | Usar gradient accumulation, mixed precision (fp16) | Fine-tuning en Colab Pro (gratis por 12hrs) |
| R-003 | Presupuesto GCP excedido | Media | Alto | **ALTO** | Monitoring diario de gastos, scale-to-zero agresivo | Deployment solo local, usar Streamlit Cloud gratuito |
| R-004 | Ollama/Llama no corre bien en 4070 | Baja | Medio | MEDIO | Usar modelo 3B (pequeño), optimizar prompts | Usar Gemini Flash API (pricing bajo: $0.075/1M tokens) |
| R-005 | Miembro del equipo no disponible temporalmente | Media | Medio | MEDIO | Pair programming, documentación clara, cross-training | Re-asignar tareas, ajustar velocity del sprint |
| R-006 | Drift detection genera falsos positivos | Media | Bajo | BAJO | Tuning de thresholds, validación manual | Aumentar threshold de alertas, review semanal |
| R-007 | Latencia de API > 1 segundo | Media | Medio | MEDIO | Caching de modelos en memoria, optimizar preprocessing | Usar modelo más pequeño, pre-computar features comunes |
| R-008 | CI/CD pipeline falla frecuentemente | Baja | Bajo | BAJO | Tests locales antes de push, linting automático | Debugging manual, rollback a último commit funcional |
| R-009 | Dataset de referencia no disponible | Baja | Medio | MEDIO | Validación con métricas estadísticas propias | Confiar en validación cruzada interna |
| R-010 | Conflictos de merge en GitHub | Media | Bajo | BAJO | PRs pequeños, reviews diarias, comunicación activa | Sesión de pair programming para resolver conflictos |

### 12.2 Plan de Monitoreo de Riesgos

**Frecuencia:** Revisión en cada Sprint Retrospective

**Métricas a Monitorear:**
- Velocity del sprint (story points completados)
- Gasto acumulado en GCP (tracking semanal)
- Performance de modelos (RMSE en validation set)
- Latencia de API (p95, p99)
- Code coverage (%)

**Responsable:** Dante (Scrum Master)

---

## 13. Plan de Entregables

### 13.1 Entregables por Sprint

#### **Sprint 1 (Semanas 1-2)**

| Entregable | Tipo | Responsable | Fecha Límite |
|------------|------|-------------|--------------|
| Repositorio GitHub configurado | Código | Arthur | Día 2 |
| Dataset limpio (versionado DVC) | Datos | Juan | Día 5 |
| Notebook EDA | Documentación | Erick | Día 7 |
| Modelos baseline entrenados | Código | Julian | Día 10 |
| Modelo Chronos fine-tuned | Código | Julian | Día 12 |
| ML Canvas completo | Documentación | Dante/Erick | Día 11 |
| Presentación Sprint 1 (PDF) | Documentación | Dante | Día 13 |
| Video Sprint 1 (7-10 min) | Media | Todos | Día 14 |
| Tabla comparativa de modelos | Documentación | Julian | Día 12 |

#### **Sprint 2 (Semanas 3-4)**

| Entregable | Tipo | Responsable | Fecha Límite |
|------------|------|-------------|--------------|
| Pipeline Prefect funcional | Código | Arthur/Julian | Día 21 |
| API FastAPI completa | Código | Dante/Julian | Día 24 |
| Dockerfile optimizado | Código | Arthur | Día 23 |
| Tests unitarios (>70% coverage) | Código | Todos | Día 26 |
| CI/CD GitHub Actions | Código | Arthur | Día 25 |
| Deployment Cloud Run | Infraestructura | Arthur | Día 27 |
| Documentación API (Swagger) | Documentación | Dante | Día 24 |
| Evidently monitoring setup | Código | Arthur/Erick | Día 26 |
| Video demo API | Media | Dante | Día 28 |

#### **Sprint 3 (Semanas 5-6)**

| Entregable | Tipo | Responsable | Fecha Límite |
|------------|------|-------------|--------------|
| Ollama + Llama 3.2 configurado | Infraestructura | Arthur | Día 32 |
| Endpoint /copilot/chat | Código | Dante/Julian | Día 35 |
| Frontend Streamlit | Código | Dante | Día 38 |
| Deployment frontend | Infraestructura | Dante/Arthur | Día 39 |
| README comprehensivo | Documentación | Todos | Día 40 |
| Presentación final (PDF) | Documentación | Dante | Día 41 |
| Video demo final (10-15 min) | Media | Todos | Día 42 |
| Informe técnico completo | Documentación | Todos | Día 42 |

### 13.2 Entregables Académicos Finales

#### **Documento Final del Proyecto**
**Contenido mínimo:**
1. Resumen Ejecutivo
2. Introducción y Contexto
3. Objetivos
4. Marco Teórico
5. Metodología (Scrum + MLOps)
6. Arquitectura de la Solución
7. Implementación
   - Data Engineering
   - Feature Engineering
   - Model Training & Evaluation
   - MLOps Pipeline
   - API Development
   - Copilot Development
8. Resultados
   - Comparativa de modelos
   - Métricas de performance
   - Análisis de costos
9. Discusión
   - Limitaciones
   - Aprendizajes
   - Trabajo futuro
10. Conclusiones
11. Referencias
12. Anexos

**Formato:** PDF, 40-60 páginas  
**Responsable:** Todos (coordinación: Dante)  
**Fecha límite:** Día 42

#### **Presentación Ejecutiva Final**
**Contenido:** 15-20 slides profesionales  
**Duración presentación:** 15-20 minutos  
**Formato:** PDF + PPT  
**Responsable:** Dante  
**Fecha límite:** Día 41

#### **Video Demostrativo**
**Duración:** 10-15 minutos  
**Contenido:**
- Intro al problema (2 min)
- Arquitectura (2 min)
- Demo live de predicción (3 min)
- Demo live del Copiloto (4 min)
- Resultados y conclusiones (3 min)

**Formato:** MP4, 1080p  
**Plataforma:** YouTube (unlisted) + link en GitHub  
**Responsable:** Todos (edición: Dante)  
**Fecha límite:** Día 42

#### **Repositorio GitHub**
**Debe contener:**
- [ ] Código fuente completo
- [ ] Tests con >70% coverage
- [ ] Documentación README
- [ ] Notebooks de EDA
- [ ] Configuración Docker
- [ ] CI/CD workflows
- [ ] ML Canvas
- [ ] Licencia (MIT)
- [ ] Contribuidores claramente identificados
- [ ] Badges (build status, coverage, license)

**Visibilidad:** Público  
**URL:** https://github.com/[username]/energy-optimization-copilot  
**Responsable:** Arthur  
**Fecha límite:** Día 42

---

## 14. Conclusiones

### 14.1 Resumen del Plan

Este documento presenta un plan comprehensivo y ejecutable para desarrollar un **Copiloto de IA para Optimización Energética** en la industria siderúrgica durante un período de 6 semanas, con 3 sprints de 2 semanas cada uno.

**Elementos Diferenciadores:**

1. **Innovación Técnica:** Uso de Foundation Models SOTA (Chronos-T5, TimesFM) que representan el estado del arte 2024-2025, superando benchmarks de modelos clásicos como CUBIST.

2. **Arquitectura Híbrida Pragmática:** Combinación de entrenamiento local (aprovechando GPU 4070) con deployment cloud optimizado para FinOps, manteniendo costos bajo $50 USD.

3. **MLOps Profesional:** Pipeline completo con versionado (DVC), tracking (MLflow), orquestación (Prefect), testing (pytest), CI/CD (GitHub Actions) y monitoreo (Evidently AI).

4. **Enfoque Human-Centric:** No solo predicción numérica, sino un Copiloto conversacional que traduce datos en decisiones accionables mediante IA generativa (Llama 3.2).

5. **Metodología Rigurosa:** Scrum adaptado para ML con ceremonias estructuradas, matriz RACI clara, y Definition of Done precisa.

### 14.2 Criterios de Éxito

El proyecto será considerado exitoso si cumple:

**Técnicos:**
- ✅ RMSE < 0.205 (15% mejor que CUBIST)
- ✅ API deployada con latencia p95 < 1 segundo
- ✅ Copiloto respondiendo correctamente a ≥5 casos de uso
- ✅ Code coverage >70%
- ✅ Pipeline 100% reproducible con DVC

**Económicos:**
- ✅ Gasto total GCP ≤ $50 USD
- ✅ Scale-to-zero implementado correctamente

**Académicos:**
- ✅ Calificación ≥ 90/100 en rúbrica del Sprint 1
- ✅ Documentación completa y profesional
- ✅ Video demo de calidad

**Colaborativos:**
- ✅ ≥15 commits por integrante en GitHub
- ✅ 100% de historias de usuario completadas
- ✅ Velocity consistente entre sprints

### 14.3 Impacto Esperado

**Impacto Académico:**
- Proyecto digno de maestría que demuestra dominio de MLOps end-to-end
- Referencia para futuros estudiantes del programa
- Potencial publicación en blog técnico o conferencia estudiantil

**Impacto Profesional (Portafolio):**
- Proyecto destacable en LinkedIn/GitHub mostrando skills actuales
- Experiencia práctica con tecnologías demandadas (Transformers, MLOps, GCP)
- Demostración de trabajo en equipo y gestión de proyectos

**Impacto Técnico:**
- Comparación rigurosa de Foundation Models vs. métodos clásicos
- Arquitectura de referencia para deployment híbrido local-cloud
- Implementación práctica del patrón "Copilot" con LLMs

### 14.4 Trabajo Futuro (Post-Proyecto)

**Mejoras Técnicas:**
1. **Reentrenamiento Automático:** Trigger automático cuando Evidently detecta drift severo
2. **Multi-tenancy:** Soporte para múltiples plantas siderúrgicas
3. **Forecasting Multi-horizonte:** Predicción 1h, 24h, 7 días
4. **Optimización Multi-objetivo:** Minimizar simultáneamente costo y CO2
5. **Agentes Autónomos:** Uso de frameworks de agentes para ejecutar acciones

**Expansión de Funcionalidades:**
1. **Dashboard Ejecutivo:** Reportes semanales automáticos para gerencia
2. **Alertas Proactivas:** Notificaciones de consumo anómalo en tiempo real
3. **Integración IoT:** Conexión directa con sensores de planta
4. **Mobile App:** Versión móvil para operadores en piso de planta
5. **Explainability Avanzada:** SHAP values para explicar cada predicción

**Escalabilidad:**
1. **Multi-region Deployment:** Replicar en us-east, europe-west
2. **AutoML:** Selección automática del mejor modelo según características de datos
3. **Federated Learning:** Entrenar modelo global sin compartir datos sensibles entre plantas

### 14.5 Lecciones Aprendidas Esperadas

Al concluir el proyecto, el equipo habrá desarrollado expertise en:

1. **MLOps en la Práctica:** Diferencia entre entrenar modelos en notebooks vs. deployar en producción
2. **FinOps:** Técnicas concretas para optimizar costos en cloud
3. **Foundation Models:** Cómo fine-tunear modelos pre-entrenados para dominios específicos
4. **LLM Engineering:** Prompt engineering, chain-of-thought, integración de LLMs con APIs
5. **Gestión Ágil:** Scrum en contexto de alta incertidumbre técnica
6. **Trabajo en Equipo:** Colaboración efectiva con roles especializados

### 14.6 Compromiso del Equipo

Nosotros, el equipo de Energy Optimization Copilot, nos comprometemos a:

1. **Excelencia Técnica:** No conformarnos con "funciona", sino buscar "funciona bien y es mantenible"
2. **Comunicación Transparente:** Reportar bloqueadores temprano, no esconder problemas
3. **Colaboración Activa:** Code reviews constructivos, pair programming cuando sea necesario
4. **Aprendizaje Continuo:** Documentar aprendizajes, compartir conocimiento
5. **Responsabilidad Compartida:** El éxito o fracaso es del equipo, no individual

**Firmas (simbólicas):**
- Juan (Data Engineer)
- Erick (Data Scientist)
- Julian (Machine Learning Engineer)
- Dante (Software Engineer & Scrum Master)
- Arthur (MLOps/SRE Engineer)

---

## 15. Referencias

### 15.1 Papers Académicos

1. Dua, D., & Graff, C. (2017). UCI Machine Learning Repository: Steel Industry Energy Consumption. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

2. Ansari, A. F., et al. (2024). "Chronos: Learning the Language of Time Series." arXiv preprint arXiv:2403.07815.

3. Das, A., et al. (2025). "Benchmarking Time Series Forecasting Models: From Classical Methods to Foundation Models." arXiv preprint arXiv:2502.03395.

4. Google Research. (2024). "TimesFM: Time Series Foundation Model for Universal Forecasting." Google AI Blog.

### 15.2 Documentación Técnica

5. FastAPI Documentation. (2024). https://fastapi.tiangolo.com/

6. Streamlit Documentation. (2024). https://docs.streamlit.io/

7. MLflow Documentation. (2024). https://www.mlflow.org/docs/latest/index.html

8. DVC Documentation. (2024). https://dvc.org/doc

9. Evidently AI Documentation. (2024). https://docs.evidentlyai.com/

10. Prefect Documentation. (2024). https://docs.prefect.io/

11. HuggingFace Transformers. (2024). https://huggingface.co/docs/transformers/

12. Ollama Documentation. (2024). https://github.com/ollama/ollama

### 15.3 Google Cloud Platform

13. Google Cloud. (2024). "Cloud Run Documentation." https://cloud.google.com/run/docs

14. Google Cloud. (2024). "Best Practices for Cost Optimization." https://cloud.google.com/blog/topics/cost-management

15. Google Cloud. (2024). "Container Registry Documentation." https://cloud.google.com/container-registry/docs

### 15.4 Metodología y Mejores Prácticas

16. Huyen, C. (2022). "Designing Machine Learning Systems." O'Reilly Media.

17. Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NeurIPS.

18. Paleyes, A., et al. (2022). "Challenges in Deploying Machine Learning: A Survey of Case Studies." ACM Computing Surveys.

19. The ML Canvas Framework. https://www.louisdorard.com/machine-learning-canvas

20. Scrum Guide. (2020). https://scrumguides.org/

### 15.5 Blogs y Recursos

21. The Copilot Pattern: An Architectural Approach to AI-Assisted Software. https://www.vamsitalkstech.com/ai/the-copilot-pattern-an-architectural-approach-to-ai-assisted-software/

22. Made With ML. (2024). "MLOps Course." https://madewithml.com/

23. Full Stack Deep Learning. (2024). "Course Materials." https://fullstackdeeplearning.com/

---

## Anexos

### Anexo A: Comandos de Setup Rápido

```bash
# 1. Clonar repositorio
git clone https://github.com/[username]/energy-optimization-copilot.git
cd energy-optimization-copilot

# 2. Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 3. Instalar dependencias
poetry install

# 4. Configurar DVC
poetry run dvc remote add -d gcs gs://energy-opt-dvc-remote
poetry run dvc pull

# 5. Iniciar MLflow
poetry run mlflow ui --port 5000

# 6. Iniciar Prefect
poetry run prefect server start

# 7. Iniciar API (desarrollo)
poetry run uvicorn src.api.main:app --reload

# 8. Iniciar Streamlit
poetry run streamlit run app.py

# 9. Ejecutar tests
poetry run pytest --cov

# 10. Build Docker
docker build -t energy-api:latest .
docker run -p 8000:8000 energy-api:latest
```

### Anexo B: Estructura Completa del Repositorio

```
energy-optimization-copilot/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── deploy.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── data/
│   ├── raw/
│   │   ├── steel_dirty.csv
│   │   └── steel_clean.csv (reference)
│   ├── processed/
│   │   ├── steel_cleaned.parquet.dvc
│   │   ├── steel_featured.parquet.dvc
│   │   ├── train.parquet.dvc
│   │   ├── val.parquet.dvc
│   │   └── test.parquet.dvc
│   ├── external/
│   └── steel.duckdb
├── docs/
│   ├── ML_Canvas.md
│   ├── architecture_diagram.png
│   └── sprint_reports/
│       ├── sprint1_summary.pdf
│       ├── sprint2_summary.pdf
│       └── sprint3_summary.pdf
├── models/
│   ├── baselines/
│   │   ├── xgboost_v1.pkl.dvc
│   │   └── lightgbm_v1.pkl.dvc
│   ├── foundation/
│   │   └── chronos_finetuned_v1.pth.dvc
│   └── production/
│       ├── best_model.pkl
│       └── scaler.pkl
├── notebooks/
│   ├── 00_data_profiling.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── reports/
│   ├── figures/
│   │   ├── correlation_matrix.png
│   │   ├── stl_decomposition.png
│   │   └── model_comparison.png
│   └── monitoring/
│       └── drift_report_20250105.html
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clean_data.py
│   │   └── load_to_duckdb.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_baseline.py
│   │   ├── train_chronos.py
│   │   └── predict.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── copilot.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detection.py
│   └── pipelines/
│       ├── __init__.py
│       └── training_pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaning.py
│   ├── test_features.py
│   ├── test_api.py
│   └── test_models.py
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── app.py (Streamlit frontend)
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

### Anexo C: Checklist Pre-Entrega

**Sprint 1:**
- [ ] Dataset limpio en DVC con tag `data-v1.0`
- [ ] Notebook EDA con visualizaciones exportadas
- [ ] Modelos entrenados con métricas en MLflow
- [ ] Chronos RMSE < 0.205
- [ ] ML Canvas completado
- [ ] Presentación PDF revisada
- [ ] Video subido a YouTube
- [ ] Todos los integrantes con ≥5 commits

**Sprint 2:**
- [ ] API respondiendo en `/health`, `/predict`
- [ ] Tests >70% coverage
- [ ] CI/CD ejecutando exitosamente
- [ ] Deployment en Cloud Run funcional
- [ ] Evidently generando reportes
- [ ] Documentación Swagger accesible
- [ ] Costos GCP < $15

**Sprint 3:**
- [ ] Copiloto respondiendo a 5 casos de uso
- [ ] Streamlit deployado y accesible
- [ ] README comprehensivo
- [ ] Video final 10-15 min
- [ ] Presentación final 15-20 slides
- [ ] Repositorio público y organizado
- [ ] Todos los criterios de rúbrica cumplidos

---

**Fecha de Creación:** [Fecha]  
**Versión:** 1.0  
**Aprobado por:** Equipo Energy Optimization Copilot

---

*Este documento es un plan vivo que será actualizado conforme el proyecto avance. Todas las decisiones técnicas están sujetas a revisión en las retrospectivas de sprint.*