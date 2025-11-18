"""
OpenAPI Configuration for Energy Optimization API.

Centralizes OpenAPI metadata, tags, and custom schema generation.
"""

from typing import Any

from fastapi.openapi.utils import get_openapi

# OpenAPI Tags with descriptions
OPENAPI_TAGS = [
    {"name": "Root", "description": "Endpoint raíz con información general de la API"},
    {
        "name": "Predictions",
        "description": """
        Endpoints para predicción de consumo energético.

        Soporta predicciones individuales y batch con modelos ensemble de ML.
        Incluye intervalos de confianza y metadata completa de predicción.
        """,
    },
    {
        "name": "Health",
        "description": """
        Health check y monitoreo del servicio.

        Útil para load balancers, sistemas de orquestación (Kubernetes),
        y monitoreo de disponibilidad del servicio.
        """,
    },
    {
        "name": "Model",
        "description": """
        Información y métricas del modelo de ML.

        Proporciona metadata del modelo, métricas de entrenamiento y producción,
        información de features y arquitectura del ensemble.
        """,
    },
]

# API Contact Information
API_CONTACT = {
    "name": "MLOps Team - Proyecto Atreides",
    "email": "mlops@atreides.com",
    "url": "https://github.com/DanteA0179/mlops_proyecto_atreides",
}

# License Information
API_LICENSE = {"name": "MIT License", "url": "https://opensource.org/licenses/MIT"}

# Extended API Description
API_DESCRIPTION = """
🔋 **Energy Optimization Copilot API**

API RESTful para predicción y optimización de consumo energético en la industria siderúrgica.

## 🎯 Características

- ✅ **Predicciones precisas**: Modelos ensemble con RMSE < 13 kWh
- ✅ **Batch processing**: Hasta 1000 predicciones por request
- ✅ **Intervalos de confianza**: Predicciones con estimación de incertidumbre
- ✅ **Monitoreo en tiempo real**: Health checks y métricas de producción
- ✅ **Production-ready**: Diseñado para deployment en Google Cloud Run

## 🚀 Inicio Rápido

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Predicción Individual
```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "nsm": 36000,
    "day_of_week": 1,
    "load_type": "Medium"
  }'
```

### 3. Información del Modelo
```bash
curl http://localhost:8000/model/info
```

## 📊 Modelos Disponibles

- **XGBoost**: Baseline con gradient boosting
- **LightGBM**: Optimizado para velocidad y memoria
- **CatBoost**: Manejo robusto de variables categóricas
- **Stacking Ensemble**: Combinación de modelos (default) ⭐
- **Chronos-2**: Foundation model para series temporales

## 🔍 Casos de Uso

1. **Predicción en Tiempo Real**: Estimar consumo para próximas horas
2. **What-If Analysis**: Evaluar impacto de cambios operacionales
3. **Optimización de Turnos**: Identificar ventanas de menor consumo
4. **Monitoreo de Anomalías**: Detectar desviaciones del patrón esperado
5. **Planificación de Mantenimiento**: Programar actividades en períodos de baja demanda

## 📈 Métricas de Performance

| Métrica | Valor |
|---------|-------|
| RMSE (Test) | 12.87 kWh |
| MAE (Test) | 9.32 kWh |
| R² Score | 0.9823 |
| Latencia P95 | < 100ms |
| Throughput | ~200 req/s |

## 🔗 Enlaces Útiles

- [Documentación Completa](https://github.com/DanteA0179/mlops_proyecto_atreides/docs)
- [Repositorio GitHub](https://github.com/DanteA0179/mlops_proyecto_atreides)
- [Reportar Issues](https://github.com/DanteA0179/mlops_proyecto_atreides/issues)
- [Guía de Inicio Rápido](/docs/api/QUICK_START.md)
- [Ejemplos de Código](/docs/api/EXAMPLES.md)

## 📝 Versiones

- **v1.0.0** (2025-11-16): Release inicial
  - 5 endpoints operacionales
  - Soporte para predicciones individuales y batch
  - Modelos ensemble con intervalos de confianza
  - Documentación OpenAPI completa
  - Health checks y métricas de producción
"""

# Terms of Service
API_TERMS_OF_SERVICE = "https://github.com/DanteA0179/mlops_proyecto_atreides/blob/main/LICENSE"


def get_custom_openapi_schema(app) -> dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced metadata.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance

    Returns
    -------
    Dict[str, Any]
        Custom OpenAPI schema

    Examples
    --------
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> schema = get_custom_openapi_schema(app)
    >>> assert "servers" in schema
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Add contact and license info
    openapi_schema["info"]["contact"] = API_CONTACT
    openapi_schema["info"]["license"] = API_LICENSE

    # Add custom fields
    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/DanteA0179/mlops_proyecto_atreides/main/docs/assets/logo.png",
        "altText": "Energy Optimization Copilot Logo",
    }

    # Add servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server (Local)"},
        {"url": "http://0.0.0.0:8000", "description": "Development server (Docker)"},
        {
            "url": "https://energy-optimization-api.run.app",
            "description": "Production server (GCP Cloud Run)",
        },
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Documentación completa del proyecto",
        "url": "https://github.com/DanteA0179/mlops_proyecto_atreides/tree/main/docs",
    }

    # Add security schemes (for future use)
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key para autenticación (próximamente)",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token (próximamente)",
        },
    }

    # Add additional metadata
    openapi_schema["info"]["x-api-id"] = "energy-optimization-api"
    openapi_schema["info"]["x-audience"] = "external-public"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Common response examples for error cases
ERROR_RESPONSES = {
    400: {
        "description": "Bad Request - Invalid input",
        "content": {"application/json": {"example": {"detail": "Invalid input parameters"}}},
    },
    422: {
        "description": "Unprocessable Entity - Validation error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "load_type"],
                            "msg": "load_type must be one of ['Light', 'Medium', 'Maximum']",
                            "type": "value_error",
                        }
                    ]
                }
            }
        },
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred", "error_id": "err_abc123"}
            }
        },
    },
    503: {
        "description": "Service Unavailable - Model not loaded",
        "content": {"application/json": {"example": {"detail": "Model service is not available"}}},
    },
}
