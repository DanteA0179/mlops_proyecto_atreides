"""
Health check endpoint for Energy Optimization API.

This module provides /health endpoint for service monitoring.
"""

import logging
import time
from datetime import datetime

import psutil
from fastapi import APIRouter, status

from src.api.models.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

# Global service instances and start time
model_service = None
start_time = time.time()


def set_services(model_svc):
    """
    Set global service instances.

    Parameters
    ----------
    model_svc : ModelService
        Model service instance
    """
    global model_service
    model_service = model_svc


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="""
    Retorna el estado de salud del servicio con métricas del sistema.

    Este endpoint es utilizado por load balancers, sistemas de orquestación
    (Kubernetes, Cloud Run) y herramientas de monitoreo para verificar
    la disponibilidad y estado del servicio.
    """,
    response_description="Estado de salud del servicio",
    responses={
        200: {
            "description": "Servicio saludable",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "energy-optimization-api",
                        "version": "1.0.0",
                        "timestamp": "2025-11-16T10:30:00Z",
                        "model_loaded": True,
                        "model_version": "stacking_ensemble_v1",
                        "uptime_seconds": 3600.50,
                        "memory_usage_mb": 512.34,
                        "cpu_usage_percent": 15.20,
                    }
                }
            },
        }
    },
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    ## Descripción
    Proporciona información sobre el estado del servicio, incluyendo:
    - **Estado general**: healthy, degraded, unhealthy
    - **Modelo cargado**: Si el modelo de ML está disponible
    - **Métricas del sistema**: Uso de CPU, memoria y uptime

    ## Estados Posibles
    - **healthy**: Servicio operativo, modelo cargado, recursos normales
    - **degraded**: Servicio operativo pero con recursos limitados (CPU>80% o Memoria>1GB)
    - **unhealthy**: Modelo no cargado o servicio con errores críticos

    ## Uso en Load Balancers
    Configure health checks en su balanceador:
    ```yaml
    health_check:
      path: /health
      interval: 30s
      timeout: 5s
      healthy_threshold: 2
      unhealthy_threshold: 3
    ```

    ## Ejemplo curl
    ```bash
    curl http://localhost:8000/health
    ```

    ## Ejemplo Python
    ```python
    import requests

    response = requests.get("http://localhost:8000/health")
    health = response.json()

    if health["status"] == "healthy":
        print("✅ Servicio operativo")
    else:
        print(f"⚠️ Estado: {health['status']}")
    ```

    ## Monitoreo Continuo
    ```bash
    # Check cada 10 segundos
    watch -n 10 curl -s http://localhost:8000/health | jq
    ```

    Returns
    -------
    HealthResponse
        Health status with system metrics

    Examples
    --------
    >>> response = await health_check()
    >>> response.status
    'healthy'
    """
    try:
        # Check model status
        model_loaded = model_service.is_loaded if model_service else False
        model_version = model_service.model_version if model_loaded else None

        # Get system metrics
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent(interval=0.1)
        uptime = time.time() - start_time

        # Determine health status
        if not model_loaded:
            health_status = "unhealthy"
        elif memory_usage > 1000 or cpu_usage > 80:
            health_status = "degraded"
        else:
            health_status = "healthy"

        response = HealthResponse(
            status=health_status,
            service="energy-optimization-api",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_loaded=model_loaded,
            model_version=model_version,
            uptime_seconds=round(uptime, 2),
            memory_usage_mb=round(memory_usage, 2),
            cpu_usage_percent=round(cpu_usage, 2),
        )

        logger.info(f"Health check: status={health_status}, model_loaded={model_loaded}")
        return response

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)

        # Return degraded status on error
        return HealthResponse(
            status="degraded",
            service="energy-optimization-api",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_loaded=False,
            model_version=None,
            uptime_seconds=round(time.time() - start_time, 2),
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
        )
