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
    description="Returns the health status of the API service with system metrics",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, model status, and system metrics.

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
