"""
FastAPI Application - Energy Optimization Copilot API.

Main application entry point for the Energy Optimization API.
Provides endpoints for energy consumption predictions, health checks,
and model information.

Examples
--------
Run the API server:
    $ python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Or directly:
    $ python src/api/main.py
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware.error_handler import (
    general_exception_handler,
    validation_exception_handler,
)
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.routes import health, model, predict
from src.api.services.feature_engineering import FeatureService
from src.api.services.model_service import ModelService
from src.api.utils.config import config, setup_logging

# Setup logging
setup_logging(log_level=config.LOG_LEVEL, log_file=config.LOG_FILE)
logger = logging.getLogger(__name__)

# Global service instances
model_service_instance = None
feature_service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for loading/unloading services.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance

    Yields
    ------
    None
        Control is yielded to the application
    """
    # Startup
    logger.info("Starting Energy Optimization API...")
    global model_service_instance, feature_service_instance

    try:
        # Initialize services
        logger.info(f"Loading model: {config.MODEL_TYPE}")
        model_service_instance = ModelService(
            model_type=config.MODEL_TYPE,
            mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
        )
        model_service_instance.load_model()
        logger.info(f"Model loaded successfully: {model_service_instance.model_version}")

        logger.info("Loading feature service...")
        feature_service_instance = FeatureService()
        logger.info("Feature service loaded successfully")

        # Inject services into routes
        predict.set_services(model_service_instance, feature_service_instance)
        health.set_services(model_service_instance)
        model.set_services(model_service_instance, feature_service_instance)

        logger.info("Energy Optimization API started successfully")

    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Energy Optimization API...")
    model_service_instance = None
    feature_service_instance = None
    logger.info("Shutdown complete")


# Import OpenAPI configuration
from src.api.utils.openapi_config import (
    API_CONTACT,
    API_DESCRIPTION,
    API_LICENSE,
    API_TERMS_OF_SERVICE,
    OPENAPI_TAGS,
    get_custom_openapi_schema,
)

# Create FastAPI app instance
app = FastAPI(
    title=config.APP_NAME,
    description=API_DESCRIPTION,
    version=config.APP_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE,
    terms_of_service=API_TERMS_OF_SERVICE,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add error handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(model.router)

# Override default OpenAPI schema with custom one
app.openapi = lambda: get_custom_openapi_schema(app)


@app.get(
    "/",
    tags=["Root"],
    summary="Root Endpoint",
    description="Welcome message and API information",
)
async def root() -> Dict:
    """
    Root endpoint with API information.

    Returns
    -------
    Dict
        Welcome message and links to documentation

    Examples
    --------
    >>> response = await root()
    >>> response["message"]
    'Welcome to Energy Optimization Copilot API'
    """
    return {
        "message": "Welcome to Energy Optimization Copilot API",
        "version": config.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "model_metrics": "/model/metrics",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
