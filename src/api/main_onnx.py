"""
FastAPI Application - Energy Optimization API (ONNX Only).

Simplified version that only uses ONNX models, no pickle models.
"""

import logging
from datetime import datetime

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.error_handler import (
    general_exception_handler,
    validation_exception_handler,
)
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.routes import predict_onnx
from src.api.utils.config import config, setup_logging

# Setup logging
setup_logging(log_level=config.LOG_LEVEL, log_file=config.LOG_FILE)
logger = logging.getLogger(__name__)


# Create FastAPI app instance
app = FastAPI(
    title=config.APP_NAME + " (ONNX)",
    description=(
        "AI-powered energy consumption prediction using ONNX models. "
        "Optimized for production deployment with minimal dependencies."
    ),
    version=config.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add error handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers (ONNX only)
app.include_router(predict_onnx.router)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "energy-optimization-api-onnx",
        "version": config.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_available": ["lightgbm", "catboost", "lightgbm_ensemble"],
    }


@app.get(
    "/",
    tags=["Root"],
    summary="Root Endpoint",
    description="Welcome message and API information",
)
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Energy Optimization API (ONNX)",
        "version": config.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "predict_onnx": "/predict_onnx",
            "batch_predict_onnx": "/predict_onnx/batch",
            "list_models": "/predict_onnx/models",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main_onnx:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
