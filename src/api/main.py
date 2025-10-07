"""
FastAPI Application - Energy Optimization Copilot API

Main application entry point for the Energy Optimization API.
Provides endpoints for energy consumption predictions and health checks.
"""

from datetime import datetime
from typing import Dict

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create FastAPI app instance
app = FastAPI(
    title="Energy Optimization Copilot API",
    description="AI-powered energy consumption prediction and optimization for steel industry",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    tags=["Health"],
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, str],
    summary="Health Check",
    description="Returns the health status of the API service",
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dict with status, timestamp, and service information
    """
    return {
        "status": "healthy",
        "service": "energy-optimization-api",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(
    "/",
    tags=["Root"],
    summary="Root Endpoint",
    description="Welcome message and API information",
)
async def root() -> Dict[str, str]:
    """
    Root endpoint with API information.

    Returns:
        Welcome message and links to documentation
    """
    return {
        "message": "Welcome to Energy Optimization Copilot API",
        "docs": "/docs",
        "health": "/health",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
