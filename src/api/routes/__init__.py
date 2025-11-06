"""
API routes for Energy Optimization API.
"""

from src.api.routes import health, model, predict

__all__ = ["predict", "health", "model"]
