"""
Pydantic Models for Energy Optimization API.

This module exports all request and response models.
"""

from src.api.models.requests import BatchPredictionRequest, PredictionRequest
from src.api.models.responses import (
    BaseModelInfo,
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchPredictionSummary,
    FeatureInfo,
    HealthResponse,
    MetaModelInfo,
    ModelInfoResponse,
    ModelMetricsResponse,
    PredictionDistribution,
    PredictionResponse,
    ProductionMetrics,
    SystemHealth,
    TrainingDatasetInfo,
)

__all__ = [
    # Request models
    "PredictionRequest",
    "BatchPredictionRequest",
    # Response models
    "PredictionResponse",
    "BatchPredictionItem",
    "BatchPredictionSummary",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "ModelMetricsResponse",
    # Supporting models
    "BaseModelInfo",
    "MetaModelInfo",
    "FeatureInfo",
    "TrainingDatasetInfo",
    "ProductionMetrics",
    "PredictionDistribution",
    "SystemHealth",
]
