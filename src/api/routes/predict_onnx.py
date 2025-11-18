"""
ONNX Prediction Routes for Energy Optimization Copilot API.

This module provides FastAPI routes for ONNX model inference with
support for multiple model types and optimized performance.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from src.api.models.requests import BatchPredictionRequest, PredictionRequest
from src.api.models.responses import (
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchPredictionSummary,
    PredictionResponse,
)
from src.api.services.feature_engineering import FeatureService
from src.api.services.onnx_service import ONNXModelService, get_onnx_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict_onnx", tags=["ONNX Prediction"])

# Initialize feature service (singleton)
try:
    feature_service = FeatureService()
    logger.info("FeatureService initialized for ONNX routes")
except Exception as e:
    logger.error(f"Failed to initialize FeatureService: {e}")
    feature_service = None


@router.post("", response_model=PredictionResponse)
async def predict_onnx(
    request: PredictionRequest,
    model_type: str = Query(
        default="lightgbm_ensemble",
        description="ONNX model to use for prediction",
        enum=ONNXModelService.list_available_models(),
    ),
) -> PredictionResponse:
    """
    Predict energy consumption using ONNX model.

    Optimized endpoint with lower latency than /predict.
    Supports 8 different models via model_type parameter.

    Parameters
    ----------
    request : PredictionRequest
        Prediction request with input features
    model_type : str, default='lightgbm'
        ONNX model to use

    Returns
    -------
    PredictionResponse
        Prediction response with energy consumption

    Raises
    ------
    HTTPException
        If model loading or prediction fails
    """
    try:
        if feature_service is None:
            raise HTTPException(
                status_code=503,
                detail="Feature service not available. Check preprocessing pipeline.",
            )

        onnx_service = get_onnx_service(model_type=model_type, use_gpu=True)

        # Use centralized feature engineering service
        features = feature_service.transform_request(request)

        prediction = onnx_service.predict(features)

        return PredictionResponse(
            predicted_usage_kwh=float(prediction[0]),
            model_version=f"{model_type}_onnx",
            model_type="onnx",
            prediction_timestamp=datetime.utcnow().isoformat() + "Z",
            features_used=18,
            prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
        )

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {model_type}") from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_onnx_batch(
    request: BatchPredictionRequest,
    model_type: str = Query(
        default="lightgbm_ensemble",
        description="ONNX model to use for batch prediction",
        enum=ONNXModelService.list_available_models(),
    ),
) -> BatchPredictionResponse:
    """
    Batch prediction using ONNX model.

    Optimized for high throughput with batch processing.

    Parameters
    ----------
    request : BatchPredictionRequest
        Batch prediction request
    model_type : str, default='lightgbm'
        ONNX model to use

    Returns
    -------
    BatchPredictionResponse
        Batch prediction response

    Raises
    ------
    HTTPException
        If batch size exceeds limit or prediction fails
    """
    if len(request.predictions) > 1000:
        raise HTTPException(status_code=400, detail="Max batch size is 1000")

    try:
        if feature_service is None:
            raise HTTPException(
                status_code=503,
                detail="Feature service not available. Check preprocessing pipeline.",
            )

        onnx_service = get_onnx_service(model_type=model_type, use_gpu=True)

        # Use centralized feature engineering service for batch
        features_batch = feature_service.transform_batch(request.predictions)

        start = time.perf_counter()
        predictions = onnx_service.predict(features_batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        prediction_items = [
            BatchPredictionItem(
                predicted_usage_kwh=float(pred), prediction_id=f"pred_{uuid.uuid4().hex[:8]}"
            )
            for pred in predictions
        ]

        summary = BatchPredictionSummary(
            total_predictions=len(predictions),
            avg_predicted_usage=float(np.mean(predictions)),
            min_predicted_usage=float(np.min(predictions)),
            max_predicted_usage=float(np.max(predictions)),
            processing_time_ms=elapsed_ms,
        )

        return BatchPredictionResponse(
            predictions=prediction_items,
            summary=summary,
            model_version=f"{model_type}_onnx",
            batch_timestamp=datetime.utcnow().isoformat() + "Z",
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}") from e


@router.get("/models")
async def list_onnx_models() -> dict[str, Any]:
    """
    List all available ONNX models.

    Returns model names, types, sizes, and availability status.

    Returns
    -------
    dict[str, Any]
        Dictionary with model information
    """
    models = []

    for model_name in ONNXModelService.list_available_models():
        try:
            model_info = ONNXModelService.get_model_info(model_name)

            model_path = Path(model_info["path"])
            if model_path.exists():
                if model_path.is_dir():
                    size_bytes = sum(f.stat().st_size for f in model_path.rglob("*.onnx"))
                else:
                    size_bytes = model_path.stat().st_size
                size_mb = round(size_bytes / (1024 * 1024), 2)
            else:
                size_mb = 0

            models.append(
                {
                    "name": model_name,
                    "type": model_info["type"],
                    "size_mb": size_mb,
                    "available": model_info["exists"],
                    "is_ensemble": model_info["is_ensemble"],
                }
            )

        except Exception as e:
            logger.error(f"Error getting info for {model_name}: {e}")
            models.append(
                {
                    "name": model_name,
                    "type": "unknown",
                    "size_mb": 0,
                    "available": False,
                    "error": str(e),
                }
            )

    return {
        "total_models": len(models),
        "models": models,
        "default_model": "lightgbm_ensemble",
    }


@router.get("/benchmark")
async def get_onnx_benchmark() -> dict[str, Any]:
    """
    Get ONNX vs Original benchmark comparison for all models.

    Returns latency, throughput, and memory metrics by model type.

    Returns
    -------
    dict[str, Any]
        Benchmark comparison data

    Raises
    ------
    HTTPException
        If benchmark file not found
    """
    benchmark_path = Path("models/benchmarks/onnx_comparison.json")

    if not benchmark_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark data not found. Run benchmark script first.",
        )

    try:
        with open(benchmark_path) as f:
            data = json.load(f)

        return {
            "benchmark_date": data.get("report_date"),
            "hardware": data.get("hardware"),
            "models": data.get("models"),
            "summary": data.get("summary"),
        }

    except Exception as e:
        logger.error(f"Error reading benchmark data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error reading benchmark data: {str(e)}"
        ) from e


@router.get("/info/{model_name}")
async def get_model_info(model_name: str) -> dict[str, Any]:
    """
    Get detailed information about a specific ONNX model.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    dict[str, Any]
        Model information

    Raises
    ------
    HTTPException
        If model not found
    """
    try:
        model_info = ONNXModelService.get_model_info(model_name)

        model_path = Path(model_info["path"])
        if model_path.exists():
            if model_path.is_dir():
                size_bytes = sum(f.stat().st_size for f in model_path.rglob("*.onnx"))
                num_files = len(list(model_path.rglob("*.onnx")))
            else:
                size_bytes = model_path.stat().st_size
                num_files = 1
            size_mb = round(size_bytes / (1024 * 1024), 2)
        else:
            size_mb = 0
            num_files = 0

        metadata_path = (
            model_path.with_suffix(".json")
            if model_path.is_file()
            else model_path / "metadata.json"
        )
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        return {
            "name": model_name,
            "type": model_info["type"],
            "path": str(model_path),
            "exists": model_info["exists"],
            "is_ensemble": model_info["is_ensemble"],
            "size_mb": size_mb,
            "num_files": num_files,
            "metadata": metadata,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}") from e
