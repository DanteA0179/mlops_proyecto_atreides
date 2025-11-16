"""
Prediction endpoints for Energy Optimization API.

This module provides /predict and /predict/batch endpoints.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, status

from src.api.models.requests import BatchPredictionRequest, PredictionRequest
from src.api.models.responses import (
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchPredictionSummary,
    PredictionResponse,
)
from src.monitoring.log_prediction import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])

# Global service instances (injected at startup)
model_service = None
feature_service = None


def set_services(model_svc, feature_svc):
    """
    Set global service instances.

    Parameters
    ----------
    model_svc : ModelService
        Model service instance
    feature_svc : FeatureService
        Feature service instance
    """
    global model_service, feature_service
    model_service = model_svc
    feature_service = feature_svc


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Energy Consumption",
    description="Predicts energy consumption for a single observation based on provided features",
)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    Predict energy consumption for single observation.

    Parameters
    ----------
    request : PredictionRequest
        Input features for prediction

    Returns
    -------
    PredictionResponse
        Prediction with confidence intervals and metadata

    Raises
    ------
    HTTPException 422
        Validation error in request data
    HTTPException 500
        Internal server error during prediction
    """
    try:
        logger.info(f"Received prediction request: {request.model_dump()}")

        # Transform request to model-ready features
        features = feature_service.transform_request(request)
        logger.debug(f"Transformed features shape: {features.shape}")

        # Make prediction
        prediction = model_service.predict(features)
        logger.info(f"Prediction result: {prediction[0]}")

        # Log prediction for monitoring
        try:
            feature_names = feature_service.get_feature_names()
            features_dict = dict(zip(feature_names, features[0].tolist()))
            log_prediction(features_dict, float(prediction[0]))
        except Exception as e:
            logger.warning(f"Failed to log prediction for monitoring: {e}")

        # Calculate confidence intervals (if model supports it)
        ci_lower, ci_upper = model_service.predict_interval(features, alpha=0.05)

        # Generate response
        response = PredictionResponse(
            predicted_usage_kwh=float(prediction[0]),
            confidence_interval_lower=float(ci_lower) if ci_lower else None,
            confidence_interval_upper=float(ci_upper) if ci_upper else None,
            model_version=model_service.model_version,
            model_type=model_service.model_type,
            prediction_timestamp=datetime.utcnow().isoformat() + "Z",
            features_used=features.shape[1],
            prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
        )

        logger.info(f"Successfully generated prediction {response.prediction_id}")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Predict Energy Consumption",
    description="Predicts energy consumption for multiple observations in a single request",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict energy consumption for batch of observations.

    Parameters
    ----------
    request : BatchPredictionRequest
        List of prediction requests (max 1000)

    Returns
    -------
    BatchPredictionResponse
        Batch predictions with summary statistics

    Raises
    ------
    HTTPException 400
        Batch validation error
    HTTPException 422
        Validation error in request data
    HTTPException 500
        Internal server error during prediction
    """
    try:
        start_time = time.time()

        logger.info(f"Received batch prediction request: {len(request.predictions)} items")

        # Transform batch to model-ready features
        features = feature_service.transform_batch(request.predictions)
        logger.debug(f"Transformed batch features shape: {features.shape}")

        # Make batch prediction
        predictions = model_service.predict(features)
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")

        # Generate prediction items
        prediction_items: List[BatchPredictionItem] = []
        for pred in predictions:
            prediction_items.append(
                BatchPredictionItem(
                    predicted_usage_kwh=float(pred),
                    prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                )
            )

        # Calculate summary statistics
        processing_time = (time.time() - start_time) * 1000  # ms
        summary = BatchPredictionSummary(
            total_predictions=len(predictions),
            avg_predicted_usage=float(predictions.mean()),
            min_predicted_usage=float(predictions.min()),
            max_predicted_usage=float(predictions.max()),
            processing_time_ms=round(processing_time, 2),
        )

        # Generate response
        response = BatchPredictionResponse(
            predictions=prediction_items,
            summary=summary,
            model_version=model_service.model_version,
            batch_timestamp=datetime.utcnow().isoformat() + "Z",
        )

        logger.info(
            f"Batch prediction successful: {len(predictions)} predictions "
            f"in {processing_time:.2f}ms"
        )
        return response

    except ValueError as e:
        logger.error(f"Batch validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )
