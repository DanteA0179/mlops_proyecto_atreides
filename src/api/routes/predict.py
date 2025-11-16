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
    description="""
    Predice el consumo energético para una observación individual.
    
    Este endpoint utiliza un modelo ensemble de ML para predecir el consumo 
    de energía basado en características operacionales de la planta siderúrgica.
    Incluye intervalos de confianza al 95% para cuantificar la incertidumbre.
    """,
    response_description="Predicción exitosa con intervalos de confianza",
    responses={
        200: {
            "description": "Predicción exitosa",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_usage_kwh": 45.67,
                        "confidence_interval_lower": 42.10,
                        "confidence_interval_upper": 49.24,
                        "model_version": "stacking_ensemble_v1",
                        "model_type": "stacking_ensemble",
                        "prediction_timestamp": "2025-11-16T10:30:00Z",
                        "features_used": 18,
                        "prediction_id": "pred_8f3a9b2c"
                    }
                }
            }
        },
        422: {
            "description": "Error de validación",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "load_type"],
                                "msg": "load_type must be one of ['Light', 'Medium', 'Maximum']",
                                "type": "value_error"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Error interno del servidor"
        }
    }
)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    Predict energy consumption for single observation.
    
    ## Descripción
    Este endpoint recibe las características de operación de la planta siderúrgica
    y retorna una predicción del consumo energético junto con intervalos de confianza.
    
    ## Características de Entrada
    - **lagging_reactive_power**: Potencia reactiva en atraso (kVarh)
    - **leading_reactive_power**: Potencia reactiva en adelanto (kVarh)
    - **co2**: Emisiones de CO2 (tCO2)
    - **lagging_power_factor**: Factor de potencia en atraso (0-1)
    - **leading_power_factor**: Factor de potencia en adelanto (0-1)
    - **nsm**: Segundos desde medianoche (0-86400)
    - **day_of_week**: Día de la semana (0=Lunes, 6=Domingo)
    - **load_type**: Tipo de carga (Light, Medium, Maximum)
    
    ## Ejemplo curl
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
    
    ## Ejemplo Python
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium"
        }
    )
    prediction = response.json()
    print(f"Consumo predicho: {prediction['predicted_usage_kwh']} kWh")
    ```

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
    description="""
    Predice el consumo energético para múltiples observaciones en una sola petición.
    
    Este endpoint permite procesar hasta 1000 predicciones simultáneamente,
    optimizando el throughput para escenarios de planificación o análisis masivo.
    Incluye estadísticas agregadas del lote procesado.
    """,
    response_description="Predicciones batch exitosas con resumen estadístico",
    responses={
        200: {
            "description": "Batch procesado exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {"predicted_usage_kwh": 28.34, "prediction_id": "pred_abc123"},
                            {"predicted_usage_kwh": 45.67, "prediction_id": "pred_def456"}
                        ],
                        "summary": {
                            "total_predictions": 2,
                            "avg_predicted_usage": 37.00,
                            "min_predicted_usage": 28.34,
                            "max_predicted_usage": 45.67,
                            "processing_time_ms": 45.32
                        },
                        "model_version": "stacking_ensemble_v1",
                        "batch_timestamp": "2025-11-16T10:30:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Error de validación del batch",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Batch cannot exceed 1000 predictions"
                    }
                }
            }
        },
        422: {
            "description": "Error de validación en items del batch"
        },
        500: {
            "description": "Error interno del servidor"
        }
    }
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict energy consumption for batch of observations.
    
    ## Descripción
    Procesa múltiples predicciones en una sola petición HTTP, ideal para:
    - Planificación de turnos completos (24 horas)
    - Análisis what-if de múltiples escenarios
    - Optimización de horarios de producción
    - Generación de reportes masivos
    
    ## Límites
    - **Mínimo**: 1 predicción
    - **Máximo**: 1000 predicciones por request
    - **Timeout**: 30 segundos
    
    ## Ejemplo curl
    ```bash
    curl -X POST "http://localhost:8000/predict/batch" \\
      -H "Content-Type: application/json" \\
      -d '{
        "predictions": [
          {
            "lagging_reactive_power": 15.20,
            "leading_reactive_power": 8.50,
            "co2": 0.03,
            "lagging_power_factor": 0.88,
            "leading_power_factor": 0.95,
            "nsm": 7200,
            "day_of_week": 1,
            "load_type": "Light"
          },
          {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium"
          }
        ]
      }'
    ```
    
    ## Ejemplo Python
    ```python
    import requests
    
    batch_data = {
        "predictions": [
            {
                "lagging_reactive_power": 15.20,
                "leading_reactive_power": 8.50,
                "co2": 0.03,
                "lagging_power_factor": 0.88,
                "leading_power_factor": 0.95,
                "nsm": 7200,
                "day_of_week": 1,
                "load_type": "Light"
            },
            # ... más predicciones
        ]
    }
    
    response = requests.post(
        "http://localhost:8000/predict/batch",
        json=batch_data
    )
    result = response.json()
    print(f"Total predicciones: {result['summary']['total_predictions']}")
    print(f"Promedio: {result['summary']['avg_predicted_usage']} kWh")
    ```

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
