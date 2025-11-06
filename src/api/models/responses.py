"""
Pydantic Response Models for Energy Optimization API.

This module contains all response models used by the API endpoints,
ensuring consistent and validated responses.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response model for single energy consumption prediction.

    Includes prediction, confidence intervals, and metadata.

    Attributes
    ----------
    predicted_usage_kwh : float
        Predicted energy consumption in kWh
    confidence_interval_lower : Optional[float]
        Lower bound of 95% confidence interval (if available)
    confidence_interval_upper : Optional[float]
        Upper bound of 95% confidence interval (if available)
    model_version : str
        Version of the model used for prediction
    model_type : str
        Type of model architecture
    prediction_timestamp : str
        ISO 8601 timestamp of prediction
    features_used : int
        Number of features used in prediction
    prediction_id : str
        Unique identifier for this prediction
    """

    predicted_usage_kwh: float = Field(
        ..., description="Predicted energy consumption in kWh", examples=[45.67]
    )
    confidence_interval_lower: Optional[float] = Field(
        None,
        description="Lower bound of 95% confidence interval (if available)",
        examples=[42.10],
    )
    confidence_interval_upper: Optional[float] = Field(
        None,
        description="Upper bound of 95% confidence interval (if available)",
        examples=[49.24],
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
        examples=["lightgbm_ensemble_v1"],
    )
    model_type: str = Field(
        ...,
        description="Type of model architecture",
        examples=["stacking_ensemble"],
    )
    prediction_timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of prediction",
        examples=["2025-11-05T10:30:00Z"],
    )
    features_used: int = Field(
        ..., description="Number of features used in prediction", examples=[18]
    )
    prediction_id: str = Field(
        ...,
        description="Unique identifier for this prediction",
        examples=["pred_8f3a9b2c"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_usage_kwh": 45.67,
                "confidence_interval_lower": 42.10,
                "confidence_interval_upper": 49.24,
                "model_version": "lightgbm_ensemble_v1",
                "model_type": "stacking_ensemble",
                "prediction_timestamp": "2025-11-05T10:30:00Z",
                "features_used": 18,
                "prediction_id": "pred_8f3a9b2c",
            }
        }
    }


class BatchPredictionItem(BaseModel):
    """
    Individual prediction in batch response.

    Attributes
    ----------
    predicted_usage_kwh : float
        Predicted energy consumption in kWh
    prediction_id : str
        Unique identifier for this prediction
    """

    predicted_usage_kwh: float = Field(..., description="Predicted energy consumption")
    prediction_id: str = Field(..., description="Unique prediction identifier")


class BatchPredictionSummary(BaseModel):
    """
    Summary statistics for batch predictions.

    Attributes
    ----------
    total_predictions : int
        Total number of predictions in batch
    avg_predicted_usage : float
        Average predicted energy usage
    min_predicted_usage : float
        Minimum predicted energy usage
    max_predicted_usage : float
        Maximum predicted energy usage
    processing_time_ms : float
        Time taken to process batch in milliseconds
    """

    total_predictions: int = Field(..., description="Total predictions in batch")
    avg_predicted_usage: float = Field(..., description="Average predicted usage")
    min_predicted_usage: float = Field(..., description="Minimum predicted usage")
    max_predicted_usage: float = Field(..., description="Maximum predicted usage")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.

    Attributes
    ----------
    predictions : List[BatchPredictionItem]
        List of individual predictions
    summary : BatchPredictionSummary
        Statistical summary of the batch
    model_version : str
        Version of model used
    batch_timestamp : str
        ISO 8601 timestamp of batch processing
    """

    predictions: List[BatchPredictionItem] = Field(
        ..., description="List of predictions"
    )
    summary: BatchPredictionSummary = Field(..., description="Batch summary statistics")
    model_version: str = Field(..., description="Model version used")
    batch_timestamp: str = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Attributes
    ----------
    status : Literal["healthy", "degraded", "unhealthy"]
        Current health status of the service
    service : str
        Service name
    version : str
        API version
    timestamp : str
        Current timestamp
    model_loaded : bool
        Whether model is loaded
    model_version : Optional[str]
        Version of loaded model
    uptime_seconds : float
        Service uptime in seconds
    memory_usage_mb : float
        Current memory usage in MB
    cpu_usage_percent : float
        Current CPU usage percentage
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Health status"
    )
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Model loaded status")
    model_version: Optional[str] = Field(None, description="Model version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percent")


class BaseModelInfo(BaseModel):
    """
    Information about a base model in ensemble.

    Attributes
    ----------
    name : str
        Base model name
    contribution_pct : float
        Percentage contribution to ensemble
    """

    name: str = Field(..., description="Base model name")
    contribution_pct: float = Field(..., description="Contribution percentage")


class MetaModelInfo(BaseModel):
    """
    Information about meta-model in ensemble.

    Attributes
    ----------
    type : str
        Type of meta-model
    max_depth : Optional[int]
        Maximum tree depth (if applicable)
    n_estimators : Optional[int]
        Number of estimators (if applicable)
    """

    type: str = Field(..., description="Meta-model type")
    max_depth: Optional[int] = Field(None, description="Maximum depth")
    n_estimators: Optional[int] = Field(None, description="Number of estimators")


class FeatureInfo(BaseModel):
    """
    Information about a feature.

    Attributes
    ----------
    name : str
        Feature name
    type : str
        Feature data type
    importance : str
        Feature importance level (high/medium/low)
    description : str
        Human-readable description
    """

    name: str = Field(..., description="Feature name")
    type: str = Field(..., description="Feature type")
    importance: str = Field(..., description="Feature importance")
    description: str = Field(..., description="Feature description")


class TrainingDatasetInfo(BaseModel):
    """
    Information about training dataset.

    Attributes
    ----------
    name : str
        Dataset name
    samples : int
        Number of samples
    features : int
        Number of features
    """

    name: str = Field(..., description="Dataset name")
    samples: int = Field(..., description="Number of samples")
    features: int = Field(..., description="Number of features")


class ModelInfoResponse(BaseModel):
    """
    Response model for model information endpoint.

    Attributes
    ----------
    model_type : str
        Type of model architecture
    model_version : str
        Model version identifier
    model_name : str
        Human-readable model name
    trained_on : str
        Training timestamp
    training_dataset : TrainingDatasetInfo
        Training dataset information
    base_models : Optional[List[BaseModelInfo]]
        Base models (for ensembles)
    meta_model : Optional[MetaModelInfo]
        Meta-model information (for ensembles)
    features : List[FeatureInfo]
        List of features used
    training_metrics : Dict[str, float]
        Training performance metrics
    mlflow_run_id : str
        MLflow run identifier
    artifact_location : str
        Model artifact location
    """

    model_type: str = Field(..., description="Model architecture type")
    model_version: str = Field(..., description="Model version")
    model_name: str = Field(..., description="Model name")
    trained_on: str = Field(..., description="Training timestamp")
    training_dataset: TrainingDatasetInfo = Field(
        ..., description="Training dataset info"
    )
    base_models: Optional[List[BaseModelInfo]] = Field(
        None, description="Base models (ensembles)"
    )
    meta_model: Optional[MetaModelInfo] = Field(None, description="Meta-model info")
    features: List[FeatureInfo] = Field(..., description="Feature list")
    training_metrics: Dict[str, float] = Field(..., description="Training metrics")
    mlflow_run_id: str = Field(..., description="MLflow run ID")
    artifact_location: str = Field(..., description="Artifact location")


class ProductionMetrics(BaseModel):
    """
    Production metrics for model.

    Attributes
    ----------
    total_predictions : int
        Total predictions made
    predictions_last_24h : int
        Predictions in last 24 hours
    avg_prediction_time_ms : float
        Average prediction time
    p95_prediction_time_ms : float
        95th percentile prediction time
    p99_prediction_time_ms : float
        99th percentile prediction time
    error_rate_percent : float
        Error rate percentage
    """

    total_predictions: int = Field(..., description="Total predictions")
    predictions_last_24h: int = Field(..., description="Predictions last 24h")
    avg_prediction_time_ms: float = Field(..., description="Average prediction time")
    p95_prediction_time_ms: float = Field(..., description="P95 prediction time")
    p99_prediction_time_ms: float = Field(..., description="P99 prediction time")
    error_rate_percent: float = Field(..., description="Error rate percentage")


class PredictionDistribution(BaseModel):
    """
    Distribution statistics for predictions.

    Attributes
    ----------
    min : float
        Minimum prediction value
    max : float
        Maximum prediction value
    mean : float
        Mean prediction value
    median : float
        Median prediction value
    std : float
        Standard deviation
    """

    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    mean: float = Field(..., description="Mean value")
    median: float = Field(..., description="Median value")
    std: float = Field(..., description="Standard deviation")


class SystemHealth(BaseModel):
    """
    System health metrics.

    Attributes
    ----------
    memory_usage_mb : float
        Memory usage in MB
    cpu_usage_percent : float
        CPU usage percentage
    uptime_seconds : float
        Uptime in seconds
    """

    memory_usage_mb: float = Field(..., description="Memory usage MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percent")
    uptime_seconds: float = Field(..., description="Uptime seconds")


class ModelMetricsResponse(BaseModel):
    """
    Response model for model metrics endpoint.

    Attributes
    ----------
    model_version : str
        Model version
    timestamp : str
        Current timestamp
    training_metrics : Dict[str, Any]
        Training metrics
    production_metrics : ProductionMetrics
        Production metrics
    load_type_distribution : Dict[str, int]
        Distribution by load type
    prediction_distribution : PredictionDistribution
        Prediction statistics
    system_health : SystemHealth
        System health metrics
    """

    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Current timestamp")
    training_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    production_metrics: ProductionMetrics = Field(..., description="Production metrics")
    load_type_distribution: Dict[str, int] = Field(
        ..., description="Load type distribution"
    )
    prediction_distribution: PredictionDistribution = Field(
        ..., description="Prediction distribution"
    )
    system_health: SystemHealth = Field(..., description="System health")
