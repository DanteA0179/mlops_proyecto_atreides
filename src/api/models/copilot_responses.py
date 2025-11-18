"""Response models for copilot endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionData(BaseModel):
    """Prediction data when intent is 'prediction'."""

    predicted_usage_kwh: float = Field(..., description="Predicted energy consumption")
    confidence_score: float | None = Field(None, ge=0.0, le=1.0)
    model_version: str
    features_used: dict[str, float | str | int]


class ChatResponse(BaseModel):
    """Response model for /copilot/chat endpoint."""

    response: str = Field(
        ...,
        description="LLM-generated response to user query"
    )

    conversation_id: str = Field(
        ...,
        description="UUID for conversation tracking"
    )

    intent: Literal["prediction", "analysis", "what-if", "explanation", "general"] = Field(
        ...,
        description="Detected intent of user query"
    )

    prediction_data: PredictionData | None = Field(
        default=None,
        description="Prediction results if intent is 'prediction'"
    )

    sources: list[str] | None = Field(
        default=None,
        description="Data sources used (for future RAG implementation)"
    )

    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Based on the model, the predicted consumption...",
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "intent": "prediction",
                    "prediction_data": {
                        "predicted_usage_kwh": 44.456,
                        "model_version": "stacking_ensemble_v1",
                        "features_used": {"lagging_reactive_power": 23.45}
                    },
                    "processing_time_ms": 2345.67
                }
            ]
        }
    }
