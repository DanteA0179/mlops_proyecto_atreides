"""Response formatting for copilot endpoint."""

from typing import Literal

from src.api.models.copilot_responses import ChatResponse, PredictionData

Intent = Literal["prediction", "analysis", "what-if", "explanation", "general"]


class ResponseFormatter:
    """
    Formats LLM response into ChatResponse model.

    Handles conversion of raw LLM output and prediction data
    into structured API response format.
    """

    def format(
        self,
        llm_response: str,
        intent: Intent,
        prediction_data: dict | None,
        conversation_id: str,
        processing_time_ms: float
    ) -> ChatResponse:
        """
        Format response for API.

        Converts raw LLM output and optional prediction data into
        a structured ChatResponse object.

        Parameters
        ----------
        llm_response : str
            Raw LLM response text
        intent : Intent
            Detected intent category
        prediction_data : dict | None
            Prediction results if available
        conversation_id : str
            Conversation UUID
        processing_time_ms : float
            Total processing time in milliseconds

        Returns
        -------
        ChatResponse
            Formatted API response

        Examples
        --------
        >>> formatter = ResponseFormatter()
        >>> response = formatter.format(
        ...     "The consumption will be 44.5 kWh",
        ...     "prediction",
        ...     {"predicted_usage_kwh": 44.5, "model_version": "v1", "features_used": {}},
        ...     "test-id",
        ...     1234.56
        ... )
        >>> response.intent
        'prediction'
        """
        # Build prediction data model if available
        prediction_model = None
        if prediction_data:
            prediction_model = PredictionData(**prediction_data)

        return ChatResponse(
            response=llm_response,
            conversation_id=conversation_id,
            intent=intent,
            prediction_data=prediction_model,
            processing_time_ms=processing_time_ms
        )
