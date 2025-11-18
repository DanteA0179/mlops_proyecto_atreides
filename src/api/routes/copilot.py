"""Copilot endpoint for natural language queries."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.models.copilot_requests import ChatRequest
from src.api.models.copilot_responses import ChatResponse
from src.api.services.copilot_service import CopilotService

router = APIRouter(prefix="/copilot", tags=["Copilot"])
logger = logging.getLogger(__name__)

# Global instance (singleton pattern for service)
_copilot_service: CopilotService | None = None


def get_copilot_service() -> CopilotService:
    """
    Get or create CopilotService instance.

    Returns
    -------
    CopilotService
        Singleton instance of CopilotService
    """
    global _copilot_service
    if _copilot_service is None:
        _copilot_service = CopilotService()
    return _copilot_service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process natural language queries about energy consumption.

    This endpoint accepts user queries in natural language and returns
    intelligent responses powered by LLM and ML models.

    Supports multiple intents:
    - **Prediction**: "What will be the consumption tomorrow at 10am?"
    - **Analysis**: "Why was there a spike last Friday?"
    - **What-if**: "How much would I save if I shift production to 2am?"
    - **Explanation**: "Explain the factors affecting consumption"
    - **General**: "What is power factor?"

    Parameters
    ----------
    request : ChatRequest
        User message and optional conversation_id and context

    Returns
    -------
    ChatResponse
        LLM response with optional prediction data and processing metadata

    Raises
    ------
    HTTPException
        400: Invalid request (validation error)
        500: Internal server error (processing failure)
        503: Service unavailable (LLM or model not available)
        504: Gateway timeout (request took too long)

    Examples
    --------
    Request:
    ```json
    {
        "message": "What will be the consumption tomorrow at 10am with Medium load?",
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```

    Response:
    ```json
    {
        "response": "Based on the model predictions...",
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "intent": "prediction",
        "prediction_data": {
            "predicted_usage_kwh": 44.456,
            "model_version": "stacking_ensemble",
            "features_used": {...}
        },
        "processing_time_ms": 2345.67
    }
    ```
    """
    try:
        copilot_service = get_copilot_service()

        # Process message
        response = await copilot_service.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            context=request.context
        )

        logger.info(
            f"Chat request processed successfully: "
            f"conversation_id={response.conversation_id}, "
            f"intent={response.intent}, "
            f"processing_time={response.processing_time_ms:.2f}ms"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        raise HTTPException(
            status_code=504,
            detail="Request timeout. The LLM or model took too long to respond."
        )

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again later."
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please contact support if the problem persists."
        )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str) -> dict:
    """
    Clear conversation history.

    Deletes all messages associated with the given conversation ID.

    Parameters
    ----------
    conversation_id : str
        UUID of the conversation to clear

    Returns
    -------
    dict
        Success message

    Examples
    --------
    Request:
    ```
    DELETE /copilot/conversation/550e8400-e29b-41d4-a716-446655440000
    ```

    Response:
    ```json
    {
        "message": "Conversation cleared successfully",
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    """
    try:
        copilot_service = get_copilot_service()
        copilot_service.clear_conversation(conversation_id)

        logger.info(f"Conversation cleared: {conversation_id}")

        return {
            "message": "Conversation cleared successfully",
            "conversation_id": conversation_id
        }

    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str) -> dict:
    """
    Get conversation history.

    Retrieves all messages for the given conversation ID.

    Parameters
    ----------
    conversation_id : str
        UUID of the conversation

    Returns
    -------
    dict
        Conversation messages and metadata

    Examples
    --------
    Request:
    ```
    GET /copilot/conversation/550e8400-e29b-41d4-a716-446655440000
    ```

    Response:
    ```json
    {
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "messages": [
            {
                "role": "user",
                "content": "What is power factor?",
                "timestamp": "2025-11-17T10:30:00"
            },
            {
                "role": "assistant",
                "content": "Power factor is...",
                "timestamp": "2025-11-17T10:30:02"
            }
        ],
        "message_count": 2
    }
    ```
    """
    try:
        copilot_service = get_copilot_service()
        messages = copilot_service.get_conversation_history(conversation_id)

        logger.info(f"Retrieved conversation history: {conversation_id} ({len(messages)} messages)")

        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "message_count": len(messages)
        }

    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
