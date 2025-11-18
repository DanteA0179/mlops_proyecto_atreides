"""Main orchestration service for copilot functionality."""

import logging
import os
import time

from src.api.models.copilot_requests import ConversationMessage
from src.api.models.copilot_responses import ChatResponse
from src.api.services.conversation_manager import ConversationManager
from src.api.services.intent_parser import IntentParser
from src.api.services.model_service import ModelService
from src.api.services.parameter_extractor import ParameterExtractor
from src.api.services.prompt_builder import PromptBuilder
from src.api.services.response_formatter import ResponseFormatter
from src.llm.llm_factory import get_llm_client

logger = logging.getLogger(__name__)


class CopilotService:
    """
    Main orchestration service for copilot functionality.

    Coordinates:
    - Intent detection
    - Prompt building
    - LLM interaction
    - Model prediction (when needed)
    - Response formatting

    This service implements the core copilot logic by orchestrating
    multiple specialized components to process natural language queries
    about energy consumption.

    Examples
    --------
    >>> service = CopilotService()
    >>> response = await service.process_message(
    ...     "What will be the consumption tomorrow at 10am?",
    ...     "test-conversation-id"
    ... )
    >>> response.intent
    'prediction'
    """

    def __init__(self):
        """Initialize CopilotService with all required components."""
        self.llm_client = get_llm_client()  # Factory pattern from US-029
        self.intent_parser = IntentParser()
        self.prompt_builder = PromptBuilder()
        self.parameter_extractor = ParameterExtractor()
        self.conversation_manager = ConversationManager()
        self.response_formatter = ResponseFormatter()

        # Initialize ModelService with default model
        model_type = os.getenv("MODEL_TYPE", "stacking_ensemble")
        self.model_service = ModelService(model_type=model_type)

        # Load model on initialization
        try:
            self.model_service.load_model()
            logger.info("Model loaded successfully in CopilotService")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("CopilotService will continue but predictions will fail")

        logger.info("CopilotService initialized successfully")

    async def process_message(
        self,
        message: str,
        conversation_id: str,
        context: list[ConversationMessage] | None = None
    ) -> ChatResponse:
        """
        Process user message and generate response.

        Orchestrates the complete flow:
        1. Detect intent
        2. Get conversation context
        3. Build prompt
        4. Get LLM response
        5. Make prediction if needed
        6. Format response
        7. Save to history

        Parameters
        ----------
        message : str
            User query in natural language
        conversation_id : str
            UUID for conversation tracking
        context : list[ConversationMessage] | None
            Previous messages for context

        Returns
        -------
        ChatResponse
            Formatted response with LLM output and optional prediction

        Raises
        ------
        ValueError
            If message is empty or invalid
        TimeoutError
            If LLM request times out
        RuntimeError
            If processing fails
        """
        start_time = time.time()

        try:
            # Step 1: Detect intent
            intent = self.intent_parser.parse(message)
            logger.info(f"Detected intent: {intent} for message: {message[:50]}...")

            # Step 2: Get conversation context
            if context is None:
                context_dicts = self.conversation_manager.get_context(conversation_id)
            else:
                # Convert ConversationMessage to dict
                context_dicts = [
                    {"role": msg.role, "content": msg.content}
                    for msg in context
                ]

            # Step 3: Build prompt
            prompt = self.prompt_builder.build_prompt(
                intent=intent,
                message=message,
                context=context_dicts
            )

            # Step 4: Get LLM response
            logger.debug(f"Sending {len(prompt)} messages to LLM")
            llm_response = self.llm_client.chat(prompt)
            logger.info(f"LLM response received: {len(llm_response)} chars")

            # Step 5: If prediction intent, call model
            prediction_data = None
            if intent == "prediction":
                prediction_data = await self._make_prediction(message)

            # Step 6: Format response
            processing_time = (time.time() - start_time) * 1000

            response = self.response_formatter.format(
                llm_response=llm_response,
                intent=intent,
                prediction_data=prediction_data,
                conversation_id=conversation_id,
                processing_time_ms=processing_time
            )

            # Step 7: Save to conversation history
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message
            )
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response.response
            )

            logger.info(
                f"Message processed successfully in {processing_time:.2f}ms "
                f"(intent={intent}, has_prediction={prediction_data is not None})"
            )

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            raise

    async def _make_prediction(self, message: str) -> dict | None:
        """
        Make prediction based on extracted parameters.

        Parameters
        ----------
        message : str
            User message to extract parameters from

        Returns
        -------
        dict | None
            Prediction data or None if prediction fails
        """
        try:
            # Extract parameters from message
            params = self.parameter_extractor.extract(message)
            logger.info(f"Extracted parameters: {params}")

            # Prepare features for model
            # Note: This is a simplified version. In production, you would
            # construct the full feature vector according to the model's expectations
            features_dict = {
                "Lagging_Current_Reactive.Power_kVarh": params.get("lagging_reactive_power", 23.45),
                "Leading_Current_Reactive_Power_kVarh": params.get("leading_reactive_power", 12.30),
                "CO2(tCO2)": params.get("co2", 0.05),
                "Lagging_Current_Power_Factor": params.get("lagging_power_factor", 0.85),
                "Leading_Current_Power_Factor": params.get("leading_power_factor", 0.92),
                "NSM": params.get("nsm", 36000),
                "WeekStatus": params.get("week_status", "Weekday"),
                "Day_of_week": params.get("day_of_week", 0),
                "Load_Type": params.get("load_type", "Medium")
            }

            # Make prediction using model service
            # Note: The actual implementation depends on the ModelService interface
            # For now, we'll use a placeholder prediction
            # TODO: Integrate with actual ModelService.predict() method

            prediction_value = 44.5  # Placeholder
            logger.info(f"Prediction made: {prediction_value} kWh")

            prediction_data = {
                "predicted_usage_kwh": prediction_value,
                "model_version": self.model_service.model_type,
                "features_used": features_dict
            }

            return prediction_data

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            # Return None to continue without prediction
            return None

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear conversation history.

        Parameters
        ----------
        conversation_id : str
            Conversation UUID to clear
        """
        self.conversation_manager.clear_conversation(conversation_id)
        logger.info(f"Cleared conversation {conversation_id}")

    def get_conversation_history(self, conversation_id: str) -> list[dict]:
        """
        Get conversation history.

        Parameters
        ----------
        conversation_id : str
            Conversation UUID

        Returns
        -------
        list[dict]
            Conversation messages
        """
        return self.conversation_manager.get_context(conversation_id)
