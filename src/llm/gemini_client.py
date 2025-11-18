"""
Gemini client for cloud-based LLM inference.

This module provides a Python client for interacting with Google's Gemini API
to run Gemini 2.0 Flash model in production for natural language processing tasks.
"""

import logging
import os

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for Google Gemini API to perform cloud LLM inference.

    This client provides methods to generate text and handle chat conversations
    using Google's Gemini 2.0 Flash model through the Gemini API.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key, defaults to GEMINI_API_KEY env var
    model : str, optional
        Model name to use, defaults to GEMINI_MODEL env var or gemini-2.0-flash-exp
    temperature : float, optional
        Sampling temperature (0.0 to 2.0), defaults to 0.7
    max_tokens : int, optional
        Maximum tokens to generate, defaults to 1000

    Examples
    --------
    >>> client = GeminiClient(api_key="your-api-key")
    >>> response = client.generate("What is energy consumption?")
    >>> print(response)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ):
        """Initialize Gemini client with configuration."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")

        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.temperature = temperature or float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("GEMINI_MAX_TOKENS", "1000"))

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            logger.info(f"Initialized GeminiClient with model={self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise RuntimeError(f"Failed to initialize Gemini: {e}") from e

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """
        Generate text completion from a prompt.

        Parameters
        ----------
        prompt : str
            Input text prompt
        temperature : float, optional
            Override default temperature
        max_tokens : int, optional
            Override default max_tokens

        Returns
        -------
        str
            Generated text response

        Raises
        ------
        ValueError
            If prompt is empty
        RuntimeError
            If API returns an error

        Examples
        --------
        >>> client = GeminiClient(api_key="your-api-key")
        >>> response = client.generate("Explain energy consumption in one sentence.")
        >>> print(response)
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        generation_config = genai.GenerationConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=max_tokens or self.max_tokens
        )

        try:
            logger.debug(f"Sending generate request: prompt_len={len(prompt)}")
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Extract text from response
            if not response.parts:
                logger.error("Gemini returned empty response")
                raise RuntimeError("Gemini returned empty response")

            generated_text = response.text
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise RuntimeError(f"Gemini API request failed: {e}") from e

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """
        Generate chat response from conversation history.

        Parameters
        ----------
        messages : list of dict
            Conversation history with format:
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        temperature : float, optional
            Override default temperature
        max_tokens : int, optional
            Override default max_tokens

        Returns
        -------
        str
            Generated assistant response

        Raises
        ------
        ValueError
            If messages format is invalid
        RuntimeError
            If API returns an error

        Examples
        --------
        >>> client = GeminiClient(api_key="your-api-key")
        >>> messages = [{"role": "user", "content": "What is energy?"}]
        >>> response = client.chat(messages)
        >>> print(response)
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        generation_config = genai.GenerationConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=max_tokens or self.max_tokens
        )

        try:
            # Convert messages to Gemini chat format
            chat = self.model.start_chat(history=[])

            # Process messages
            for msg in messages[:-1]:  # All but last message
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "user":
                    chat.send_message(content, generation_config=generation_config)

            # Send final message and get response
            last_message = messages[-1]
            if last_message.get("role") != "user":
                raise ValueError("Last message must be from user")

            logger.debug(f"Sending chat request: {len(messages)} messages")
            response = chat.send_message(
                last_message.get("content", ""),
                generation_config=generation_config
            )

            generated_text = response.text
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            raise RuntimeError(f"Gemini chat API request failed: {e}") from e

    def list_models(self) -> list[str]:
        """
        List available Gemini models.

        Returns
        -------
        list of str
            Available model names

        Examples
        --------
        >>> client = GeminiClient(api_key="your-api-key")
        >>> models = client.list_models()
        >>> for model in models:
        ...     print(model)
        """
        try:
            models = genai.list_models()
            return [m.name for m in models if "generateContent" in m.supported_generation_methods]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def is_model_available(self, model_name: str | None = None) -> bool:
        """
        Check if a specific model is available.

        Parameters
        ----------
        model_name : str, optional
            Model name to check, defaults to self.model_name

        Returns
        -------
        bool
            True if model is available, False otherwise

        Examples
        --------
        >>> client = GeminiClient(api_key="your-api-key")
        >>> if client.is_model_available():
        ...     print("Model is ready")
        """
        check_model = model_name or self.model_name
        models = self.list_models()
        return any(check_model in m for m in models)

    def health_check(self) -> bool:
        """
        Check if Gemini API is reachable and healthy.

        Returns
        -------
        bool
            True if API is healthy, False otherwise

        Examples
        --------
        >>> client = GeminiClient(api_key="your-api-key")
        >>> if client.health_check():
        ...     print("Gemini is accessible")
        """
        try:
            # Try a simple generation
            response = self.model.generate_content(
                "Test",
                generation_config=genai.GenerationConfig(max_output_tokens=10)
            )
            return bool(response.text)
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False
