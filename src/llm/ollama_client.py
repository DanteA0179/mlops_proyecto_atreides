"""
Ollama client for local LLM inference.

This module provides a Python client for interacting with Ollama API
to run Llama 3.2 3B model locally for natural language processing tasks.
"""

import logging
import os
from typing import Any

import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama API to perform local LLM inference.

    This client provides methods to generate text and handle chat conversations
    using locally hosted LLM models through Ollama.

    Parameters
    ----------
    host : str, optional
        Ollama API host URL, defaults to OLLAMA_HOST env var or localhost
    model : str, optional
        Model name to use, defaults to OLLAMA_MODEL env var or llama3.2:3b
    timeout : int, optional
        Request timeout in seconds, defaults to OLLAMA_TIMEOUT env var or 120
    temperature : float, optional
        Sampling temperature (0.0 to 1.0), defaults to 0.7
    max_tokens : int, optional
        Maximum tokens to generate, defaults to 1000

    Examples
    --------
    >>> client = OllamaClient()
    >>> response = client.generate("What is energy consumption?")
    >>> print(response)
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ):
        """Initialize Ollama client with configuration."""
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.temperature = temperature or float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))

        self.api_url = f"{self.host}/api/generate"
        self.chat_url = f"{self.host}/api/chat"
        self.tags_url = f"{self.host}/api/tags"

        logger.info(f"Initialized OllamaClient with model={self.model}, host={self.host}")

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False
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
        stream : bool, optional
            Stream response (not implemented)

        Returns
        -------
        str
            Generated text response

        Raises
        ------
        TimeoutError
            If request exceeds timeout
        RuntimeError
            If API returns an error

        Examples
        --------
        >>> client = OllamaClient()
        >>> response = client.generate("Explain energy consumption in one sentence.")
        >>> print(response)
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }

        try:
            logger.debug(f"Sending generate request: prompt_len={len(prompt)}")
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            generated_text = data.get("response", "")

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Timeout as e:
            logger.error(f"Request timeout after {self.timeout}s: {e}")
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s") from e

        except RequestException as e:
            logger.error(f"Request failed: {e}")
            raise RuntimeError(f"Ollama API request failed: {e}") from e

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
        TimeoutError
            If request exceeds timeout
        RuntimeError
            If API returns an error

        Examples
        --------
        >>> client = OllamaClient()
        >>> messages = [{"role": "user", "content": "What is energy?"}]
        >>> response = client.chat(messages)
        >>> print(response)
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }

        try:
            logger.debug(f"Sending chat request: {len(messages)} messages")
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            assistant_message = data.get("message", {})
            generated_text = assistant_message.get("content", "")

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except Timeout as e:
            logger.error(f"Chat request timeout after {self.timeout}s: {e}")
            raise TimeoutError(f"Ollama chat request timed out after {self.timeout}s") from e

        except RequestException as e:
            logger.error(f"Chat request failed: {e}")
            raise RuntimeError(f"Ollama chat API request failed: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        """
        List available models in Ollama.

        Returns
        -------
        list of dict
            Available models with metadata

        Examples
        --------
        >>> client = OllamaClient()
        >>> models = client.list_models()
        >>> for model in models:
        ...     print(model['name'])
        """
        try:
            response = requests.get(self.tags_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])

        except RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def is_model_available(self, model_name: str | None = None) -> bool:
        """
        Check if a specific model is available.

        Parameters
        ----------
        model_name : str, optional
            Model name to check, defaults to self.model

        Returns
        -------
        bool
            True if model is available, False otherwise

        Examples
        --------
        >>> client = OllamaClient()
        >>> if client.is_model_available():
        ...     print("Model is ready")
        """
        check_model = model_name or self.model
        models = self.list_models()
        return any(m.get("name") == check_model for m in models)

    def get_model_info(self, model_name: str | None = None) -> dict[str, Any] | None:
        """
        Get information about a specific model.

        Parameters
        ----------
        model_name : str, optional
            Model name, defaults to self.model

        Returns
        -------
        dict or None
            Model information if available

        Examples
        --------
        >>> client = OllamaClient()
        >>> info = client.get_model_info()
        >>> print(info['size'])
        """
        check_model = model_name or self.model
        models = self.list_models()

        for model in models:
            if model.get("name") == check_model:
                return model

        return None

    def health_check(self) -> bool:
        """
        Check if Ollama API is reachable and healthy.

        Returns
        -------
        bool
            True if API is healthy, False otherwise

        Examples
        --------
        >>> client = OllamaClient()
        >>> if client.health_check():
        ...     print("Ollama is running")
        """
        try:
            response = requests.get(self.tags_url, timeout=5)
            return response.status_code == 200
        except RequestException:
            return False
