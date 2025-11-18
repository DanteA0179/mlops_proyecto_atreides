"""
Factory pattern for creating LLM clients.

This module provides a factory function to create the appropriate LLM client
based on configuration, allowing easy switching between Ollama (local) and
Gemini (cloud) providers.
"""

import logging
import os

from src.llm.gemini_client import GeminiClient
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def get_llm_client(
    provider: str | None = None,
    **kwargs
) -> OllamaClient | GeminiClient:
    """
    Factory function to create LLM client based on provider.

    This function reads the LLM_PROVIDER environment variable (or uses the
    provided parameter) to determine which LLM client to instantiate.

    Parameters
    ----------
    provider : str, optional
        LLM provider name ('ollama' or 'gemini').
        If not provided, reads from LLM_PROVIDER env var.
        Defaults to 'ollama' if env var not set.
    **kwargs
        Additional parameters to pass to the client constructor

    Returns
    -------
    OllamaClient or GeminiClient
        Initialized LLM client

    Raises
    ------
    ValueError
        If provider is not supported

    Examples
    --------
    >>> # Using default provider from environment
    >>> client = get_llm_client()
    >>> response = client.generate("What is energy?")

    >>> # Explicitly specify provider
    >>> ollama_client = get_llm_client(provider="ollama")
    >>> gemini_client = get_llm_client(provider="gemini", api_key="key")
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")

    provider = provider.lower()

    logger.info(f"Creating LLM client with provider={provider}")

    if provider == "ollama":
        return OllamaClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: ollama, gemini"
        )


def get_available_providers() -> dict:
    """
    Get information about available LLM providers.

    Returns
    -------
    dict
        Dictionary with provider names as keys and availability status as values

    Examples
    --------
    >>> providers = get_available_providers()
    >>> print(providers)
    {'ollama': {'available': True, 'configured': True},
     'gemini': {'available': True, 'configured': False}}
    """
    providers = {}

    # Check Ollama
    try:
        ollama_client = OllamaClient()
        ollama_healthy = ollama_client.health_check()
        providers["ollama"] = {
            "available": True,
            "configured": True,
            "healthy": ollama_healthy,
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        }
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        providers["ollama"] = {
            "available": False,
            "configured": bool(os.getenv("OLLAMA_HOST")),
            "healthy": False,
            "error": str(e)
        }

    # Check Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key != "your-gemini-api-key-here":
        try:
            gemini_client = GeminiClient()
            gemini_healthy = gemini_client.health_check()
            providers["gemini"] = {
                "available": True,
                "configured": True,
                "healthy": gemini_healthy,
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            }
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")
            providers["gemini"] = {
                "available": False,
                "configured": True,
                "healthy": False,
                "error": str(e)
            }
    else:
        providers["gemini"] = {
            "available": True,
            "configured": False,
            "healthy": False,
            "message": "API key not configured"
        }

    return providers
