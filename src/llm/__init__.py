"""
LLM module for Energy Optimization Copilot.

This module provides clients for interacting with different LLM providers:
- OllamaClient: Local inference with Llama 3.2 3B
- GeminiClient: Google Gemini 2.0 Flash API
- LLMFactory: Factory pattern for creating clients based on configuration

Usage:
    from src.llm import get_llm_client

    # Get client based on environment configuration
    client = get_llm_client()
    response = client.generate("What is energy consumption?")
"""

from src.llm.gemini_client import GeminiClient
from src.llm.llm_factory import get_llm_client
from src.llm.ollama_client import OllamaClient

__all__ = [
    "get_llm_client",
    "OllamaClient",
    "GeminiClient",
]
