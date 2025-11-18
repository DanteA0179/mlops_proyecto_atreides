"""
Utility functions for LLM operations.

This module provides helper functions for working with LLM clients,
including prompt formatting, response parsing, and model validation.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_llm_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load LLM configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file, defaults to config/llm_config.yaml

    Returns
    -------
    dict
        Configuration dictionary

    Examples
    --------
    >>> config = load_llm_config()
    >>> print(config['prompts']['energy_query'])
    """
    if config_path is None:
        config_path = Path("config") / "llm_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def format_prompt(template: str, **kwargs) -> str:
    """
    Format prompt template with provided variables.

    Parameters
    ----------
    template : str
        Prompt template with {variable} placeholders
    **kwargs
        Variables to substitute in template

    Returns
    -------
    str
        Formatted prompt

    Examples
    --------
    >>> template = "Analyze {metric} for {time_period}"
    >>> prompt = format_prompt(template, metric="energy consumption", time_period="last week")
    >>> print(prompt)
    Analyze energy consumption for last week
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing variable in template: {e}")
        raise ValueError(f"Template requires variable: {e}") from e


def parse_energy_query(query: str) -> dict[str, Any]:
    """
    Parse natural language energy query into structured format.

    Parameters
    ----------
    query : str
        Natural language query

    Returns
    -------
    dict
        Structured query with intent and entities

    Examples
    --------
    >>> result = parse_energy_query("What was the energy consumption yesterday at 10am?")
    >>> print(result['intent'])
    historical_query
    """
    query_lower = query.lower()

    # Detect intent
    intent = "unknown"
    if any(word in query_lower for word in ["predict", "forecast", "will be", "tomorrow"]):
        intent = "prediction"
    elif any(word in query_lower for word in ["was", "yesterday", "last week", "historical"]):
        intent = "historical_query"
    elif any(word in query_lower for word in ["why", "explain", "reason"]):
        intent = "explanation"
    elif any(word in query_lower for word in ["optimize", "reduce", "improve"]):
        intent = "optimization"
    elif any(word in query_lower for word in ["compare", "difference", "versus"]):
        intent = "comparison"

    # Extract entities
    entities = {}

    # Time entities
    time_keywords = {
        "morning": "06:00-12:00",
        "afternoon": "12:00-18:00",
        "evening": "18:00-22:00",
        "night": "22:00-06:00",
        "today": "today",
        "yesterday": "yesterday",
        "tomorrow": "tomorrow",
        "week": "week",
        "month": "month"
    }

    for keyword, value in time_keywords.items():
        if keyword in query_lower:
            entities["time_period"] = value

    # Load type entities
    load_keywords = {"light": "Light", "medium": "Medium", "maximum": "Maximum", "max": "Maximum"}
    for keyword, value in load_keywords.items():
        if keyword in query_lower:
            entities["load_type"] = value

    # Metric entities
    metric_keywords = {
        "consumption": "Usage_kWh",
        "energy": "Usage_kWh",
        "co2": "CO2(tCO2)",
        "emissions": "CO2(tCO2)",
        "power factor": "power_factor",
        "reactive power": "reactive_power"
    }

    for keyword, value in metric_keywords.items():
        if keyword in query_lower:
            entities["metric"] = value
            break

    return {
        "original_query": query,
        "intent": intent,
        "entities": entities
    }


def validate_model_response(response: str) -> bool:
    """
    Validate that LLM response is not empty and meets quality criteria.

    Parameters
    ----------
    response : str
        LLM generated response

    Returns
    -------
    bool
        True if response is valid

    Examples
    --------
    >>> validate_model_response("Energy consumption refers to...")
    True
    >>> validate_model_response("")
    False
    """
    if not response or not response.strip():
        return False

    # Check minimum length
    if len(response.strip()) < 10:
        return False

    # Check for common error messages
    error_patterns = [
        "i don't know",
        "i cannot",
        "error",
        "failed",
        "unable to"
    ]

    response_lower = response.lower()
    if any(pattern in response_lower for pattern in error_patterns):
        logger.warning(f"Response contains error pattern: {response[:100]}")
        return False

    return True


def truncate_prompt(prompt: str, max_length: int = 2000) -> str:
    """
    Truncate prompt to maximum length while preserving meaning.

    Parameters
    ----------
    prompt : str
        Input prompt
    max_length : int
        Maximum character length

    Returns
    -------
    str
        Truncated prompt

    Examples
    --------
    >>> long_prompt = "..." * 1000
    >>> short = truncate_prompt(long_prompt, 100)
    >>> len(short) <= 100
    True
    """
    if len(prompt) <= max_length:
        return prompt

    logger.warning(f"Truncating prompt from {len(prompt)} to {max_length} chars")

    # Try to truncate at sentence boundary
    truncated = prompt[:max_length]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclamation = truncated.rfind('!')

    boundary = max(last_period, last_question, last_exclamation)

    if boundary > max_length // 2:
        return truncated[:boundary + 1]

    return truncated + "..."


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """
    Extract JSON object from LLM response text.

    Parameters
    ----------
    response : str
        LLM response that may contain JSON

    Returns
    -------
    dict or None
        Extracted JSON object if found

    Examples
    --------
    >>> response = "Here is the data: {'key': 'value'}"
    >>> extract_json_from_response(response)
    None
    """
    import json
    import re

    # Try to find JSON in response
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None
