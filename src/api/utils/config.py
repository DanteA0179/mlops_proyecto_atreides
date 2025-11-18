"""
Configuration utilities for Energy Optimization API.

This module provides logging setup and configuration utilities.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure structured logging for API.

    Sets up console and optionally file logging with structured format.

    Parameters
    ----------
    log_level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file (if None, logs to stdout only)

    Examples
    --------
    >>> setup_logging(log_level="DEBUG", log_file="logs/api.log")
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("API started")
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Structured format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


class Config:
    """
    Application configuration.

    Attributes
    ----------
    APP_NAME : str
        Application name
    APP_VERSION : str
        Application version
    API_PREFIX : str
        API route prefix
    LOG_LEVEL : str
        Logging level
    LOG_FILE : Optional[str]
        Log file path
    MODEL_PATH : str
        Default model path
    MODEL_TYPE : str
        Default model type
    MLFLOW_TRACKING_URI : str
        MLflow tracking server URI
    """

    APP_NAME: str = "Energy Optimization Copilot API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str | None = None

    # Model configuration
    MODEL_PATH: str = "models/ensembles/ensemble_lightgbm_v3.pkl"
    MODEL_TYPE: str = "stacking_ensemble"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # Performance
    MAX_BATCH_SIZE: int = 1000
    REQUEST_TIMEOUT: int = 30


# Global config instance
config = Config()
