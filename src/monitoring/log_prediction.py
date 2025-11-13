"""
Prediction logging for model monitoring.

This module provides simple CSV logging functionality for capturing
predictions and input features from the production API.

Functions
---------
log_prediction : Append prediction to CSV file
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_prediction(features: dict[str, Any], prediction: float) -> None:
    """
    Append prediction to CSV file.

    Logs input features, prediction, and timestamp to a CSV file
    for monitoring and drift detection. Creates the file with headers
    if it doesn't exist.

    Parameters
    ----------
    features : dict[str, Any]
        Input features dictionary (18 features after engineering)
        Keys should match the feature names from preprocessing
    prediction : float
        Model prediction (energy usage value)

    Examples
    --------
    >>> features = {
    ...     'Ladle_Temperature': 1650.0,
    ...     'Tundish_Temperature': 1520.0,
    ...     'Casting_Speed': 1.5,
    ...     # ... other 15 features
    ... }
    >>> log_prediction(features, 42.5)

    Notes
    -----
    - CSV file location: data/monitoring/predictions.csv
    - Format: timestamp, feature1, feature2, ..., prediction
    - Failures are logged but do not raise exceptions
    """
    log_file = Path("data/monitoring/predictions.csv")

    try:
        # Create file with header if doesn't exist
        if not log_file.exists():
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Header: timestamp + feature names + prediction
                header = ["timestamp"] + list(features.keys()) + ["prediction"]
                writer.writerow(header)
            logger.info(f"Created prediction log file: {log_file}")

        # Append prediction
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Row: timestamp + feature values + prediction
            row = [datetime.now().isoformat()] + list(features.values()) + [prediction]
            writer.writerow(row)

        logger.debug(f"Logged prediction: {prediction:.2f}")

    except Exception as e:
        # Log error but don't fail the prediction request
        logger.warning(f"Failed to log prediction: {e}")
