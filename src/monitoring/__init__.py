"""
Monitoring module for drift detection and model performance tracking.

This module provides automated monitoring using Evidently AI to detect:
- Data drift in input features
- Target drift in ground truth labels
- Prediction drift in model outputs
- Model performance degradation

Components:
    - data_loader: Load reference and production data
    - drift_analyzer: Analyze drift using Evidently AI
    - report_generator: Generate HTML reports and metrics
    - alert_manager: Send email alerts on drift detection
    - metrics_tracker: Track historical metrics
    - config: Configuration management
    - evidently_monitor: Main CLI script
    - log_prediction: Log predictions for monitoring

Example:
    Basic usage from command line:

    ```bash
    python -m src.monitoring.evidently_monitor \
        --config config/monitoring_config.yaml \
        --production-data data/production/latest.parquet \
        --send-alerts
    ```

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

__version__ = "1.0.0"
__author__ = "Arthur - MLOps Team"

from src.monitoring.log_prediction import log_prediction

__all__ = [
    "log_prediction",
]
