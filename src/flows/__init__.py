"""
Prefect flows for ML pipeline orchestration.

This module contains Prefect flows and tasks for automating the ML training pipeline.
"""

from .example_flow import example_energy_flow
from .training_pipeline import training_flow
from .training_tasks import (
    check_threshold_task,
    dvc_add_task,
    dvc_push_task,
    evaluate_model_task,
    load_config_task,
    load_data_task,
    log_mlflow_task,
    notify_task,
    prepare_features_task,
    save_artifacts_task,
    train_model_task,
    validate_data_task,
)

__all__ = [
    "example_energy_flow",
    "training_flow",
    "load_config_task",
    "load_data_task",
    "validate_data_task",
    "prepare_features_task",
    "train_model_task",
    "evaluate_model_task",
    "check_threshold_task",
    "log_mlflow_task",
    "save_artifacts_task",
    "dvc_add_task",
    "dvc_push_task",
    "notify_task",
]
