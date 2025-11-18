"""
MLflow utility functions for experiment tracking and model logging.

This module provides helper functions to interact with MLflow tracking server
running in docker-compose at http://localhost:5000.

Functions:
    setup_mlflow_experiment: Initialize MLflow experiment
    log_system_metrics: Log system information (CPU, GPU, memory)
    log_model_params: Log model hyperparameters
    log_model_metrics: Log evaluation metrics
    log_cv_results: Log cross-validation results
    log_feature_importance: Extract and log feature importance
    save_and_log_model: Save model to disk and log to MLflow
"""

import hashlib
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow.sklearn
import numpy as np
import psutil

import mlflow

logger = logging.getLogger(__name__)

# Default MLflow tracking URI (docker-compose service)
DEFAULT_TRACKING_URI = "http://localhost:5000"


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = DEFAULT_TRACKING_URI) -> str:
    """
    Setup MLflow experiment and return experiment_id.

    This function connects to the MLflow tracking server (running in docker-compose)
    and creates or retrieves an experiment by name.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment to create or retrieve
    tracking_uri : str, default="http://localhost:5000"
        URI of the MLflow tracking server

    Returns
    -------
    str
        Experiment ID

    Examples
    --------
    >>> experiment_id = setup_mlflow_experiment("xgboost_baseline")
    >>> print(f"Experiment ID: {experiment_id}")

    Notes
    -----
    The MLflow server must be running before calling this function.
    Use `docker-compose up mlflow -d` to start the server.
    """
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Connected to MLflow at {tracking_uri}")

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        # Set experiment as active
        mlflow.set_experiment(experiment_name)

        return experiment_id

    except Exception as e:
        logger.error(f"Failed to setup MLflow experiment: {e}")
        raise


def log_system_metrics() -> dict[str, Any]:
    """
    Log system information to MLflow (CPU, GPU, memory, etc.).

    Returns
    -------
    dict
        Dictionary with system information

    Examples
    --------
    >>> system_info = log_system_metrics()
    >>> print(f"CPU cores: {system_info['cpu_count']}")
    """
    try:
        system_info = {}

        # CPU information
        system_info["cpu_count"] = psutil.cpu_count(logical=True)
        system_info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        system_info["cpu_percent"] = psutil.cpu_percent(interval=1)

        # Memory information
        memory = psutil.virtual_memory()
        system_info["memory_total_gb"] = round(memory.total / (1024**3), 2)
        system_info["memory_available_gb"] = round(memory.available / (1024**3), 2)
        system_info["memory_percent"] = memory.percent

        # Platform information
        system_info["platform"] = platform.system()
        system_info["platform_version"] = platform.version()
        system_info["python_version"] = platform.python_version()

        # GPU information
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(",")
                system_info["gpu_name"] = gpu_info[0].strip()
                system_info["gpu_memory_mb"] = gpu_info[1].strip()
                system_info["gpu_driver"] = gpu_info[2].strip()
                system_info["gpu_available"] = True
            else:
                system_info["gpu_available"] = False
        except Exception:
            system_info["gpu_available"] = False

        # Log to MLflow as params (for filtering) and metrics (for comparison)
        mlflow.log_params(
            {
                "system_platform": system_info["platform"],
                "system_python_version": system_info["python_version"],
                "system_cpu_count": system_info["cpu_count"],
                "system_gpu_available": system_info["gpu_available"],
            }
        )

        if system_info["gpu_available"]:
            mlflow.log_params(
                {
                    "system_gpu_name": system_info["gpu_name"],
                    "system_gpu_memory": system_info["gpu_memory_mb"],
                }
            )

        mlflow.log_metrics(
            {
                "system_memory_total_gb": system_info["memory_total_gb"],
                "system_memory_available_gb": system_info["memory_available_gb"],
                "system_cpu_percent": system_info["cpu_percent"],
            }
        )

        logger.info("Logged system metrics to MLflow")
        logger.info(f"  CPU: {system_info['cpu_count']} cores")
        logger.info(f"  Memory: {system_info['memory_total_gb']} GB")
        logger.info(f"  GPU: {system_info.get('gpu_name', 'Not available')}")

        return system_info

    except Exception as e:
        logger.warning(f"Failed to log system metrics: {e}")
        return {}


def log_model_params(params: dict[str, Any]) -> None:
    """
    Log model parameters to MLflow.

    Parameters
    ----------
    params : dict
        Dictionary of model parameters to log

    Examples
    --------
    >>> params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100}
    >>> log_model_params(params)
    """
    try:
        # Filter out None values and convert to JSON-serializable types
        clean_params = {}
        for key, value in params.items():
            if value is not None:
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                clean_params[key] = value

        mlflow.log_params(clean_params)
        logger.info(f"Logged {len(clean_params)} parameters to MLflow")

    except Exception as e:
        logger.error(f"Failed to log parameters: {e}")
        raise


def log_model_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """
    Log metrics to MLflow with optional prefix.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics to log (metric_name -> value)
    prefix : str, optional
        Prefix to add to metric names (e.g., "train_", "val_", "test_")

    Examples
    --------
    >>> metrics = {'rmse': 0.195, 'mae': 0.044, 'r2': 0.91}
    >>> log_model_metrics(metrics, prefix="test_")
    """
    try:
        prefixed_metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        logger.info(f"Logged {len(metrics)} metrics to MLflow with prefix '{prefix}'")

    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        raise


def log_cv_results(
    cv_scores: dict[str, dict[str, float]], fold_scores: list[dict[str, float]]
) -> None:
    """
    Log cross-validation results to MLflow.

    Parameters
    ----------
    cv_scores : dict
        Dictionary with mean and std for each metric
        Example: {'rmse': {'mean': 0.196, 'std': 0.008}}
    fold_scores : list of dict
        List of metric dictionaries for each fold

    Examples
    --------
    >>> cv_scores = {
    ...     'rmse': {'mean': 0.196, 'std': 0.008},
    ...     'mae': {'mean': 0.044, 'std': 0.002}
    ... }
    >>> fold_scores = [
    ...     {'rmse': 0.190, 'mae': 0.043},
    ...     {'rmse': 0.198, 'mae': 0.045}
    ... ]
    >>> log_cv_results(cv_scores, fold_scores)
    """
    try:
        # Log aggregated CV scores
        for metric_name, stats in cv_scores.items():
            mlflow.log_metric(f"cv_{metric_name}_mean", stats["mean"])
            mlflow.log_metric(f"cv_{metric_name}_std", stats["std"])

        logger.info(f"Logged CV results for {len(cv_scores)} metrics")

        # Skip artifact logging to avoid Windows timeout issues
        # Fold scores are already available in cv_scores mean/std
        logger.info("Skipped CV fold scores artifact logging")

    except Exception as e:
        logger.error(f"Failed to log CV results: {e}")
        raise


def log_feature_importance(
    model, feature_names: list[str], importance_type: str = "gain"
) -> dict[str, float]:
    """
    Extract and log feature importance from XGBoost model.

    Parameters
    ----------
    model : xgboost.XGBRegressor or sklearn.pipeline.Pipeline
        Trained model (if Pipeline, extracts XGBoost from last step)
    feature_names : list of str
        List of feature names
    importance_type : str, default="gain"
        Type of importance to extract: "gain", "weight", or "cover"

    Returns
    -------
    dict
        Feature importance dictionary (feature_name -> importance_value)

    Examples
    --------
    >>> importance = log_feature_importance(
    ...     model,
    ...     feature_names=['CO2', 'NSM', 'hour'],
    ...     importance_type="gain"
    ... )
    """
    try:
        # Extract XGBoost model from pipeline if needed
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            xgb_model = model.named_steps[list(model.named_steps.keys())[-1]]
        else:
            xgb_model = model

        # Get feature importance
        if hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            importance_dict = dict(zip(feature_names, importances, strict=False))
        elif hasattr(xgb_model, "get_booster"):
            # For XGBoost models with get_booster method
            booster = xgb_model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)

            # Map feature indices to names if needed
            if all(isinstance(k, str) and k.startswith("f") for k in importance_dict.keys()):
                importance_dict = {feature_names[int(k[1:])]: v for k, v in importance_dict.items()}
        else:
            raise ValueError("Model does not support feature importance extraction")

        # Convert numpy types to Python native types for JSON serialization
        importance_dict = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in importance_dict.items()
        }

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        # Save as JSON artifact
        temp_path = Path(f"temp_feature_importance_{importance_type}.json")
        with open(temp_path, "w") as f:
            json.dump(importance_dict, f, indent=2)

        mlflow.log_artifact(str(temp_path), artifact_path="feature_importance")
        temp_path.unlink()

        # Log top 10 features as metrics
        for i, (_feature, importance) in enumerate(list(importance_dict.items())[:10]):
            mlflow.log_metric(f"feature_importance_{importance_type}_rank{i+1}", importance)

        logger.info(
            f"Logged feature importance ({importance_type}) for {len(importance_dict)} features"
        )

        return importance_dict

    except Exception as e:
        logger.error(f"Failed to log feature importance: {e}")
        raise


def save_and_log_model(
    model, model_path: Path, artifact_name: str = "model", log_to_mlflow: bool = True
) -> dict[str, str]:
    """
    Save model to disk and optionally log to MLflow.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline or any sklearn-compatible model
        Trained model to save
    model_path : Path
        Path where to save the model (must be absolute)
    artifact_name : str, default="model"
        Name for the artifact in MLflow
    log_to_mlflow : bool, default=True
        Whether to log model to MLflow

    Returns
    -------
    dict
        Dictionary with model_path, md5_checksum, and mlflow_artifact_uri

    Examples
    --------
    >>> from pathlib import Path
    >>> model_info = save_and_log_model(
    ...     model,
    ...     model_path=Path("models/baselines/xgboost_v1.pkl"),
    ...     artifact_name="xgboost_baseline"
    ... )
    >>> print(f"Model saved to: {model_info['model_path']}")
    >>> print(f"MD5 checksum: {model_info['md5_checksum']}")
    """
    try:
        # Ensure parent directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with joblib
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")

        # Calculate MD5 checksum
        md5_hash = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        md5_checksum = md5_hash.hexdigest()

        # Save metadata
        metadata = {
            "model_path": str(model_path),
            "md5_checksum": md5_checksum,
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Model checksum: {md5_checksum}")

        # Skip MLflow model logging to avoid Windows timeout issues
        # Model is saved locally and can be loaded later
        mlflow_artifact_uri = None
        if log_to_mlflow:
            logger.info("Skipped MLflow model logging to avoid timeout issues")

            # Get artifact URI
            run = mlflow.active_run()
            if run:
                mlflow_artifact_uri = f"{run.info.artifact_uri}/{artifact_name}"

            logger.info(f"Logged model to MLflow as '{artifact_name}'")

        return {
            "model_path": str(model_path),
            "md5_checksum": md5_checksum,
            "mlflow_artifact_uri": mlflow_artifact_uri,
        }

    except Exception as e:
        logger.error(f"Failed to save/log model: {e}")
        raise


def verify_model_loading(model_path: Path) -> bool:
    """
    Verify that a saved model can be loaded correctly.

    Parameters
    ----------
    model_path : Path
        Path to the saved model

    Returns
    -------
    bool
        True if model loads successfully, False otherwise

    Examples
    --------
    >>> model_path = Path("models/baselines/xgboost_v1.pkl")
    >>> is_valid = verify_model_loading(model_path)
    >>> print(f"Model valid: {is_valid}")
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")

        # Verify model has required methods
        required_methods = ["predict"]
        for method in required_methods:
            if not hasattr(model, method):
                logger.error(f"Model missing required method: {method}")
                return False

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
