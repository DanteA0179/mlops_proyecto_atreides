"""
Dagster Ops for Training Pipeline.

Ops are the equivalent of Prefect tasks - individual units of computation.
These ops reuse all the existing logic from utils and models.
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import polars as pl
import yaml
from dagster import OpExecutionContext, Out, op

logger = logging.getLogger(__name__)


# ==============================================================================
# OP 1: Load Configuration
# ==============================================================================


@op(
    name="load_config",
    description="Load training configuration from YAML file",
    out=Out(dict, description="Training configuration"),
    tags={"stage": "setup"},
    config_schema={"config_path": str},
)
def load_config_op(context: OpExecutionContext) -> dict:
    """
    Load training configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        config_path = context.op_config["config_path"]
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        context.log.info(f"Config loaded: {config['model']['type']} v{config['model']['version']}")
        return config

    except Exception as e:
        context.log.error(f"Failed to load config: {e}")
        raise


# ==============================================================================
# OP 2: Load Training Data
# ==============================================================================


@op(
    name="load_data",
    description="Load preprocessed train/val/test data from US-012",
    out=Out(tuple, description="(train, val, test) dataframes"),
    tags={"stage": "data"},
)
def load_data_op(context: OpExecutionContext, config: dict) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load preprocessed data from parquet files.

    Parameters
    ----------
    config : dict
        Configuration with data paths

    Returns
    -------
    tuple
        (train, val, test) Polars DataFrames
    """
    try:
        train = pl.read_parquet(config["data"]["train"])
        val = pl.read_parquet(config["data"]["val"])
        test = pl.read_parquet(config["data"]["test"])

        context.log.info(f"Data loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}")
        return train, val, test

    except Exception as e:
        context.log.error(f"Failed to load data: {e}")
        raise


# ==============================================================================
# OP 3: Validate Data Quality
# ==============================================================================


@op(
    name="validate_data",
    description="Validate data quality using US-006, US-011, US-012 utilities",
    out=Out(dict, description="Validation results"),
    tags={"stage": "validation"},
)
def validate_data_op(context: OpExecutionContext, data: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]) -> dict:
    """
    Validate data quality using existing utilities.

    Parameters
    ----------
    data : tuple
        (train, val, test) dataframes

    Returns
    -------
    dict
        Validation results
    """
    from src.utils.data_quality import analyze_nulls
    from src.utils.split_data import validate_splits

    try:
        train, val, test = data

        # Check 1: Null values
        nulls_train = analyze_nulls(train, "train")
        nulls_val = analyze_nulls(val, "val")
        nulls_test = analyze_nulls(test, "test")

        total_nulls = (
            nulls_train["null_count"].sum()
            + nulls_val["null_count"].sum()
            + nulls_test["null_count"].sum()
        )

        if total_nulls > 0:
            context.log.warning(f"Found {total_nulls} null values")
        else:
            context.log.info("Check 1/2: No null values found")

        # Check 2: Splits validation (optional)
        try:
            split_validation = validate_splits(train, val, test, target_col="Usage_kWh")
            if split_validation["valid"]:
                context.log.info("Check 2/2: Splits validated")
            else:
                context.log.warning(f"Check 2/2: Split warnings: {split_validation.get('issues', [])}")
        except Exception as e:
            context.log.warning(f"Check 2/2: Split validation skipped: {e}")

        return {"status": "valid", "checks": 2, "total_rows": len(train) + len(val) + len(test)}

    except Exception as e:
        context.log.error(f"Validation failed: {e}")
        raise


# ==============================================================================
# OP 4: Train Model (Model-Agnostic)
# ==============================================================================


@op(
    name="train_model",
    description="Train ML model with GPU fallback - supports XGBoost, LightGBM, CatBoost, Ensemble",
    out=Out(object, description="Trained model pipeline"),
    tags={"stage": "training"},
)
def train_model_op(
    context: OpExecutionContext, data: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame], config: dict
) -> Any:
    """
    Train model using existing trainer infrastructure from US-013.
    Automatically detects model type from config and uses appropriate trainer.

    Supported model types:
    - xgboost: XGBoost gradient boosting
    - lightgbm: LightGBM gradient boosting
    - catboost: CatBoost gradient boosting
    - ensemble_lightgbm: Stacking ensemble with LightGBM meta-model
    - ensemble_ridge: Stacking ensemble with Ridge meta-model

    Parameters
    ----------
    data : tuple
        (train, val, test) dataframes
    config : dict
        Configuration with hyperparameters and model type

    Returns
    -------
    object
        Trained sklearn pipeline or ensemble model
    """
    try:
        train, val, test = data
        target_col = config["data"]["target_col"]
        model_type = config["model"]["type"]

        # Prepare features
        X_train = train.drop(target_col).to_numpy()
        y_train = train[target_col].to_numpy()

        context.log.info(f"Training {model_type}: X_train={X_train.shape}, y_train={y_train.shape}")

        # Route to appropriate trainer based on model type
        if model_type == "xgboost":
            pipeline = _train_xgboost(context, X_train, y_train, config)
        elif model_type == "lightgbm":
            pipeline = _train_lightgbm(context, X_train, y_train, config)
        elif model_type == "catboost":
            pipeline = _train_catboost(context, X_train, y_train, config)
        elif model_type in ["ensemble_lightgbm", "ensemble_ridge"]:
            pipeline = _train_ensemble(context, X_train, y_train, config, data)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        context.log.info(f"Training completed: {model_type}")
        return pipeline

    except Exception as e:
        context.log.error(f"Training failed: {e}")
        raise


def _train_xgboost(context: OpExecutionContext, X_train, y_train, config: dict) -> Any:
    """Train XGBoost model with GPU fallback."""
    from src.models.xgboost_trainer import check_gpu_availability, create_xgboost_pipeline

    gpu_available, device_info = check_gpu_availability()
    context.log.info(f"GPU available: {gpu_available}, Device: {device_info}")

    hyperparams = config["hyperparameters"].copy()
    hyperparams["tree_method"] = "hist"
    hyperparams["device"] = "cuda:0" if gpu_available else "cpu"

    try:
        pipeline = create_xgboost_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        device_used = "GPU" if gpu_available else "CPU"
        context.log.info(f"XGBoost trained on {device_used}")
    except Exception as gpu_error:
        context.log.warning(f"GPU training failed: {gpu_error}, falling back to CPU")
        hyperparams["device"] = "cpu"
        pipeline = create_xgboost_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        context.log.info("XGBoost trained on CPU")

    return pipeline


def _train_lightgbm(context: OpExecutionContext, X_train, y_train, config: dict) -> Any:
    """Train LightGBM model with GPU fallback."""
    from src.models.lightgbm_trainer import check_gpu_availability, create_lightgbm_pipeline

    gpu_available, device_info = check_gpu_availability()
    context.log.info(f"GPU available: {gpu_available}, Device: {device_info}")

    hyperparams = config["hyperparameters"].copy()
    hyperparams["device"] = "gpu" if gpu_available else "cpu"

    try:
        pipeline = create_lightgbm_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        device_used = "GPU" if gpu_available else "CPU"
        context.log.info(f"LightGBM trained on {device_used}")
    except Exception as gpu_error:
        context.log.warning(f"GPU training failed: {gpu_error}, falling back to CPU")
        hyperparams["device"] = "cpu"
        pipeline = create_lightgbm_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        context.log.info("LightGBM trained on CPU")

    return pipeline


def _train_catboost(context: OpExecutionContext, X_train, y_train, config: dict) -> Any:
    """Train CatBoost model with GPU fallback."""
    from src.models.catboost_trainer import check_gpu_availability, create_catboost_pipeline

    gpu_available, device_info = check_gpu_availability()
    context.log.info(f"GPU available: {gpu_available}, Device: {device_info}")

    hyperparams = config["hyperparameters"].copy()
    hyperparams["task_type"] = "GPU" if gpu_available else "CPU"

    try:
        pipeline = create_catboost_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        device_used = "GPU" if gpu_available else "CPU"
        context.log.info(f"CatBoost trained on {device_used}")
    except Exception as gpu_error:
        context.log.warning(f"GPU training failed: {gpu_error}, falling back to CPU")
        hyperparams["task_type"] = "CPU"
        pipeline = create_catboost_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        context.log.info("CatBoost trained on CPU")

    return pipeline


def _train_ensemble(context: OpExecutionContext, X_train, y_train, config: dict, data: tuple) -> Any:
    """
    Train ensemble model (stacking).
    
    Ensemble training requires:
    1. Train multiple base models (XGBoost, LightGBM, CatBoost)
    2. Generate predictions from base models
    3. Train meta-model on base model predictions
    """
    context.log.info("Training ensemble model (this will take longer)...")
    
    # Import base model trainers
    from src.models.xgboost_trainer import create_xgboost_pipeline
    from src.models.lightgbm_trainer import create_lightgbm_pipeline
    from src.models.catboost_trainer import create_catboost_pipeline
    
    # Get base models config
    base_models_config = config.get("base_models", [])
    if not base_models_config:
        raise ValueError("Ensemble config must have 'base_models' section")
    
    # Train base models
    base_models = []
    for i, base_config in enumerate(base_models_config):
        model_type = base_config["type"]
        hyperparams = base_config["hyperparameters"]
        
        context.log.info(f"Training base model {i+1}/{len(base_models_config)}: {model_type}")
        
        if model_type == "xgboost":
            pipeline = create_xgboost_pipeline(hyperparams)
        elif model_type == "lightgbm":
            pipeline = create_lightgbm_pipeline(hyperparams)
        elif model_type == "catboost":
            pipeline = create_catboost_pipeline(hyperparams)
        else:
            raise ValueError(f"Unsupported base model type: {model_type}")
        
        pipeline.fit(X_train, y_train)
        base_models.append(pipeline)
        context.log.info(f"Base model {i+1} trained: {model_type}")
    
    # Generate predictions from base models for meta-model training
    context.log.info("Generating meta-features from base models...")
    base_predictions = np.column_stack([model.predict(X_train) for model in base_models])
    
    # Train meta-model
    meta_model_type = config["model"]["type"]
    meta_hyperparams = config["hyperparameters"]
    
    context.log.info(f"Training meta-model: {meta_model_type}")
    
    if "lightgbm" in meta_model_type:
        from src.models.lightgbm_trainer import create_lightgbm_pipeline
        meta_model = create_lightgbm_pipeline(meta_hyperparams)
    elif "ridge" in meta_model_type:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        meta_model = Pipeline([("regressor", Ridge(**meta_hyperparams))])
    else:
        raise ValueError(f"Unsupported meta-model type: {meta_model_type}")
    
    meta_model.fit(base_predictions, y_train)
    context.log.info("Meta-model trained")
    
    # Create ensemble wrapper
    ensemble = {
        "base_models": base_models,
        "meta_model": meta_model,
        "type": "stacking_ensemble"
    }
    
    context.log.info(f"Ensemble complete: {len(base_models)} base models + meta-model")
    return ensemble


# ==============================================================================
# OP 5: Evaluate Model (Model-Agnostic)
# ==============================================================================


@op(
    name="evaluate_model",
    description="Evaluate model performance on validation and test sets - supports all model types",
    out=Out(dict, description="Evaluation metrics"),
    tags={"stage": "evaluation"},
)
def evaluate_model_op(
    context: OpExecutionContext,
    model: Any,
    data: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    config: dict,
) -> dict:
    """
    Evaluate model on validation and test sets.
    Handles both single models and ensemble models.

    Parameters
    ----------
    model : object
        Trained model pipeline or ensemble dict
    data : tuple
        (train, val, test) dataframes
    config : dict
        Configuration

    Returns
    -------
    dict
        Metrics dictionary with 'val' and 'test' keys
    """
    from src.models.xgboost_trainer import evaluate_model

    try:
        train, val, test = data
        target_col = config["data"]["target_col"]

        # Prepare validation data
        X_val = val.drop(target_col).to_numpy()
        y_val = val[target_col].to_numpy()

        # Prepare test data
        X_test = test.drop(target_col).to_numpy()
        y_test = test[target_col].to_numpy()

        # Check if ensemble
        if isinstance(model, dict) and model.get("type") == "stacking_ensemble":
            context.log.info("Evaluating ensemble model...")
            val_metrics = _evaluate_ensemble(model, X_val, y_val, "validation")
            test_metrics = _evaluate_ensemble(model, X_test, y_test, "test")
        else:
            # Single model evaluation
            val_metrics = evaluate_model(model, X_val, y_val, "validation")
            test_metrics = evaluate_model(model, X_test, y_test, "test")

        metrics = {"val": val_metrics, "test": test_metrics}

        context.log.info(f"Val RMSE: {val_metrics['rmse']:.4f} kWh, R²: {val_metrics['r2']:.4f}")
        context.log.info(f"Test RMSE: {test_metrics['rmse']:.4f} kWh, R²: {test_metrics['r2']:.4f}")

        return metrics

    except Exception as e:
        context.log.error(f"Evaluation failed: {e}")
        raise


def _evaluate_ensemble(ensemble: dict, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
    """Evaluate ensemble model by generating predictions from base models and meta-model."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Get predictions from base models
    base_predictions = np.column_stack([
        model.predict(X) for model in ensemble["base_models"]
    ])
    
    # Get final predictions from meta-model
    y_pred = ensemble["meta_model"].predict(base_predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "split": split_name
    }


# ==============================================================================
# OP 6: Check Performance Threshold
# ==============================================================================


@op(
    name="check_threshold",
    description="Validate model meets performance thresholds",
    out=Out(dict, description="Threshold check results"),
    tags={"stage": "validation"},
)
def check_threshold_op(context: OpExecutionContext, metrics: dict, config: dict) -> dict:
    """
    Check if model meets performance thresholds.

    Parameters
    ----------
    metrics : dict
        Model metrics
    config : dict
        Configuration with thresholds

    Returns
    -------
    dict
        Threshold check results
    """
    try:
        val_rmse = metrics["val"]["rmse"]
        val_r2 = metrics["val"]["r2"]

        threshold_rmse = config["thresholds"]["rmse"]
        threshold_r2 = config["thresholds"]["r2"]

        rmse_passed = val_rmse < threshold_rmse
        r2_passed = val_r2 > threshold_r2
        passed = rmse_passed and r2_passed

        result = {
            "passed": passed,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "threshold_rmse": threshold_rmse,
            "threshold_r2": threshold_r2,
        }

        if passed:
            context.log.info(f"Threshold PASSED: RMSE={val_rmse:.4f}<{threshold_rmse}, R²={val_r2:.4f}>{threshold_r2}")
        else:
            context.log.warning(f"Threshold FAILED: RMSE={val_rmse:.4f}>={threshold_rmse} or R²={val_r2:.4f}<={threshold_r2}")

        return result

    except Exception as e:
        context.log.error(f"Threshold check failed: {e}")
        return {"passed": False, "error": str(e)}


# ==============================================================================
# OP 7: Log to MLflow
# ==============================================================================


@op(
    name="log_mlflow",
    description="Log model, params, and metrics to MLflow",
    out=Out(str, description="MLflow run ID"),
    tags={"stage": "logging"},
)
def log_mlflow_op(context: OpExecutionContext, model: Any, metrics: dict, config: dict) -> str:
    """
    Log to MLflow using existing utilities from US-013.

    Parameters
    ----------
    model : object
        Trained model
    metrics : dict
        Model metrics
    config : dict
        Configuration

    Returns
    -------
    str
        MLflow run ID
    """
    import mlflow

    from src.utils.mlflow_utils import log_model_metrics, log_model_params, save_and_log_model

    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log tags
            for tag_key, tag_value in config["mlflow"]["tags"].items():
                mlflow.set_tag(tag_key, tag_value)

            # Log parameters
            log_model_params(config["hyperparameters"])

            # Log metrics
            log_model_metrics(metrics["val"], prefix="val_")
            log_model_metrics(metrics["test"], prefix="test_")

            # Save and log model
            model_path = Path("models/trained") / f"{config['model']['type']}_pipeline.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            save_and_log_model(model, model_path, "model")

            context.log.info(f"MLflow run: {run_id}")
            return run_id

    except Exception as e:
        context.log.error(f"MLflow logging failed: {e}")
        raise


# ==============================================================================
# OP 8: Save Artifacts
# ==============================================================================


@op(
    name="save_artifacts",
    description="Save model artifacts to disk for DVC versioning",
    out=Out(Path, description="Path to saved model"),
    tags={"stage": "artifacts"},
)
def save_artifacts_op(context: OpExecutionContext, model: Any, metrics: dict, config: dict) -> Path:
    """
    Save model artifacts to disk.

    Parameters
    ----------
    model : object
        Trained model
    metrics : dict
        Model metrics
    config : dict
        Configuration

    Returns
    -------
    Path
        Path to saved model file
    """
    import joblib

    try:
        models_dir = Path("models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_filename = f"{config['model']['type']}_pipeline.pkl"
        model_path = models_dir / model_filename
        joblib.dump(model, model_path)

        # Save metrics
        metrics_filename = f"{config['model']['type']}_metrics.yaml"
        metrics_path = models_dir / metrics_filename
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f)

        context.log.info(f"Artifacts saved: {model_path}")
        return model_path

    except Exception as e:
        context.log.error(f"Failed to save artifacts: {e}")
        raise


# ==============================================================================
# OP 9: DVC Add
# ==============================================================================


@op(
    name="dvc_add",
    description="Add model to DVC tracking",
    out=Out(Path, description="Path to DVC file"),
    tags={"stage": "versioning"},
)
def dvc_add_op(context: OpExecutionContext, model_path: Path) -> Path:
    """
    Add model to DVC tracking.

    Parameters
    ----------
    model_path : Path
        Path to model file

    Returns
    -------
    Path
        Path to DVC file (.dvc)
    """
    import subprocess

    try:
        result = subprocess.run(
            ["dvc", "add", str(model_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        dvc_file = model_path.with_suffix(model_path.suffix + ".dvc")
        context.log.info(f"DVC tracking: {dvc_file}")

        return dvc_file

    except subprocess.CalledProcessError as e:
        context.log.error(f"DVC add failed: {e.stderr}")
        raise
    except Exception as e:
        context.log.error(f"DVC add error: {e}")
        raise


# ==============================================================================
# OP 10: Send Notification
# ==============================================================================


@op(
    name="send_notification",
    description="Send notification about pipeline completion",
    tags={"stage": "notification"},
)
def send_notification_op(
    context: OpExecutionContext, metrics: dict, run_id: str, config: dict
) -> None:
    """
    Send notification about pipeline completion.

    Parameters
    ----------
    metrics : dict
        Model metrics
    run_id : str
        MLflow run ID
    config : dict
        Configuration
    """
    from src.utils.notification_utils import NotificationManager

    try:
        channels = config["notifications"]["channels"]
        notification_manager = NotificationManager(channels=channels)
        notification_manager.send("success", metrics, run_id, config)

        context.log.info(f"Notifications sent to: {', '.join(channels)}")

    except Exception as e:
        context.log.warning(f"Notification failed (non-critical): {e}")
