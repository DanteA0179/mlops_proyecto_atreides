"""
Working Dagster Pipeline with Separate Ops.

This version has separate ops for each step so you can see them individually in the UI.
"""

from pathlib import Path
from typing import Any, Tuple

import polars as pl
import yaml
from dagster import Config, OpExecutionContext, job, op


class PipelineConfig(Config):
    """Configuration for the pipeline."""

    config_path: str = "config/training/xgboost_config.yaml"


# ==============================================================================
# OP 1: Load Configuration
# ==============================================================================


@op(description="Load training configuration from YAML file", tags={"stage": "setup"})
def load_config(context: OpExecutionContext, config: PipelineConfig) -> dict:
    """Load training configuration."""
    config_file = Path(config.config_path)
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    context.log.info(f"Config loaded: {cfg['model']['type']} v{cfg['model']['version']}")
    return cfg


# ==============================================================================
# OP 2: Load Data
# ==============================================================================


@op(description="Load preprocessed train/val/test data", tags={"stage": "data"})
def load_data(context: OpExecutionContext, cfg: dict) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load preprocessed data from parquet files."""
    train = pl.read_parquet(cfg["data"]["train"])
    val = pl.read_parquet(cfg["data"]["val"])
    test = pl.read_parquet(cfg["data"]["test"])

    context.log.info(f"Data loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


# ==============================================================================
# OP 3: Validate Data
# ==============================================================================


@op(description="Validate data quality", tags={"stage": "validation"})
def validate_data(
    context: OpExecutionContext, data: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
) -> dict:
    """Validate data quality."""
    from src.utils.data_quality import analyze_nulls

    train, val, test = data

    # Check nulls
    nulls_train = analyze_nulls(train, "train")
    total_nulls = nulls_train["null_count"].sum()

    if total_nulls > 0:
        context.log.warning(f"Found {total_nulls} null values")
    else:
        context.log.info("No null values found")

    return {"status": "valid", "total_rows": len(train) + len(val) + len(test)}


# ==============================================================================
# OP 4: Train Model
# ==============================================================================


@op(description="Train ML model with GPU fallback", tags={"stage": "training"})
def train_model(
    context: OpExecutionContext,
    cfg: dict,
    data: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> Any:
    """Train model using existing trainer infrastructure."""
    from src.models.xgboost_trainer import check_gpu_availability, create_xgboost_pipeline

    train, val, test = data
    target_col = cfg["data"]["target_col"]

    # Prepare features
    X_train = train.drop(target_col).to_numpy()
    y_train = train[target_col].to_numpy()

    context.log.info(f"Features prepared: X_train={X_train.shape}")

    # Check GPU
    gpu_available, device_info = check_gpu_availability()
    context.log.info(f"GPU available: {gpu_available}, Device: {device_info}")

    # Train
    hyperparams = cfg["hyperparameters"].copy()
    hyperparams["tree_method"] = "hist"
    hyperparams["device"] = "cuda:0" if gpu_available else "cpu"

    try:
        pipeline = create_xgboost_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        device_used = "GPU" if gpu_available else "CPU"
        context.log.info(f"Training completed: XGBoost ({device_used})")
    except Exception as e:
        context.log.warning(f"GPU training failed: {e}, falling back to CPU")
        hyperparams["device"] = "cpu"
        pipeline = create_xgboost_pipeline(hyperparams)
        pipeline.fit(X_train, y_train)
        context.log.info("Training completed: XGBoost (CPU)")

    return pipeline


# ==============================================================================
# OP 5: Evaluate Model
# ==============================================================================


@op(description="Evaluate model performance", tags={"stage": "evaluation"})
def evaluate_model(
    context: OpExecutionContext,
    model: Any,
    cfg: dict,
    data: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> dict:
    """Evaluate model on validation and test sets."""
    from src.models.xgboost_trainer import evaluate_model as eval_fn

    train, val, test = data
    target_col = cfg["data"]["target_col"]

    X_val = val.drop(target_col).to_numpy()
    y_val = val[target_col].to_numpy()
    X_test = test.drop(target_col).to_numpy()
    y_test = test[target_col].to_numpy()

    val_metrics = eval_fn(model, X_val, y_val, "validation")
    test_metrics = eval_fn(model, X_test, y_test, "test")

    metrics = {"val": val_metrics, "test": test_metrics}

    context.log.info(f"Val RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
    context.log.info(f"Test RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")

    return metrics


# ==============================================================================
# OP 6: Check Threshold
# ==============================================================================


@op(description="Validate model meets performance thresholds", tags={"stage": "validation"})
def check_threshold(context: OpExecutionContext, metrics: dict, cfg: dict) -> dict:
    """Check if model meets performance thresholds."""
    val_rmse = metrics["val"]["rmse"]
    val_r2 = metrics["val"]["r2"]

    threshold_rmse = cfg["thresholds"]["rmse"]
    threshold_r2 = cfg["thresholds"]["r2"]

    passed = (val_rmse < threshold_rmse) and (val_r2 > threshold_r2)

    if passed:
        context.log.info(f"Threshold PASSED: RMSE={val_rmse:.4f}<{threshold_rmse}, R²={val_r2:.4f}>{threshold_r2}")
    else:
        context.log.warning(f"Threshold FAILED: RMSE={val_rmse:.4f}>={threshold_rmse} or R²={val_r2:.4f}<={threshold_r2}")

    return {"passed": passed, "val_rmse": val_rmse, "val_r2": val_r2}


# ==============================================================================
# OP 7: Log to MLflow
# ==============================================================================


@op(description="Log model and metrics to MLflow", tags={"stage": "logging"})
def log_mlflow(context: OpExecutionContext, model: Any, metrics: dict, cfg: dict) -> str:
    """Log to MLflow using existing utilities."""
    import mlflow

    from src.utils.mlflow_utils import log_model_metrics, log_model_params, save_and_log_model

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log tags
        for tag_key, tag_value in cfg["mlflow"]["tags"].items():
            mlflow.set_tag(tag_key, tag_value)

        # Log parameters
        log_model_params(cfg["hyperparameters"])

        # Log metrics
        log_model_metrics(metrics["val"], prefix="val_")
        log_model_metrics(metrics["test"], prefix="test_")

        # Save model
        model_path = Path("models/trained") / f"{cfg['model']['type']}_pipeline.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_and_log_model(model, model_path, "model")

        context.log.info(f"MLflow run: {run_id}")
        return run_id


# ==============================================================================
# OP 8: Save Artifacts
# ==============================================================================


@op(description="Save model artifacts to disk", tags={"stage": "artifacts"})
def save_artifacts(context: OpExecutionContext, model: Any, metrics: dict, cfg: dict) -> Path:
    """Save model artifacts to disk."""
    import joblib

    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_filename = f"{cfg['model']['type']}_pipeline.pkl"
    model_path = models_dir / model_filename
    joblib.dump(model, model_path)

    # Save metrics
    metrics_filename = f"{cfg['model']['type']}_metrics.yaml"
    metrics_path = models_dir / metrics_filename
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)

    context.log.info(f"Artifacts saved: {model_path}")
    return model_path


# ==============================================================================
# OP 9: DVC Add
# ==============================================================================


@op(description="Add model to DVC tracking", tags={"stage": "versioning"})
def dvc_add(context: OpExecutionContext, model_path: Path) -> Path:
    """Add model to DVC tracking."""
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


# ==============================================================================
# OP 10: Send Notification
# ==============================================================================


@op(description="Send notification about pipeline completion", tags={"stage": "notification"})
def send_notification(context: OpExecutionContext, metrics: dict, run_id: str, cfg: dict) -> None:
    """Send notification about pipeline completion."""
    from src.utils.notification_utils import NotificationManager

    try:
        channels = cfg["notifications"]["channels"]
        notification_manager = NotificationManager(channels=channels)
        notification_manager.send("success", metrics, run_id, cfg)

        context.log.info(f"Notifications sent to: {', '.join(channels)}")

    except Exception as e:
        context.log.warning(f"Notification failed (non-critical): {e}")


# ==============================================================================
# JOB: Complete Training Pipeline
# ==============================================================================


@job(
    name="training_pipeline",
    description="Complete ML training pipeline with 10 separate ops",
    tags={"pipeline": "training", "model": "xgboost"},
)
def complete_training_job():
    """
    Complete training pipeline with 10 separate ops.

    Each op is visible individually in Dagster UI.
    """
    # Step 1: Load config
    cfg = load_config()

    # Step 2: Load data
    data = load_data(cfg)

    # Step 3: Validate data
    validation = validate_data(data)

    # Step 4: Train model
    model = train_model(cfg, data)

    # Step 5: Evaluate model
    metrics = evaluate_model(model, cfg, data)

    # Step 6: Check threshold
    threshold_result = check_threshold(metrics, cfg)

    # Step 7: Log to MLflow
    run_id = log_mlflow(model, metrics, cfg)

    # Step 8: Save artifacts
    model_path = save_artifacts(model, metrics, cfg)

    # Step 9: DVC add
    dvc_file = dvc_add(model_path)

    # Step 10: Send notification
    send_notification(metrics, run_id, cfg)
