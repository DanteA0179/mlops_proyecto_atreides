"""
Simplified Dagster Pipeline - Working Version.

This is a simplified version that actually works with Dagster's execution model.
"""

from pathlib import Path

import polars as pl
import yaml
from dagster import Config, OpExecutionContext, job, op


class TrainingConfig(Config):
    """Configuration for training pipeline."""

    config_path: str = "config/training/xgboost_config.yaml"


@op
def load_and_train(context: OpExecutionContext, config: TrainingConfig) -> dict:
    """
    Load config, data, train model, and return results.

    This simplified op combines multiple steps to avoid Dagster's complex dependency management.
    """
    # Load config
    config_file = Path(config.config_path)
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    context.log.info(f"Config loaded: {cfg['model']['type']}")

    # Load data
    train = pl.read_parquet(cfg["data"]["train"])
    val = pl.read_parquet(cfg["data"]["val"])
    test = pl.read_parquet(cfg["data"]["test"])

    context.log.info(f"Data loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    # Train model
    from src.models.xgboost_trainer import check_gpu_availability, create_xgboost_pipeline

    target_col = cfg["data"]["target_col"]
    X_train = train.drop(target_col).to_numpy()
    y_train = train[target_col].to_numpy()
    X_test = test.drop(target_col).to_numpy()
    y_test = test[target_col].to_numpy()

    gpu_available, device_info = check_gpu_availability()
    context.log.info(f"GPU: {gpu_available}, Device: {device_info}")

    hyperparams = cfg["hyperparameters"].copy()
    hyperparams["tree_method"] = "hist"
    hyperparams["device"] = "cuda:0" if gpu_available else "cpu"

    pipeline = create_xgboost_pipeline(hyperparams)
    pipeline.fit(X_train, y_train)

    context.log.info("Training completed")

    # Evaluate
    from src.models.xgboost_trainer import evaluate_model

    test_metrics = evaluate_model(pipeline, X_test, y_test, "test")

    context.log.info(f"Test RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")

    return {
        "status": "success",
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }


@job
def simple_training_job():
    """Simple training job that works."""
    load_and_train()
