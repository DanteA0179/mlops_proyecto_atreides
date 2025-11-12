"""
Dagster Ops for Chronos-2 Foundation Model Training Pipeline.

These ops handle the specific requirements of Chronos-2 models:
- Temporal data format (sequences vs tabular)
- PyTorch/HuggingFace API
- Probabilistic evaluation
- Three variants: zero-shot, fine-tuned, fine-tuned with covariates
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml
from chronos import Chronos2Pipeline
from dagster import Config, OpExecutionContext, op

logger = logging.getLogger(__name__)


class ChronosConfig(Config):
    """Configuration for Chronos-2 pipeline."""

    config_path: str = "config/training/chronos2_zeroshot_config.yaml"


# ==============================================================================
# OP 1: Load Chronos Configuration
# ==============================================================================


@op(
    name="load_chronos_config",
    description="Load Chronos-2 configuration from YAML file",
    tags={"stage": "setup", "model": "chronos"},
)
def load_chronos_config_op(context: OpExecutionContext, config: ChronosConfig) -> dict:
    """
    Load Chronos-2 configuration from YAML file.

    Supports three model types:
    - chronos2_zeroshot: Zero-shot inference
    - chronos2_finetuned: Fine-tuning without covariates
    - chronos2_covariates: Fine-tuning with past covariates
    """
    try:
        config_path = config.config_path
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        model_type = cfg["model"]["type"]
        variant = cfg["model"]["variant"]

        context.log.info(f"Config loaded: {model_type} ({variant})")
        return cfg

    except Exception as e:
        context.log.error(f"Failed to load Chronos config: {e}")
        raise


# ==============================================================================
# OP 2: Load Data
# ==============================================================================


@op(
    name="load_chronos_data",
    description="Load preprocessed train/val/test data for Chronos-2",
    tags={"stage": "data", "model": "chronos"},
)
def load_chronos_data_op(context: OpExecutionContext, cfg: dict):
    """
    Load preprocessed data from parquet files.

    Parameters
    ----------
    cfg : dict
        Configuration with data paths

    Returns
    -------
    tuple
        (train, val, test) Polars DataFrames
    """
    try:
        train = pl.read_parquet(cfg["data"]["train"])
        val = pl.read_parquet(cfg["data"]["val"])
        test = pl.read_parquet(cfg["data"]["test"])

        context.log.info(f"Data loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}")
        return train, val, test

    except Exception as e:
        context.log.error(f"Failed to load data: {e}")
        raise


# ==============================================================================
# OP 3: Load Chronos Pipeline
# ==============================================================================


@op(
    name="load_chronos_pipeline",
    description="Load pre-trained Chronos-2 pipeline with GPU support",
    tags={"stage": "setup", "model": "chronos"},
)
def load_chronos_pipeline_op(context: OpExecutionContext, cfg: dict):
    """
    Load Chronos-2 pre-trained pipeline.

    Automatically detects GPU and loads model accordingly.
    """
    try:
        # Use the S3 path that works (from US-014)
        model_name = "s3://autogluon/chronos-2"

        # Detect GPU
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            context.log.info(f"Using GPU: {gpu_name}")
        else:
            device = "cpu"
            context.log.info("Using CPU")

        # Load pipeline (using torch_dtype for now, will be deprecated later)
        context.log.info(f"Loading Chronos-2 pipeline: {model_name}")
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

        context.log.info(f"Pipeline loaded successfully on {device}")
        return pipeline

    except Exception as e:
        context.log.error(f"Failed to load Chronos pipeline: {e}")
        raise


# ==============================================================================
# OP 4: Prepare Chronos Data
# ==============================================================================


@op(
    name="prepare_chronos_data",
    description="Prepare temporal data for Chronos-2",
    tags={"stage": "data", "model": "chronos"},
)
def prepare_chronos_data_op(
    context: OpExecutionContext,
    data: tuple,
    cfg: dict,
):
    """
    Prepare data in temporal format for Chronos-2.

    Returns dict with:
    - train_data: Training sequences
    - val_data: Validation sequences
    - test_data: Test sequences
    - target_col: Target column name
    - covariates: List of covariate columns (if applicable)
    """
    try:
        train, val, test = data
        target_col = cfg["data"]["target_col"]
        model_type = cfg["model"]["type"]

        # Get covariates if applicable
        covariates = None
        if model_type == "chronos2_covariates":
            covariates = cfg["chronos"].get("past_covariates", [])
            context.log.info(f"Using {len(covariates)} past covariates")

        prepared_data = {
            "train": train,
            "val": val,
            "test": test,
            "target_col": target_col,
            "covariates": covariates,
        }

        context.log.info(f"Data prepared for {model_type}")
        context.log.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        return prepared_data

    except Exception as e:
        context.log.error(f"Failed to prepare Chronos data: {e}")
        raise


# ==============================================================================
# OP 5: Train/Evaluate Chronos Model
# ==============================================================================


@op(
    name="train_chronos_model",
    description="Train or evaluate Chronos-2 model (zero-shot or fine-tuned)",
    tags={"stage": "training", "model": "chronos"},
)
def train_chronos_model_op(
    context: OpExecutionContext,
    pipeline: Any,
    prepared_data: dict,
    cfg: dict,
):
    """
    Train or evaluate Chronos-2 model based on type.

    - chronos2_zeroshot: Direct evaluation (no training)
    - chronos2_finetuned: Fine-tune without covariates
    - chronos2_covariates: Fine-tune with past covariates
    """
    try:
        model_type = cfg["model"]["type"]

        if model_type == "chronos2_zeroshot":
            # Zero-shot: just return pipeline (no training)
            context.log.info("Zero-shot mode: using pre-trained pipeline")
            return pipeline

        elif model_type == "chronos2_finetuned":
            # Fine-tune without covariates
            context.log.info("Fine-tuning without covariates...")
            pipeline = _finetune_chronos_simple(context, pipeline, prepared_data, cfg)

        elif model_type == "chronos2_covariates":
            # Fine-tune with covariates
            context.log.info("Fine-tuning with covariates...")
            pipeline = _finetune_chronos_covariates(context, pipeline, prepared_data, cfg)

        else:
            raise ValueError(f"Unknown Chronos model type: {model_type}")

        context.log.info("Training/evaluation completed")
        return pipeline

    except Exception as e:
        context.log.error(f"Failed to train Chronos model: {e}")
        raise


def _finetune_chronos_simple(context, pipeline, prepared_data, cfg):
    """Fine-tune Chronos-2 without covariates."""
    from src.models.chronos2_finetuning import finetune_chronos2

    train_df = prepared_data["train"]
    val_df = prepared_data["val"]
    target_col = prepared_data["target_col"]

    chronos_config = cfg["chronos"]

    context.log.info(f"Fine-tuning with {chronos_config['num_steps']} steps")

    finetuned_pipeline = finetune_chronos2(
        pipeline=pipeline,
        df_train=train_df,
        df_val=val_df,
        target_col=target_col,
        prediction_length=chronos_config.get("prediction_length", 1),
        num_steps=chronos_config.get("num_steps", 1000),
        learning_rate=chronos_config.get("learning_rate", 1e-5),
        batch_size=chronos_config.get("batch_size", 8),
        gradient_accumulation_steps=chronos_config.get("gradient_accumulation_steps", 4),
    )

    return finetuned_pipeline


def _finetune_chronos_covariates(context, pipeline, prepared_data, cfg):
    """Fine-tune Chronos-2 with past covariates."""
    from src.models.chronos2_finetuning_covariates import finetune_chronos2_with_covariates

    train_df = prepared_data["train"]
    val_df = prepared_data["val"]
    target_col = prepared_data["target_col"]
    covariates = prepared_data["covariates"]

    chronos_config = cfg["chronos"]

    context.log.info(f"Fine-tuning with {len(covariates)} covariates")

    finetuned_pipeline = finetune_chronos2_with_covariates(
        pipeline=pipeline,
        df_train=train_df,
        df_val=val_df,
        target_col=target_col,
        prediction_length=chronos_config.get("prediction_length", 1),
        num_steps=chronos_config.get("num_steps", 1000),
        learning_rate=chronos_config.get("learning_rate", 1e-5),
        batch_size=chronos_config.get("batch_size", 8),
        gradient_accumulation_steps=chronos_config.get("gradient_accumulation_steps", 4),
        past_covariates=covariates,
        future_covariates=None,  # Not used during training
    )

    return finetuned_pipeline


# ==============================================================================
# OP 6: Evaluate Chronos Model
# ==============================================================================


@op(
    name="evaluate_chronos_model",
    description="Evaluate Chronos-2 model on validation and test sets",
    tags={"stage": "evaluation", "model": "chronos"},
)
def evaluate_chronos_model_op(
    context: OpExecutionContext,
    pipeline: Any,
    prepared_data: dict,
    cfg: dict,
):
    """
    Evaluate Chronos-2 model on validation and test sets.

    Uses batch processing for GPU efficiency.
    """
    from src.models.train_chronos2 import evaluate_chronos2

    try:
        val_df = prepared_data["val"]
        test_df = prepared_data["test"]
        target_col = prepared_data["target_col"]

        chronos_config = cfg["chronos"]
        context_length = chronos_config.get("context_length", 512)
        batch_size = chronos_config.get("batch_size", 512)
        max_predictions = chronos_config.get("max_predictions")

        # Evaluate on validation set
        context.log.info("Evaluating on validation set...")
        val_results = evaluate_chronos2(
            pipeline=pipeline,
            df=val_df,
            target_col=target_col,
            context_length=context_length,
            prediction_length=1,
            batch_size=batch_size,
            max_predictions=max_predictions,
        )

        # Evaluate on test set
        context.log.info("Evaluating on test set...")
        test_results = evaluate_chronos2(
            pipeline=pipeline,
            df=test_df,
            target_col=target_col,
            context_length=context_length,
            prediction_length=1,
            batch_size=batch_size,
            max_predictions=max_predictions,
        )

        metrics = {
            "val": val_results["metrics"],
            "test": test_results["metrics"],
        }

        context.log.info(
            f"Val RMSE: {metrics['val']['rmse']:.4f} kWh, R²: {metrics['val']['r2']:.4f}"
        )
        context.log.info(
            f"Test RMSE: {metrics['test']['rmse']:.4f} kWh, R²: {metrics['test']['r2']:.4f}"
        )

        return metrics

    except Exception as e:
        context.log.error(f"Failed to evaluate Chronos model: {e}")
        raise


# ==============================================================================
# OP 7: Save Chronos Model
# ==============================================================================


@op(
    name="save_chronos_model",
    description="Save fine-tuned Chronos-2 model",
    tags={"stage": "saving", "model": "chronos"},
)
def save_chronos_model_op(
    context: OpExecutionContext,
    pipeline: Any,
    cfg: dict,
):
    """
    Save fine-tuned Chronos-2 model to disk.

    Zero-shot models are not saved (use pre-trained).
    """
    from datetime import datetime

    try:
        model_type = cfg["model"]["type"]

        if model_type == "chronos2_zeroshot":
            context.log.info("Zero-shot model not saved (using pre-trained)")
            return None

        # Create save directory
        models_dir = Path(cfg["save"]["model_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"
        model_path = models_dir / model_name

        # Save model
        context.log.info(f"Saving model to {model_path}")
        pipeline.save_pretrained(model_path)

        context.log.info(f"Model saved: {model_path}")
        return model_path

    except Exception as e:
        context.log.error(f"Failed to save Chronos model: {e}")
        raise


# ==============================================================================
# OP 8: Log Chronos to MLflow
# ==============================================================================


@op(
    name="log_chronos_mlflow",
    description="Log Chronos-2 model and metrics to MLflow",
    tags={"stage": "logging", "model": "chronos"},
)
def log_chronos_mlflow_op(
    context: OpExecutionContext,
    metrics: dict,
    cfg: dict,
    model_path: Path | None = None,
):
    """
    Log Chronos-2 results to MLflow.

    Note: Model files are NOT logged to MLflow (too large, ~455MB).
    Only metrics and model path are logged.
    """
    import mlflow

    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log tags
            for tag_key, tag_value in cfg["mlflow"]["tags"].items():
                mlflow.set_tag(tag_key, tag_value)

            # Log Chronos parameters
            chronos_params = cfg["chronos"].copy()
            # Remove lists from params (MLflow doesn't support)
            if "past_covariates" in chronos_params and chronos_params["past_covariates"]:
                chronos_params["num_covariates"] = len(chronos_params["past_covariates"])
                chronos_params.pop("past_covariates")
            if "future_covariates" in chronos_params:
                chronos_params.pop("future_covariates")

            mlflow.log_params(chronos_params)

            # Log metrics
            for metric_name, metric_value in metrics["val"].items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)

            for metric_name, metric_value in metrics["test"].items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            # Log model path (not the model itself)
            if model_path:
                mlflow.log_param("model_path", str(model_path))

            context.log.info(f"MLflow run: {run_id}")
            return run_id

    except Exception as e:
        context.log.error(f"MLflow logging failed: {e}")
        raise
