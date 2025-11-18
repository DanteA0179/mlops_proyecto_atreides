"""
Chronos-2 Fine-Tuning Implementation.

Based on official Chronos-2 quickstart notebook:
https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb
"""

import logging
from pathlib import Path

import polars as pl
from chronos import Chronos2Pipeline

from src.utils.chronos_data_prep import prepare_chronos_finetuning_data, validate_finetuning_data

logger = logging.getLogger(__name__)


def finetune_chronos2(
    pipeline: Chronos2Pipeline,
    df_train: pl.DataFrame,
    df_val: pl.DataFrame | None = None,
    target_col: str = "Usage_kWh",
    prediction_length: int = 1,
    num_steps: int = 1000,
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    logging_steps: int = 10,
    gradient_accumulation_steps: int = 1,
    past_covariates: list[str] | None = None,
    future_covariates: list[str] | None = None,
) -> Chronos2Pipeline:
    """
    Fine-tune Chronos-2 model on custom data.

    This function uses the official Chronos-2 `.fit()` method which handles:
    - Tokenization of time series values
    - Training loop with gradient descent
    - Validation monitoring (if validation data provided)
    - Model checkpointing

    Args:
        pipeline: Pre-loaded Chronos2Pipeline
        df_train: Training DataFrame
        df_val: Optional validation DataFrame
        target_col: Target column name
        prediction_length: Forecast horizon
        num_steps: Number of training steps
        learning_rate: Learning rate for AdamW optimizer
        batch_size: Batch size for training
        logging_steps: Log training metrics every N steps
        past_covariates: Optional list of past covariate column names
        future_covariates: Optional list of future covariate column names

    Returns:
        Fine-tuned Chronos2Pipeline

    Examples:
        >>> from chronos import Chronos2Pipeline
        >>> import polars as pl
        >>>
        >>> # Load base model
        >>> pipeline = Chronos2Pipeline.from_pretrained("s3://autogluon/chronos-2")
        >>>
        >>> # Load data
        >>> df_train = pl.read_parquet("data/processed/steel_preprocessed_train.parquet")
        >>> df_val = pl.read_parquet("data/processed/steel_preprocessed_val.parquet")
        >>>
        >>> # Fine-tune
        >>> finetuned = finetune_chronos2(
        ...     pipeline=pipeline,
        ...     df_train=df_train,
        ...     df_val=df_val,
        ...     num_steps=1000,
        ... )
    """
    logger.info("=" * 70)
    logger.info("Chronos-2 Fine-Tuning")
    logger.info("=" * 70)

    # Prepare training data
    logger.info("Preparing training data")
    train_inputs = prepare_chronos_finetuning_data(
        df=df_train,
        target_col=target_col,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )

    train_stats = validate_finetuning_data(train_inputs)
    logger.info(f"Training series: {train_stats['n_series']}")
    logger.info(f"Avg length: {train_stats['avg_length']:.0f} timesteps")

    # Prepare validation data (optional)
    validation_inputs = None
    if df_val is not None:
        logger.info("Preparing validation data")
        validation_inputs = prepare_chronos_finetuning_data(
            df=df_val,
            target_col=target_col,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        val_stats = validate_finetuning_data(validation_inputs)
        logger.info(f"Validation series: {val_stats['n_series']}")

    # Fine-tune
    logger.info("=" * 70)
    logger.info("Starting fine-tuning")
    logger.info(f"  Steps: {num_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Prediction length: {prediction_length}")
    logger.info("=" * 70)

    # Clear GPU cache before training
    if pipeline.model.device.type == "cuda":
        import torch

        torch.cuda.empty_cache()
        logger.info("GPU cache cleared before training")

    try:
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            prediction_length=prediction_length,
            validation_inputs=validation_inputs,
            num_steps=num_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
        )

        logger.info("=" * 70)
        logger.info("Fine-tuning completed successfully!")
        logger.info("=" * 70)

        return finetuned_pipeline

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise RuntimeError(f"Fine-tuning failed: {e}") from e


def save_finetuned_pipeline(
    pipeline: Chronos2Pipeline,
    save_path: Path,
) -> None:
    """
    Save fine-tuned pipeline to disk.

    The pipeline is saved in HuggingFace format and can be reloaded
    using Chronos2Pipeline.from_pretrained().

    Args:
        pipeline: Fine-tuned Chronos2Pipeline
        save_path: Directory to save model

    Examples:
        >>> save_finetuned_pipeline(finetuned, Path("models/foundation/chronos2_finetuned"))
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving fine-tuned model to: {save_path}")

    try:
        # Chronos-2 pipelines can be saved like HuggingFace models
        pipeline.model.save_pretrained(save_path)

        # Also save tokenizer if available
        if hasattr(pipeline, "tokenizer"):
            pipeline.tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved successfully to: {save_path}")

        # Log file size
        total_size = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        logger.info(f"Total size: {size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise RuntimeError(f"Model save failed: {e}") from e


def load_finetuned_pipeline(
    load_path: Path,
    device: str = "cuda",
) -> Chronos2Pipeline:
    """
    Load fine-tuned pipeline from disk.

    Args:
        load_path: Directory containing saved model
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded Chronos2Pipeline

    Examples:
        >>> pipeline = load_finetuned_pipeline(
        ...     Path("models/foundation/chronos2_finetuned"),
        ...     device="cuda"
        ... )
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Model not found at: {load_path}")

    logger.info(f"Loading fine-tuned model from: {load_path}")

    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            str(load_path),
            device_map=device,
        )

        logger.info(f"Model loaded successfully from: {load_path}")

        return pipeline

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}") from e
