"""
Chronos-2 Zero-Shot Evaluation Script.

This script evaluates the Chronos-2 foundation model (120M parameters)
on the steel energy consumption dataset using zero-shot inference.
Chronos-2 is the latest model from Amazon Science (Oct 2025) with state-of-the-art
performance on multivariate and covariate-informed forecasting.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
from chronos import Chronos2Pipeline

from src.utils.model_evaluation import calculate_regression_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Detect and setup GPU if available."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def load_data(data_dir: Path):
    """Load preprocessed datasets."""
    logger.info("Loading preprocessed data")

    df_train = pl.read_parquet(data_dir / "steel_preprocessed_train.parquet")
    df_val = pl.read_parquet(data_dir / "steel_preprocessed_val.parquet")
    df_test = pl.read_parquet(data_dir / "steel_preprocessed_test.parquet")

    logger.info(f"Train: {len(df_train)} samples")
    logger.info(f"Val: {len(df_val)} samples")
    logger.info(f"Test: {len(df_test)} samples")

    return df_train, df_val, df_test


def evaluate_chronos2(
    pipeline,
    df: pl.DataFrame,
    target_col: str = "Usage_kWh",
    context_length: int = 512,
    prediction_length: int = 1,
    batch_size: int = 512,
    max_predictions: int | None = None,
):
    """
    Evaluate Chronos-2 model on dataset with batch processing.

    Args:
        pipeline: Loaded Chronos2Pipeline
        df: DataFrame with time series data
        target_col: Target column name
        context_length: Number of historical points
        prediction_length: Number of steps to predict
        batch_size: Number of predictions per batch (for GPU efficiency)
        max_predictions: Maximum number of predictions (None = all)

    Returns:
        dict with predictions and metrics
    """

    logger.info(f"Evaluating Chronos-2 with context_length={context_length}, batch_size={batch_size}")

    y_true = df[target_col].to_numpy()
    predictions = []
    n_predictions = len(y_true) - context_length
    
    # Limit predictions if max_predictions is set
    if max_predictions is not None:
        n_predictions = min(n_predictions, max_predictions)
        logger.info(f"Limiting to {max_predictions} predictions for quick testing")
    
    n_batches = (n_predictions + batch_size - 1) // batch_size

    logger.info(f"Generating {n_predictions} predictions in {n_batches} batches")

    # Generate predictions using batched sliding windows
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_predictions)
        batch_contexts = []

        # Prepare batch
        for i in range(start_idx, end_idx):
            context = y_true[context_length + i - context_length : context_length + i]
            batch_contexts.append(context)

        # Stack into batch tensor [batch, variate, time]
        batch_tensor = torch.tensor(np.array(batch_contexts), dtype=torch.float32).unsqueeze(1)

        # Predict batch
        forecasts = pipeline.predict(
            inputs=batch_tensor,
            prediction_length=prediction_length,
        )

        # Debug on first batch
        if batch_idx == 0:
            logger.info(f"Batch forecast shape: {forecasts[0].shape}")
            logger.info(f"Using median quantile (index {forecasts[0].shape[1] // 2})")

        # Extract predictions from batch
        median_idx = forecasts[0].shape[1] // 2
        for forecast in forecasts:
            pred = float(forecast[0, median_idx, 0].item())
            predictions.append(pred)

        # Progress logging
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            completed = min(end_idx, n_predictions)
            logger.info(f"Progress: {completed}/{n_predictions} predictions ({100 * completed / n_predictions:.1f}%)")

    predictions = np.array(predictions)
    y_true_eval = y_true[context_length : context_length + n_predictions]

    # Calculate metrics
    metrics = calculate_regression_metrics(y_true_eval, predictions)

    logger.info(f"RMSE: {metrics['rmse']:.4f} kWh")
    logger.info(f"MAE: {metrics['mae']:.4f} kWh")
    logger.info(f"R2: {metrics['r2']:.4f}")

    return {
        "y_true": y_true_eval,
        "y_pred": predictions,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chronos-2 model")
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length for predictions",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=10,
        help="Maximum number of predictions to generate (default: 10 for quick testing)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="steel_energy_chronos2",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Chronos-2 Zero-Shot Evaluation")
    logger.info(f"Context length: {args.context_length}")

    # Setup
    device = setup_gpu()
    data_dir = Path("data/processed")
    output_dir = Path("models/foundation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_train, df_val, df_test = load_data(data_dir)

    # Load Chronos-2 model
    logger.info("Loading Chronos-2 model (120M parameters)")
    model_name = "s3://autogluon/chronos-2"

    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    logger.info(f"Model loaded: {model_name}")

    # Setup MLflow - use http://localhost:5000 to connect to Docker MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.experiment_name)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    with mlflow.start_run(run_name=f"chronos2_{timestamp}"):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", "chronos2")
        mlflow.log_param("model_params", "120M")
        mlflow.log_param("context_length", args.context_length)
        mlflow.log_param("device", device)
        mlflow.log_param("approach", "zero_shot")

        # Log max_predictions parameter
        mlflow.log_param("max_predictions", args.max_predictions if args.max_predictions else "all")

        # Evaluate on test set
        logger.info("Evaluating on test set")
        results = evaluate_chronos2(
            pipeline=pipeline,
            df=df_test,
            target_col="Usage_kWh",
            context_length=args.context_length,
            max_predictions=args.max_predictions,
            prediction_length=1,
        )

        # Log metrics
        for key, value in results["metrics"].items():
            mlflow.log_metric(f"test_{key}", value)

        # Save results
        results_file = output_dir / f"chronos2_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "context_length": args.context_length,
                    "metrics": results["metrics"],
                    "timestamp": timestamp,
                },
                f,
                indent=2,
            )

        mlflow.log_artifact(str(results_file))

        logger.info(f"Results saved to: {results_file}")
        logger.info("Evaluation completed")

    return results


if __name__ == "__main__":
    main()
