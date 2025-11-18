"""
Generate reference data for drift monitoring.

This script extracts a stratified sample from the training set and adds
model predictions to create reference data for Evidently AI drift detection.

Usage:
    python scripts/generate_reference_data.py \
        --train-data data/processed/steel_preprocessed_train.parquet \
        --model-path models/ensembles/ensemble_lightgbm_v1.pkl \
        --output reports/monitoring/reference_data/train_data_sample.parquet \
        --sample-size 10000

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import argparse
import logging
from pathlib import Path

import joblib
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_reference_data(
    train_data_path: Path,
    model_path: Path | None,
    output_path: Path,
    sample_size: int = 10000,
    stratify_column: str = "Load_Type",
    seed: int = 42,
) -> None:
    """
    Generate reference data for drift monitoring.

    Parameters
    ----------
    train_data_path : Path
        Path to training data parquet file
    model_path : Optional[Path]
        Path to trained model pickle file (if None, predictions will not be added)
    output_path : Path
        Path to save reference data
    sample_size : int, default=10000
        Number of samples to extract
    stratify_column : str, default="Load_Type"
        Column to stratify sampling
    seed : int, default=42
        Random seed for reproducibility
    """
    logger.info(f"Loading training data from {train_data_path}")
    train_df = pl.read_parquet(train_data_path)

    logger.info(f"Original training data shape: {train_df.shape}")

    # Verify stratify column exists
    if stratify_column not in train_df.columns:
        logger.warning(
            f"Stratify column '{stratify_column}' not found. "
            f"Using simple random sampling instead."
        )
        stratify_column = None

    # Stratified sampling
    if stratify_column:
        logger.info(f"Performing stratified sampling by '{stratify_column}'")

        # Get proportions of each class
        class_counts = train_df.group_by(stratify_column).agg(pl.count().alias("count"))
        total_count = train_df.height

        # Sample from each stratum proportionally
        samples = []
        for row in class_counts.iter_rows(named=True):
            stratum_value = row[stratify_column]
            stratum_count = row["count"]
            stratum_proportion = stratum_count / total_count
            stratum_sample_size = int(sample_size * stratum_proportion)

            stratum_df = train_df.filter(pl.col(stratify_column) == stratum_value).sample(
                n=min(stratum_sample_size, stratum_count), seed=seed
            )

            samples.append(stratum_df)
            logger.info(
                f"  {stratify_column}={stratum_value}: "
                f"{stratum_sample_size} samples ({stratum_proportion:.2%})"
            )

        train_sample = pl.concat(samples)
    else:
        logger.info("Performing simple random sampling")
        train_sample = train_df.sample(n=min(sample_size, train_df.height), seed=seed)

    logger.info(f"Sample shape: {train_sample.shape}")

    # Add predictions if model provided
    if model_path and model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Prepare features (drop target if present)
        target_column = "Usage_kWh"
        if target_column in train_sample.columns:
            X = train_sample.drop(target_column).to_pandas()
        else:
            X = train_sample.to_pandas()
            logger.warning(f"Target column '{target_column}' not found in data")

        logger.info("Generating predictions")
        predictions = model.predict(X)

        # Add predictions to sample
        train_sample = train_sample.with_columns(pl.Series("predictions", predictions))
        logger.info("Predictions added successfully")
    else:
        if model_path:
            logger.warning(f"Model not found at {model_path}. Skipping predictions.")
        else:
            logger.info("No model path provided. Skipping predictions.")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save reference data
    logger.info(f"Saving reference data to {output_path}")
    train_sample.write_parquet(output_path)

    logger.info(f"Reference data saved successfully: {train_sample.shape}")

    # Print statistics
    logger.info("\nReference Data Statistics:")
    logger.info(f"  Total samples: {train_sample.height}")
    logger.info(f"  Total features: {train_sample.width}")
    logger.info(f"  Memory size: {train_sample.estimated_size('mb'):.2f} MB")

    if stratify_column and stratify_column in train_sample.columns:
        class_distribution = (
            train_sample.group_by(stratify_column)
            .agg(pl.count().alias("count"))
            .with_columns((pl.col("count") / train_sample.height * 100).alias("percentage"))
            .sort(stratify_column)
        )

        logger.info(f"\n  Distribution by {stratify_column}:")
        for row in class_distribution.iter_rows(named=True):
            logger.info(f"    {row[stratify_column]}: {row['count']} ({row['percentage']:.2f}%)")


def main():
    """Main function to parse arguments and generate reference data."""
    parser = argparse.ArgumentParser(
        description="Generate reference data for drift monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/steel_preprocessed_train.parquet"),
        help="Path to training data parquet file",
    )

    parser.add_argument(
        "--model-path", type=Path, default=None, help="Path to trained model pickle file (optional)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/monitoring/reference_data/train_data_sample.parquet"),
        help="Path to save reference data",
    )

    parser.add_argument(
        "--sample-size", type=int, default=10000, help="Number of samples to extract"
    )

    parser.add_argument(
        "--stratify-column", type=str, default="Load_Type", help="Column to stratify sampling"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    generate_reference_data(
        train_data_path=args.train_data,
        model_path=args.model_path,
        output_path=args.output,
        sample_size=args.sample_size,
        stratify_column=args.stratify_column,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
