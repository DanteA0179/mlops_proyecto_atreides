"""
Prepare reference dataset for monitoring.

This script samples 1,000 rows from the training data to create
a reference dataset for Evidently drift detection.

Usage
-----
poetry run python scripts/prepare_reference_data.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_reference_data() -> None:
    """
    Sample 1,000 rows from training data as reference.

    Creates a CSV file with 1,000 random samples from the training set
    that will be used as reference data for drift detection.

    Raises
    ------
    FileNotFoundError
        If training data file does not exist
    """
    # Define paths
    train_data_path = Path("data/processed/steel_preprocessed_train.parquet")
    output_path = Path("data/monitoring/reference_data.csv")

    # Validate input exists
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}. " f"Run preprocessing pipeline first."
        )

    logger.info(f"Loading training data from {train_data_path}")
    df_train = pd.read_parquet(train_data_path)

    logger.info(f"Training data shape: {df_train.shape}")

    # Sample 1,000 rows
    sample_size = min(1000, len(df_train))
    df_ref = df_train.sample(n=sample_size, random_state=42)

    logger.info(f"Sampled {sample_size} rows for reference data")

    # Remove target column (we only monitor features, not targets)
    if "Usage_kWh" in df_ref.columns:
        df_ref = df_ref.drop("Usage_kWh", axis=1)
        logger.info("Removed target column 'Usage_kWh' (monitoring features only)")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df_ref.to_csv(output_path, index=False)

    logger.info(f"Reference data saved: {output_path}")
    logger.info(f"Reference data shape: {df_ref.shape}")
    logger.info(f"Columns: {list(df_ref.columns)}")


if __name__ == "__main__":
    prepare_reference_data()
