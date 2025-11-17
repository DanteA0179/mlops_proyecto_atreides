"""
Simulate drift in production data for testing monitoring system.

This script creates synthetic production data with configurable drift levels.

Usage:
    python scripts/simulate_drift.py \
        --reference-data reports/monitoring/reference_data/train_data_sample.parquet \
        --output data/production/drift_test.parquet \
        --drift-level 0.3 \
        --sample-size 2000

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_drift(
    reference_data_path: Path,
    output_path: Path,
    drift_level: float = 0.3,
    sample_size: int = 2000,
    seed: int = 42,
) -> None:
    """
    Simulate drift in production data.

    Parameters
    ----------
    reference_data_path : Path
        Path to reference data
    output_path : Path
        Path to save simulated production data
    drift_level : float, default=0.3
        Drift intensity (0.0 = no drift, 1.0 = maximum drift)
    sample_size : int, default=2000
        Number of samples to generate
    seed : int, default=42
        Random seed
    """
    logger.info(f"Loading reference data from {reference_data_path}")
    ref_df = pl.read_parquet(reference_data_path)

    logger.info(f"Original data shape: {ref_df.shape}")

    # Sample from reference data
    np.random.seed(seed)
    sample_df = ref_df.sample(n=min(sample_size, ref_df.height), seed=seed)

    logger.info(f"Sampled {sample_df.height} rows")

    # Apply drift to numerical columns
    logger.info(f"Applying drift (level={drift_level})")

    # Get numerical columns
    numerical_cols = [
        col
        for col in sample_df.columns
        if sample_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        and col != "Usage_kWh"  # Don't drift the target
    ]

    logger.info(f"Drifting numerical columns: {numerical_cols}")

    # Convert to pandas for easier manipulation
    df_pandas = sample_df.to_pandas()

    for col in numerical_cols:
        if col not in df_pandas.columns:
            continue

        mean = df_pandas[col].mean()
        std = df_pandas[col].std()

        if std == 0:
            continue

        # Add drift as shift in mean
        drift_shift = drift_level * std * np.random.uniform(0.5, 1.5)

        # Apply drift to random portion of data
        drift_mask = np.random.rand(len(df_pandas)) < drift_level
        df_pandas.loc[drift_mask, col] += drift_shift

        logger.info(f"  {col}: shifted {drift_mask.sum()} values by {drift_shift:.3f}")

    # Convert back to Polars
    drifted_df = pl.from_pandas(df_pandas)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    logger.info(f"Saving simulated data to {output_path}")
    drifted_df.write_parquet(output_path)

    logger.info(f"Simulation complete: {drifted_df.shape}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simulate drift in production data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--reference-data",
        type=Path,
        default=Path("reports/monitoring/reference_data/train_data_sample.parquet"),
        help="Path to reference data",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/production/drift_test.parquet"),
        help="Path to save simulated production data",
    )

    parser.add_argument(
        "--drift-level",
        type=float,
        default=0.3,
        help="Drift intensity (0.0 = no drift, 1.0 = maximum drift)",
    )

    parser.add_argument(
        "--sample-size", type=int, default=2000, help="Number of samples to generate"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    simulate_drift(
        reference_data_path=args.reference_data,
        output_path=args.output,
        drift_level=args.drift_level,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
