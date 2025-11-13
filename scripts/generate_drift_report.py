"""
Generate Evidently data drift report.

This script compares production predictions against reference data
to detect data drift in input features using Evidently AI.

Usage
-----
poetry run python scripts/generate_drift_report.py

Requirements
------------
- reference_data.csv must exist (run prepare_reference_data.py first)
- predictions.csv must have at least 50 rows for meaningful analysis
"""

import logging
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_drift_report() -> None:
    """
    Generate Evidently data drift report.

    Compares production predictions against reference data to detect drift.
    Generates an HTML report with visualizations and statistics.

    Raises
    ------
    FileNotFoundError
        If reference data or predictions CSV does not exist
    ValueError
        If data has insufficient rows or mismatched columns
    """
    # Define paths
    reference_path = Path("data/monitoring/reference_data.csv")
    production_path = Path("data/monitoring/predictions.csv")
    output_path = Path("reports/monitoring/drift_report.html")

    # Validate inputs exist
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference data not found: {reference_path}. "
            f"Run: poetry run python scripts/prepare_reference_data.py"
        )

    if not production_path.exists():
        raise FileNotFoundError(
            f"Production predictions not found: {production_path}. "
            f"Make predictions through API first to generate data."
        )

    # Load reference data
    logger.info(f"Loading reference data from {reference_path}")
    reference_df = pd.read_csv(reference_path)
    logger.info(f"Reference data shape: {reference_df.shape}")

    # Load production data
    logger.info(f"Loading production data from {production_path}")
    production_df = pd.read_csv(production_path)
    logger.info(f"Production data shape (raw): {production_df.shape}")

    # Validate minimum rows
    if len(production_df) < 20:
        logger.warning(
            f"Production data has only {len(production_df)} rows. "
            f"Recommend at least 50 rows for reliable drift detection."
        )

    # Remove timestamp column (not a feature)
    if "timestamp" in production_df.columns:
        production_df = production_df.drop("timestamp", axis=1)
        logger.info("Removed timestamp column from production data")

    # Remove prediction column (target, not feature)
    if "prediction" in production_df.columns:
        production_df = production_df.drop("prediction", axis=1)
        logger.info("Removed prediction column from production data")

    logger.info(f"Production data shape (processed): {production_df.shape}")

    # Validate columns match
    ref_cols = set(reference_df.columns)
    prod_cols = set(production_df.columns)

    if ref_cols != prod_cols:
        missing_in_prod = ref_cols - prod_cols
        extra_in_prod = prod_cols - ref_cols

        error_msg = "Column mismatch between reference and production data."
        if missing_in_prod:
            error_msg += f"\nMissing in production: {missing_in_prod}"
        if extra_in_prod:
            error_msg += f"\nExtra in production: {extra_in_prod}"

        raise ValueError(error_msg)

    logger.info(f"Column validation passed: {len(ref_cols)} features")

    # Create Evidently report
    logger.info("Creating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset()])

    # Run report
    logger.info("Running drift analysis...")
    report.run(reference_data=reference_df, current_data=production_df)

    # Save HTML report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))

    logger.info(f"Report generated successfully: {output_path}")
    logger.info(f"Open report in browser: open {output_path}")

    # Log summary
    logger.info("\nDrift Analysis Summary:")
    logger.info(f"  Reference samples: {len(reference_df)}")
    logger.info(f"  Production samples: {len(production_df)}")
    logger.info(f"  Features analyzed: {len(ref_cols)}")


if __name__ == "__main__":
    generate_drift_report()
