"""
Data preparation utilities for Chronos-2 fine-tuning with covariates.

This module ensures that future_covariates are a subset of past_covariates
as required by Chronos-2.
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def prepare_chronos_finetuning_data_with_covariates(
    df: pl.DataFrame,
    target_col: str = "Usage_kWh",
    series_id_col: str | None = None,
    past_covariates: list[str] | None = None,
    future_covariates: list[str] | None = None,
) -> list[dict]:
    """
    Prepare data for Chronos-2 fine-tuning with covariates.

    IMPORTANT: Chronos-2 requires future_covariates to be a subset of past_covariates.
    This function validates this constraint.

    Chronos-2 expects data in format:
    [{
        "target": np.array([...]),           # Required
        "past_covariates": {...},            # Optional
        "future_covariates": {...}           # Optional (must be subset of past)
    }]

    Args:
        df: DataFrame with time series data
        target_col: Target column name
        series_id_col: Column to group by for multiple series (optional)
        past_covariates: List of column names for past covariates
        future_covariates: List of column names for future covariates

    Returns:
        List of dicts ready for Chronos-2 fine-tuning

    Raises:
        ValueError: If future_covariates is not a subset of past_covariates

    Examples:
        >>> df = pl.read_parquet("data/processed/steel_preprocessed_train.parquet")
        >>> past_covs = ["temp", "humidity", "day_of_week", "hour"]
        >>> future_covs = ["day_of_week", "hour"]  # Subset of past
        >>> train_inputs = prepare_chronos_finetuning_data_with_covariates(
        ...     df,
        ...     target_col="Usage_kWh",
        ...     past_covariates=past_covs,
        ...     future_covariates=future_covs
        ... )
    """
    logger.info(f"Preparing data for fine-tuning: {len(df)} samples")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Validate covariates constraint
    if past_covariates and future_covariates:
        future_set = set(future_covariates)
        past_set = set(past_covariates)

        if not future_set.issubset(past_set):
            extra_vars = future_set - past_set
            raise ValueError(
                f"Chronos-2 constraint violation: future_covariates must be a subset of past_covariates.\n"
                f"Variables in future but not in past: {extra_vars}\n"
                f"Past covariates: {past_covariates}\n"
                f"Future covariates: {future_covariates}\n"
                f"Solution: Add {extra_vars} to past_covariates or remove from future_covariates"
            )

        logger.info("✓ Covariates constraint validated")

    train_inputs = []

    # If no series_id_col, treat entire dataset as single series
    if series_id_col is None or series_id_col not in df.columns:
        logger.info("Treating data as single time series")

        input_dict = {
            "target": df[target_col].to_numpy().astype(np.float32),
        }

        # Add past covariates if specified
        if past_covariates:
            past_cov_dict = {}
            for col in past_covariates:
                if col in df.columns:
                    past_cov_dict[col] = df[col].to_numpy().astype(np.float32)
                else:
                    logger.warning(f"Past covariate '{col}' not found, skipping")

            if past_cov_dict:
                input_dict["past_covariates"] = past_cov_dict
                logger.info(
                    f"Added {len(past_cov_dict)} past covariates: {list(past_cov_dict.keys())}"
                )

        # Add future covariates if specified
        if future_covariates:
            future_cov_dict = {}
            for col in future_covariates:
                if col in df.columns:
                    future_cov_dict[col] = df[col].to_numpy().astype(np.float32)
                else:
                    logger.warning(f"Future covariate '{col}' not found, skipping")

            if future_cov_dict:
                input_dict["future_covariates"] = future_cov_dict
                logger.info(
                    f"Added {len(future_cov_dict)} future covariates: {list(future_cov_dict.keys())}"
                )

        train_inputs.append(input_dict)

    else:
        # Multiple series
        logger.info(f"Grouping by '{series_id_col}'")

        for _series_id, group in df.group_by(series_id_col):
            input_dict = {
                "target": group[target_col].to_numpy().astype(np.float32),
            }

            # Add covariates (same logic as above)
            if past_covariates:
                past_cov_dict = {}
                for col in past_covariates:
                    if col in group.columns:
                        past_cov_dict[col] = group[col].to_numpy().astype(np.float32)
                if past_cov_dict:
                    input_dict["past_covariates"] = past_cov_dict

            if future_covariates:
                future_cov_dict = {}
                for col in future_covariates:
                    if col in group.columns:
                        future_cov_dict[col] = group[col].to_numpy().astype(np.float32)
                if future_cov_dict:
                    input_dict["future_covariates"] = future_cov_dict

            train_inputs.append(input_dict)

        logger.info(f"Created {len(train_inputs)} series")

    # Validate
    for idx, input_dict in enumerate(train_inputs):
        if len(input_dict["target"]) == 0:
            raise ValueError(f"Series {idx} has empty target")

    logger.info(f"Prepared {len(train_inputs)} training inputs")

    return train_inputs


def validate_finetuning_data(train_inputs: list[dict]) -> dict:
    """
    Validate fine-tuning data format and return statistics.

    Args:
        train_inputs: List of input dicts

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "n_series": len(train_inputs),
        "min_length": min(len(inp["target"]) for inp in train_inputs),
        "max_length": max(len(inp["target"]) for inp in train_inputs),
        "avg_length": np.mean([len(inp["target"]) for inp in train_inputs]),
        "has_past_covariates": any("past_covariates" in inp for inp in train_inputs),
        "has_future_covariates": any("future_covariates" in inp for inp in train_inputs),
    }

    # Get covariate names if available
    if stats["has_past_covariates"]:
        first_with_past = next(inp for inp in train_inputs if "past_covariates" in inp)
        stats["past_covariate_names"] = list(first_with_past["past_covariates"].keys())

    if stats["has_future_covariates"]:
        first_with_future = next(inp for inp in train_inputs if "future_covariates" in inp)
        stats["future_covariate_names"] = list(first_with_future["future_covariates"].keys())

    logger.info(f"Validation stats: {stats}")

    return stats
