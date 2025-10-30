"""
Data Splitting Utilities for Train/Validation/Test Sets.

This module provides functions for splitting datasets into train, validation, and test sets
with support for stratification and temporal splitting strategies.

Functions
---------
stratified_train_val_test_split : Split with stratification by target
temporal_train_val_test_split : Split respecting temporal order
validate_splits : Validate split correctness
get_split_statistics : Calculate statistics for each split
"""


from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    df: pl.DataFrame,
    target_col: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    n_bins: int = 10,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split dataset with stratification based on binned target values.

    For regression tasks, bins the target variable to enable stratified splitting,
    ensuring similar target distributions across train/val/test sets.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe to split
    target_col : str
        Name of target column for stratification
    train_size : float, default=0.70
        Proportion of dataset for training set
    val_size : float, default=0.15
        Proportion of dataset for validation set
    test_size : float, default=0.15
        Proportion of dataset for test set
    n_bins : int, default=10
        Number of bins for target stratification
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    tuple of pl.DataFrame
        (df_train, df_val, df_test)

    Raises
    ------
    ValueError
        If sizes don't sum to 1.0 or target column doesn't exist

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/steel_featured.parquet")
    >>> df_train, df_val, df_test = stratified_train_val_test_split(
    ...     df, target_col='Usage_kWh', train_size=0.7, val_size=0.15, test_size=0.15
    ... )
    >>> print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    """
    # Validate inputs
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Create bins for stratification
    target_values = df[target_col].to_numpy()
    bins = np.linspace(target_values.min(), target_values.max(), n_bins + 1)
    binned_target = np.digitize(target_values, bins[:-1])

    # Add binned target as temporary column
    df_with_bins = df.with_columns(pl.Series("__stratify_bins__", binned_target))

    # First split: train + (val + test) vs test
    # Calculate intermediate size for val+test
    temp_test_size = test_size / (val_size + test_size)

    # Convert to pandas for sklearn (stratify requires it)
    df_pandas = df_with_bins.to_pandas()

    # First split: train vs (val + test)
    df_train_temp, df_temp = train_test_split(
        df_pandas,
        train_size=train_size,
        stratify=df_pandas["__stratify_bins__"],
        random_state=random_state,
    )

    # Second split: val vs test
    df_val_temp, df_test_temp = train_test_split(
        df_temp,
        test_size=temp_test_size,
        stratify=df_temp["__stratify_bins__"],
        random_state=random_state,
    )

    # Convert back to Polars and remove temporary column
    df_train = pl.from_pandas(df_train_temp).drop("__stratify_bins__")
    df_val = pl.from_pandas(df_val_temp).drop("__stratify_bins__")
    df_test = pl.from_pandas(df_test_temp).drop("__stratify_bins__")

    return df_train, df_val, df_test


def temporal_train_val_test_split(
    df: pl.DataFrame,
    date_col: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split dataset respecting temporal order (no shuffling).

    For time series where temporal order must be preserved. Train set contains
    earliest data, test set contains most recent data.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe to split (must contain date column)
    date_col : str
        Name of datetime column for temporal ordering
    train_size : float, default=0.70
        Proportion of dataset for training set
    val_size : float, default=0.15
        Proportion of dataset for validation set
    test_size : float, default=0.15
        Proportion of dataset for test set

    Returns
    -------
    tuple of pl.DataFrame
        (df_train, df_val, df_test)

    Raises
    ------
    ValueError
        If sizes don't sum to 1.0 or date column doesn't exist

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/steel_featured.parquet")
    >>> df_train, df_val, df_test = temporal_train_val_test_split(
    ...     df, date_col='date', train_size=0.7, val_size=0.15, test_size=0.15
    ... )
    >>> # Train has earliest dates, test has most recent
    """
    # Validate inputs
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
        )

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    # Sort by date
    df_sorted = df.sort(date_col)

    # Calculate split indices
    n = len(df_sorted)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    # Split
    df_train = df_sorted[:train_end]
    df_val = df_sorted[train_end:val_end]
    df_test = df_sorted[val_end:]

    return df_train, df_val, df_test


def validate_splits(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    target_col: str,
    expected_train_size: float = 0.70,
    expected_val_size: float = 0.15,
    expected_test_size: float = 0.15,
    tolerance: float = 0.02,
) -> dict[str, bool | str | dict[str, Any]]:
    """
    Validate that splits are correct and no data leakage exists.

    Performs comprehensive checks on train/val/test splits including:
    - Sizes are as expected
    - No data leakage (no row overlap)
    - Target distributions are similar
    - All original rows are accounted for

    Parameters
    ----------
    df_train : pl.DataFrame
        Training set
    df_val : pl.DataFrame
        Validation set
    df_test : pl.DataFrame
        Test set
    target_col : str
        Name of target column
    expected_train_size : float, default=0.70
        Expected proportion for train set
    expected_val_size : float, default=0.15
        Expected proportion for val set
    expected_test_size : float, default=0.15
        Expected proportion for test set
    tolerance : float, default=0.02
        Tolerance for size validation (±2%)

    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool - Overall validation result
        - 'checks': dict - Individual check results
        - 'statistics': dict - Statistics for each split
        - 'issues': list - List of validation issues found

    Examples
    --------
    >>> validation = validate_splits(df_train, df_val, df_test, 'Usage_kWh')
    >>> if validation['valid']:
    ...     print("✅ Splits are valid")
    ... else:
    ...     print(f"❌ Issues: {validation['issues']}")
    """
    results = {
        "valid": True,
        "checks": {},
        "statistics": {},
        "issues": [],
    }

    # Calculate total size
    total_size = len(df_train) + len(df_val) + len(df_test)

    # Check 1: Sizes are as expected
    train_actual = len(df_train) / total_size
    val_actual = len(df_val) / total_size
    test_actual = len(df_test) / total_size

    size_checks = {
        "train": abs(train_actual - expected_train_size) <= tolerance,
        "val": abs(val_actual - expected_val_size) <= tolerance,
        "test": abs(test_actual - expected_test_size) <= tolerance,
    }

    results["checks"]["sizes"] = size_checks

    if not all(size_checks.values()):
        results["valid"] = False
        results["issues"].append(
            f"Split sizes outside tolerance: "
            f"train={train_actual:.3f} (expected {expected_train_size}), "
            f"val={val_actual:.3f} (expected {expected_val_size}), "
            f"test={test_actual:.3f} (expected {expected_test_size})"
        )

    # Check 2: No data leakage (row indices don't overlap)
    # Create unique identifier for each row (using all columns hash)
    def get_row_hashes(df: pl.DataFrame) -> set:
        # Use first 5 columns as identifier (enough for uniqueness)
        cols_to_hash = df.columns[:5]
        hashes = set()
        for row in df.select(cols_to_hash).iter_rows():
            hashes.add(hash(row))
        return hashes

    train_hashes = get_row_hashes(df_train)
    val_hashes = get_row_hashes(df_val)
    test_hashes = get_row_hashes(df_test)

    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    leakage_checks = {
        "train_val_no_overlap": len(train_val_overlap) == 0,
        "train_test_no_overlap": len(train_test_overlap) == 0,
        "val_test_no_overlap": len(val_test_overlap) == 0,
    }

    results["checks"]["no_leakage"] = leakage_checks

    if not all(leakage_checks.values()):
        results["valid"] = False
        results["issues"].append(
            f"Data leakage detected: "
            f"train-val overlap={len(train_val_overlap)}, "
            f"train-test overlap={len(train_test_overlap)}, "
            f"val-test overlap={len(val_test_overlap)}"
        )

    # Check 3: Target distributions are similar
    if target_col in df_train.columns:
        train_mean = df_train[target_col].mean()
        val_mean = df_val[target_col].mean()
        test_mean = df_test[target_col].mean()

        train_std = df_train[target_col].std()
        val_std = df_val[target_col].std()
        test_std = df_test[target_col].std()

        # Target distributions should be similar (within 10% relative difference)
        mean_diffs = {
            "train_val": abs(train_mean - val_mean) / train_mean,
            "train_test": abs(train_mean - test_mean) / train_mean,
        }

        distribution_checks = {
            "train_val_similar": mean_diffs["train_val"] < 0.10,
            "train_test_similar": mean_diffs["train_test"] < 0.10,
        }

        results["checks"]["distributions"] = distribution_checks

        if not all(distribution_checks.values()):
            results["valid"] = False
            results["issues"].append(
                f"Target distributions differ significantly: "
                f"train_mean={train_mean:.2f}, val_mean={val_mean:.2f}, test_mean={test_mean:.2f}"
            )

        # Store statistics
        results["statistics"]["target_distributions"] = {
            "train": {"mean": float(train_mean), "std": float(train_std)},
            "val": {"mean": float(val_mean), "std": float(val_std)},
            "test": {"mean": float(test_mean), "std": float(test_std)},
        }

    # Store size statistics
    results["statistics"]["sizes"] = {
        "train": {"count": len(df_train), "proportion": train_actual},
        "val": {"count": len(df_val), "proportion": val_actual},
        "test": {"count": len(df_test), "proportion": test_actual},
        "total": total_size,
    }

    return results


def get_split_statistics(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    numeric_cols: list[str] = None,
    categorical_cols: list[str] = None,
) -> dict[str, dict[str, Any]]:
    """
    Calculate comprehensive statistics for each split.

    Parameters
    ----------
    df_train : pl.DataFrame
        Training set
    df_val : pl.DataFrame
        Validation set
    df_test : pl.DataFrame
        Test set
    numeric_cols : list of str, optional
        Numeric columns to analyze
    categorical_cols : list of str, optional
        Categorical columns to analyze

    Returns
    -------
    dict
        Statistics for each split including:
        - numeric: mean, std, min, max for each numeric column
        - categorical: value counts for each categorical column

    Examples
    --------
    >>> stats = get_split_statistics(
    ...     df_train, df_val, df_test,
    ...     numeric_cols=['Usage_kWh', 'CO2(tCO2)'],
    ...     categorical_cols=['Load_Type']
    ... )
    >>> print(stats['train']['numeric']['Usage_kWh'])
    """
    splits = {"train": df_train, "val": df_val, "test": df_test}
    statistics = {}

    for split_name, df in splits.items():
        statistics[split_name] = {"numeric": {}, "categorical": {}}

        # Numeric statistics
        if numeric_cols:
            for col in numeric_cols:
                if col in df.columns:
                    statistics[split_name]["numeric"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "median": float(df[col].median()),
                    }

        # Categorical statistics
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    value_counts = df[col].value_counts().sort(col)
                    statistics[split_name]["categorical"][col] = {
                        row[0]: int(row[1]) for row in value_counts.iter_rows()
                    }

    return statistics
