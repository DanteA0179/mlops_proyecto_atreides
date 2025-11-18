"""
Data Cleaning Utilities

This module provides functions for cleaning and transforming data:
- Type conversion
- Null value handling
- Range correction
- Outlier treatment
- Duplicate removal
"""

import polars as pl


def convert_data_types(
    df: pl.DataFrame, schema_target: dict[str, pl.DataType], drop_columns: list[str] | None = None
) -> pl.DataFrame:
    """
    Convert columns to target data types.

    Args:
        df: Input DataFrame
        schema_target: Dictionary mapping column names to target Polars dtypes
        drop_columns: List of columns to drop (e.g., mixed type columns)

    Returns:
        DataFrame with converted types
    """
    # Drop unwanted columns
    if drop_columns:
        for col in drop_columns:
            if col in df.columns:
                df = df.drop(col)

    # Convert each column to target type
    for col, dtype in schema_target.items():
        if col not in df.columns:
            continue

        # Skip if already correct type
        if df[col].dtype == dtype:
            continue

        # For string to numeric conversion, clean whitespace first
        if df[col].dtype == pl.Utf8 and dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            df = df.with_columns(pl.col(col).str.strip_chars().alias(col))

        # Special handling for Int64: convert via Float64 first to handle decimal strings
        if df[col].dtype == pl.Utf8 and dtype in [pl.Int64, pl.Int32]:
            df = df.with_columns(
                pl.col(col).cast(pl.Float64, strict=False).cast(dtype, strict=False).alias(col)
            )
        else:
            # Convert with strict=False to allow nulls for invalid values
            df = df.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))

    return df


def handle_null_values(
    df: pl.DataFrame,
    null_threshold: float = 0.3,
    interpolate_cols: list[str] | None = None,
    forward_fill_cols: list[str] | None = None,
    backward_fill_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Handle null values using various strategies.

    Args:
        df: Input DataFrame
        null_threshold: Remove rows with more than this fraction of nulls
        interpolate_cols: Columns to interpolate
        forward_fill_cols: Columns to forward fill
        backward_fill_cols: Columns to backward fill

    Returns:
        DataFrame with handled nulls
    """
    # Remove rows with too many nulls
    total_cols = len(df.columns)
    df = df.filter(
        pl.sum_horizontal([pl.col(c).is_null() for c in df.columns])
        <= (total_cols * null_threshold)
    )

    # Interpolate specified columns
    if interpolate_cols:
        for col in interpolate_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).interpolate().alias(col))

    # Forward fill specified columns
    if forward_fill_cols:
        for col in forward_fill_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).forward_fill().alias(col))

    # Backward fill specified columns
    if backward_fill_cols:
        for col in backward_fill_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).backward_fill().alias(col))

    return df


def correct_range_violations(
    df: pl.DataFrame,
    range_rules: dict[str, dict[str, float | None]],
    make_absolute: list[str] | None = None,
) -> pl.DataFrame:
    """
    Correct values that violate range constraints.

    Args:
        df: Input DataFrame
        range_rules: Dictionary mapping column names to {'min': value, 'max': value}
        make_absolute: List of columns to convert negative values to absolute

    Returns:
        DataFrame with corrected ranges
    """
    # Convert negative values to absolute for specified columns
    if make_absolute:
        for col in make_absolute:
            if col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col) < 0)
                    .then(pl.col(col).abs())
                    .otherwise(pl.col(col))
                    .alias(col)
                )

    # Apply range rules - CAP values instead of setting to null
    for col, rules in range_rules.items():
        if col not in df.columns:
            continue

        min_val = rules.get("min")
        max_val = rules.get("max")

        # Cap values at min/max instead of setting to null
        if min_val is not None and max_val is not None:
            df = df.with_columns(
                pl.when(pl.col(col) < min_val)
                .then(min_val)
                .when(pl.col(col) > max_val)
                .then(max_val)
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif min_val is not None:
            df = df.with_columns(
                pl.when(pl.col(col) < min_val).then(min_val).otherwise(pl.col(col)).alias(col)
            )
        elif max_val is not None:
            df = df.with_columns(
                pl.when(pl.col(col) > max_val).then(max_val).otherwise(pl.col(col)).alias(col)
            )

    return df


def treat_outliers(
    df: pl.DataFrame,
    columns: list[str],
    method: str = "cap",
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pl.DataFrame:
    """
    Treat outliers using capping or removal.

    Args:
        df: Input DataFrame
        columns: List of columns to treat
        method: 'cap' to cap values, 'remove' to set to null
        lower_percentile: Lower percentile for capping (default 1%)
        upper_percentile: Upper percentile for capping (default 99%)

    Returns:
        DataFrame with treated outliers
    """
    for col in columns:
        if col not in df.columns:
            continue

        # Calculate percentiles
        p_lower = df[col].quantile(lower_percentile)
        p_upper = df[col].quantile(upper_percentile)

        if method == "cap":
            # Cap values at percentiles
            df = df.with_columns(
                pl.when(pl.col(col) < p_lower)
                .then(p_lower)
                .when(pl.col(col) > p_upper)
                .then(p_upper)
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif method == "remove":
            # Set outliers to null
            df = df.with_columns(
                pl.when((pl.col(col) < p_lower) | (pl.col(col) > p_upper))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    return df


def remove_duplicates(
    df: pl.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> pl.DataFrame:
    """
    Remove duplicate rows.

    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates (None = all columns)
        keep: Which duplicates to keep ('first', 'last', or 'none')

    Returns:
        DataFrame with duplicates removed
    """
    if keep == "first":
        df = df.unique(subset=subset, maintain_order=True)
    elif keep == "last":
        df = df.unique(subset=subset, maintain_order=True, keep="last")
    elif keep == "none":
        # Remove all duplicates including first occurrence
        if subset:
            duplicated = df.select(subset).is_duplicated()
        else:
            duplicated = df.is_duplicated()
        df = df.filter(~duplicated)

    return df


def validate_cleaned_data(
    df: pl.DataFrame, reference_df: pl.DataFrame, tolerance: int = 100
) -> dict:
    """
    Validate cleaned dataset against reference.

    Args:
        df: Cleaned DataFrame
        reference_df: Reference DataFrame
        tolerance: Allowed difference in row count

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "shape_match": abs(len(df) - len(reference_df)) <= tolerance,
        "row_count_diff": len(df) - len(reference_df),
        "schema_match": set(df.columns) == set(reference_df.columns),
        "null_count": df.null_count().sum_horizontal()[0],
        "duplicate_count": len(df) - len(df.unique()),
    }

    # Check type mismatches
    type_mismatches = []
    for col in df.columns:
        if col in reference_df.columns:
            if str(df[col].dtype) != str(reference_df[col].dtype):
                type_mismatches.append(
                    {
                        "column": col,
                        "cleaned_type": str(df[col].dtype),
                        "reference_type": str(reference_df[col].dtype),
                    }
                )

    validation_results["type_mismatches"] = type_mismatches
    validation_results["all_checks_passed"] = (
        validation_results["shape_match"]
        and validation_results["schema_match"]
        and validation_results["null_count"] == 0
        and validation_results["duplicate_count"] == 0
        and len(type_mismatches) == 0
    )

    return validation_results
