"""
Data Quality Analysis Utilities

This module provides functions for analyzing data quality issues including:
- Null value analysis
- Schema comparison
- Type validation
"""

import polars as pl


def compare_schemas(dirty_df: pl.DataFrame, clean_df: pl.DataFrame) -> dict:
    """
    Compare schemas between dirty and clean datasets.

    Args:
        dirty_df: Dirty dataset
        clean_df: Clean dataset

    Returns:
        dict: Comparison results including added, removed, renamed columns and type changes
    """
    dirty_cols = set(dirty_df.columns)
    clean_cols = set(clean_df.columns)

    # Identify column differences
    added_cols = dirty_cols - clean_cols
    removed_cols = clean_cols - dirty_cols
    common_cols = dirty_cols & clean_cols

    # Check for type changes in common columns
    type_changes = {}
    for col in common_cols:
        dirty_type = str(dirty_df[col].dtype)
        clean_type = str(clean_df[col].dtype)
        if dirty_type != clean_type:
            type_changes[col] = {"clean_type": clean_type, "dirty_type": dirty_type}

    return {
        "added_columns": sorted(added_cols),
        "removed_columns": sorted(removed_cols),
        "common_columns": sorted(common_cols),
        "type_changes": type_changes,
        "dirty_shape": dirty_df.shape,
        "clean_shape": clean_df.shape,
    }


def analyze_nulls(df: pl.DataFrame, dataset_name: str = "Dataset") -> pl.DataFrame:
    """
    Analyze null values in a dataset.

    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset for logging

    Returns:
        DataFrame with columns: [column, null_count, null_percentage, total_rows]
    """
    total_rows = df.shape[0]

    # Handle empty DataFrame
    if total_rows == 0:
        print(f"\n⚠️ WARNING: {dataset_name} is empty (0 rows)!")
        return pl.DataFrame(
            {
                "column": df.columns,
                "null_count": [0] * len(df.columns),
                "null_percentage": [0.0] * len(df.columns),
                "total_rows": [0] * len(df.columns),
            }
        )

    null_counts = []
    for col in df.columns:
        null_count = df[col].null_count()
        null_percentage = (null_count / total_rows) * 100
        null_counts.append(
            {
                "column": col,
                "null_count": null_count,
                "null_percentage": round(null_percentage, 2),
                "total_rows": total_rows,
            }
        )

    null_df = pl.DataFrame(null_counts)

    # Sort by null_count descending
    null_df = null_df.sort("null_count", descending=True)

    print(f"\n{dataset_name} - Null Analysis:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Columns with nulls: {(null_df['null_count'] > 0).sum()}/{len(df.columns)}")
    print(f"  Total null values: {null_df['null_count'].sum():,}")

    return null_df


def compare_null_patterns(dirty_nulls: pl.DataFrame, clean_nulls: pl.DataFrame) -> pl.DataFrame:
    """
    Compare null patterns between dirty and clean datasets.

    Args:
        dirty_nulls: Null analysis results from dirty dataset
        clean_nulls: Null analysis results from clean dataset

    Returns:
        DataFrame comparing null patterns
    """
    comparison = dirty_nulls.join(clean_nulls, on="column", how="outer", suffix="_clean")

    # Rename columns for clarity
    comparison = comparison.rename(
        {
            "null_count": "dirty_null_count",
            "null_percentage": "dirty_null_pct",
            "null_count_clean": "clean_null_count",
            "null_percentage_clean": "clean_null_pct",
        }
    )

    # Fill nulls with 0 for columns that don't exist in one dataset
    comparison = comparison.with_columns(
        [
            pl.col("dirty_null_count").fill_null(0),
            pl.col("dirty_null_pct").fill_null(0.0),
            pl.col("clean_null_count").fill_null(0),
            pl.col("clean_null_pct").fill_null(0.0),
        ]
    )

    # Calculate difference
    comparison = comparison.with_columns(
        [
            (pl.col("dirty_null_count") - pl.col("clean_null_count")).alias("null_count_diff"),
            (pl.col("dirty_null_pct") - pl.col("clean_null_pct")).alias("null_pct_diff"),
        ]
    )

    # Sort by absolute difference
    comparison = (
        comparison.with_columns(pl.col("null_count_diff").abs().alias("abs_diff"))
        .sort("abs_diff", descending=True)
        .drop("abs_diff")
    )

    return comparison


def validate_types(df: pl.DataFrame, expected_schema: dict, dataset_name: str = "Dataset") -> dict:
    """
    Validate column types against expected schema.

    Args:
        df: Input DataFrame
        expected_schema: Dictionary mapping column names to expected Polars dtypes
        dataset_name: Name of the dataset for logging

    Returns:
        dict: Validation results with type mismatches and examples
    """
    validation_results = {
        "valid_columns": [],
        "invalid_columns": [],
        "missing_columns": [],
        "extra_columns": [],
    }

    df_columns = set(df.columns)
    expected_columns = set(expected_schema.keys())

    # Find missing and extra columns
    validation_results["missing_columns"] = sorted(expected_columns - df_columns)
    validation_results["extra_columns"] = sorted(df_columns - expected_columns)

    # Validate types for common columns
    for col in expected_columns & df_columns:
        expected_type = expected_schema[col]
        actual_type = df[col].dtype

        if str(actual_type) == str(expected_type):
            validation_results["valid_columns"].append(col)
        else:
            validation_results["invalid_columns"].append(
                {
                    "column": col,
                    "expected_type": str(expected_type),
                    "actual_type": str(actual_type),
                }
            )

    print(f"\n{dataset_name} - Type Validation:")
    print(f"  Valid columns: {len(validation_results['valid_columns'])}")
    print(f"  Invalid columns: {len(validation_results['invalid_columns'])}")
    print(f"  Missing columns: {len(validation_results['missing_columns'])}")
    print(f"  Extra columns: {len(validation_results['extra_columns'])}")

    return validation_results


def compare_type_validation(dirty_validation: dict, clean_validation: dict) -> pl.DataFrame:
    """
    Compare type validation results between dirty and clean datasets.

    Args:
        dirty_validation: Validation results from dirty dataset
        clean_validation: Validation results from clean dataset

    Returns:
        DataFrame comparing type validation results
    """
    comparison_data = []

    # Get all columns from both validations
    all_columns = set()
    for val in [dirty_validation, clean_validation]:
        all_columns.update(val["valid_columns"])
        all_columns.update([item["column"] for item in val["invalid_columns"]])

    for col in sorted(all_columns):
        # Check dirty dataset
        dirty_valid = col in dirty_validation["valid_columns"]
        dirty_invalid = any(item["column"] == col for item in dirty_validation["invalid_columns"])

        # Check clean dataset
        clean_valid = col in clean_validation["valid_columns"]
        any(item["column"] == col for item in clean_validation["invalid_columns"])

        comparison_data.append(
            {
                "column": col,
                "dirty_valid": dirty_valid,
                "clean_valid": clean_valid,
                "issue_introduced": clean_valid and dirty_invalid,
            }
        )

    return pl.DataFrame(comparison_data)


def get_numeric_columns(df: pl.DataFrame) -> list[str]:
    """
    Get list of numeric columns from a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of numeric column names
    """
    numeric_types = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]

    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_types]

    return numeric_cols
