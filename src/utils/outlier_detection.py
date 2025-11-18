"""
Outlier Detection Utilities

This module provides functions for detecting outliers using various methods:
- IQR (Interquartile Range) method
- Z-score method
"""

import polars as pl


def detect_outliers_iqr(df: pl.DataFrame, column: str) -> dict:
    """
    Detect outliers using IQR method for a single column.

    Args:
        df: Input DataFrame
        column: Column name to analyze

    Returns:
        dict: Outlier statistics including count, percentage, bounds, and outlier values
    """
    # Calculate quartiles
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    # Calculate bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = df.filter((pl.col(column) < lower_bound) | (pl.col(column) > upper_bound))

    outlier_count = len(outliers)
    total_count = len(df)
    outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0

    return {
        "column": column,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_percentage, 2),
        "total_count": total_count,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_values": outliers[column].to_list() if outlier_count > 0 else [],
    }


def analyze_outliers_all_columns(df: pl.DataFrame, dataset_name: str = "Dataset") -> pl.DataFrame:
    """
    Detect outliers using IQR method for all numeric columns.

    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset for logging

    Returns:
        DataFrame with outlier statistics for each numeric column
    """
    from .data_quality import get_numeric_columns

    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        print(f"{dataset_name}: No numeric columns found")
        return pl.DataFrame()

    outlier_results = []
    for col in numeric_cols:
        result = detect_outliers_iqr(df, col)
        outlier_results.append(
            {
                "column": result["column"],
                "outlier_count": result["outlier_count"],
                "outlier_percentage": result["outlier_percentage"],
                "total_count": result["total_count"],
                "lower_bound": round(result["lower_bound"], 2),
                "upper_bound": round(result["upper_bound"], 2),
            }
        )

    outlier_df = pl.DataFrame(outlier_results)
    outlier_df = outlier_df.sort("outlier_count", descending=True)

    print(f"\n{dataset_name} - Outlier Analysis (IQR Method):")
    print(f"  Numeric columns analyzed: {len(numeric_cols)}")
    print(f"  Columns with outliers: {(outlier_df['outlier_count'] > 0).sum()}")
    print(f"  Total outliers detected: {outlier_df['outlier_count'].sum():,}")

    return outlier_df


def detect_outliers_zscore(df: pl.DataFrame, column: str, threshold: float = 3.0) -> dict:
    """
    Detect outliers using Z-score method for a single column.

    Args:
        df: Input DataFrame
        column: Column name to analyze
        threshold: Z-score threshold (default: 3.0)

    Returns:
        dict: Outlier statistics including count, percentage, and outlier values
    """
    # Calculate mean and standard deviation
    mean = df[column].mean()
    std = df[column].std()

    if std == 0:
        return {
            "column": column,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "total_count": len(df),
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "outlier_values": [],
        }

    # Calculate z-scores
    df_with_zscore = df.with_columns([((pl.col(column) - mean) / std).abs().alias("zscore")])

    # Identify outliers
    outliers = df_with_zscore.filter(pl.col("zscore") > threshold)

    outlier_count = len(outliers)
    total_count = len(df)
    outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0

    return {
        "column": column,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_percentage, 2),
        "total_count": total_count,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "outlier_values": outliers[column].to_list() if outlier_count > 0 else [],
    }


def analyze_outliers_zscore_all_columns(
    df: pl.DataFrame, threshold: float = 3.0, dataset_name: str = "Dataset"
) -> pl.DataFrame:
    """
    Detect outliers using Z-score method for all numeric columns.

    Args:
        df: Input DataFrame
        threshold: Z-score threshold (default: 3.0)
        dataset_name: Name of the dataset for logging

    Returns:
        DataFrame with outlier statistics for each numeric column
    """
    from .data_quality import get_numeric_columns

    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        print(f"{dataset_name}: No numeric columns found")
        return pl.DataFrame()

    outlier_results = []
    for col in numeric_cols:
        result = detect_outliers_zscore(df, col, threshold)
        outlier_results.append(
            {
                "column": result["column"],
                "outlier_count": result["outlier_count"],
                "outlier_percentage": result["outlier_percentage"],
                "total_count": result["total_count"],
                "mean": round(result["mean"], 2),
                "std": round(result["std"], 2),
            }
        )

    outlier_df = pl.DataFrame(outlier_results)
    outlier_df = outlier_df.sort("outlier_count", descending=True)

    print(f"\n{dataset_name} - Outlier Analysis (Z-score Method, threshold={threshold}):")
    print(f"  Numeric columns analyzed: {len(numeric_cols)}")
    print(f"  Columns with outliers: {(outlier_df['outlier_count'] > 0).sum()}")
    print(f"  Total outliers detected: {outlier_df['outlier_count'].sum():,}")

    return outlier_df
