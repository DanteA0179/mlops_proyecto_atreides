"""
Temporal Feature Engineering Utilities

This module provides functions for creating temporal features from time-based columns,
specifically designed for the Steel Industry Energy Consumption dataset. All functions
use Polars for high-performance data manipulation.

The module supports:
- Extraction of hour from NSM (Number of Seconds from Midnight)
- Conversion of day names to numeric day_of_week
- Weekend detection
- Cyclical encoding for periodic features (hour, day_of_week)
- Validation of temporal features

Example Usage:
    >>> import polars as pl
    >>> from src.utils.temporal_features import (
    ...     extract_hour_from_nsm,
    ...     create_cyclical_encoding,
    ...     create_all_temporal_features
    ... )
    >>>
    >>> # Load data
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>>
    >>> # Create all temporal features
    >>> df_featured = create_all_temporal_features(df)
    >>> print(df_featured.columns)
    >>> # ['..existing..',

 'hour', 'day_of_week', 'is_weekend',
    >>> #  'cyclical_hour_sin', 'cyclical_hour_cos',
    >>> #  'cyclical_day_sin', 'cyclical_day_cos']

Requirements:
    - polars >= 0.20.0
    - numpy >= 1.26.0
"""


import numpy as np
import polars as pl

# Module constants
DAY_NAME_TO_NUMBER = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

WEEKEND_DAYS = [5, 6]  # Saturday=5, Sunday=6


def extract_hour_from_nsm(
    df: pl.DataFrame, nsm_col: str = "NSM", output_col: str = "hour"
) -> pl.DataFrame:
    """
    Extract hour (0-23) from NSM (Number of Seconds from Midnight) column.

    NSM represents the number of seconds elapsed since midnight (00:00:00).
    This function converts NSM to hours by integer division by 3600.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing NSM column
    nsm_col : str, default='NSM'
        Name of the NSM column
    output_col : str, default='hour'
        Name of the output hour column

    Returns
    -------
    pl.DataFrame
        DataFrame with new hour column (0-23)

    Raises
    ------
    ValueError
        If nsm_col does not exist in dataframe
        If NSM contains values outside valid range [0, 86400]

    Examples
    --------
    >>> df = pl.DataFrame({'NSM': [0, 3600, 7200, 43200, 86399]})
    >>> df_with_hour = extract_hour_from_nsm(df)
    >>> df_with_hour['hour'].to_list()
    [0, 1, 2, 12, 23]
    """
    if nsm_col not in df.columns:
        raise ValueError(f"Column '{nsm_col}' not found in dataframe")

    # Validate NSM range
    nsm_min = df[nsm_col].min()
    nsm_max = df[nsm_col].max()

    if nsm_min < 0 or nsm_max > 86400:
        raise ValueError(
            f"NSM values out of valid range [0, 86400]. " f"Found: min={nsm_min}, max={nsm_max}"
        )

    # Extract hour using integer division
    # Special case: NSM=86400 (end of day) should be hour 23, not 24
    df = df.with_columns(
        pl.when(pl.col(nsm_col) >= 86400)
        .then(23)
        .otherwise(pl.col(nsm_col) // 3600)
        .cast(pl.Int32)
        .alias(output_col)
    )

    return df


def extract_day_of_week_numeric(
    df: pl.DataFrame, day_col: str = "Day_of_week", output_col: str = "day_of_week"
) -> pl.DataFrame:
    """
    Convert day name to numeric day_of_week (0-6).

    Mapping:
    - Monday = 0
    - Tuesday = 1
    - Wednesday = 2
    - Thursday = 3
    - Friday = 4
    - Saturday = 5
    - Sunday = 6

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing day name column
    day_col : str, default='Day_of_week'
        Name of the column containing day names
    output_col : str, default='day_of_week'
        Name of the output numeric day column

    Returns
    -------
    pl.DataFrame
        DataFrame with new numeric day_of_week column (0-6)

    Raises
    ------
    ValueError
        If day_col does not exist in dataframe
        If day_col contains invalid day names

    Examples
    --------
    >>> df = pl.DataFrame({'Day_of_week': ['Monday', 'Friday', 'Sunday']})
    >>> df_with_num = extract_day_of_week_numeric(df)
    >>> df_with_num['day_of_week'].to_list()
    [0, 4, 6]
    """
    if day_col not in df.columns:
        raise ValueError(f"Column '{day_col}' not found in dataframe")

    # Get unique days and validate
    unique_days = df[day_col].unique().to_list()
    invalid_days = [d for d in unique_days if d not in DAY_NAME_TO_NUMBER]

    if invalid_days:
        raise ValueError(
            f"Invalid day names found: {invalid_days}. "
            f"Valid days: {list(DAY_NAME_TO_NUMBER.keys())}"
        )

    # Create mapping expression using pl.when().then().otherwise() chain
    mapping_expr = pl.col(day_col)

    # Build the mapping using when-then-otherwise
    for day_name, day_num in DAY_NAME_TO_NUMBER.items():
        if day_name == "Monday":  # First condition
            mapping_expr = pl.when(pl.col(day_col) == day_name).then(day_num)
        else:
            mapping_expr = mapping_expr.when(pl.col(day_col) == day_name).then(day_num)

    # Add the mapping column
    df = df.with_columns(mapping_expr.cast(pl.Int32).alias(output_col))

    return df


def create_is_weekend(
    df: pl.DataFrame, day_num_col: str = "day_of_week", output_col: str = "is_weekend"
) -> pl.DataFrame:
    """
    Create boolean weekend indicator from numeric day_of_week.

    Saturday (5) and Sunday (6) are considered weekend days.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing numeric day_of_week column
    day_num_col : str, default='day_of_week'
        Name of the numeric day column (0-6)
    output_col : str, default='is_weekend'
        Name of the output boolean column

    Returns
    -------
    pl.DataFrame
        DataFrame with new boolean is_weekend column

    Raises
    ------
    ValueError
        If day_num_col does not exist in dataframe
        If day_num_col contains values outside [0, 6]

    Examples
    --------
    >>> df = pl.DataFrame({'day_of_week': [0, 4, 5, 6]})  # Mon, Fri, Sat, Sun
    >>> df_with_weekend = create_is_weekend(df)
    >>> df_with_weekend['is_weekend'].to_list()
    [False, False, True, True]
    """
    if day_num_col not in df.columns:
        raise ValueError(f"Column '{day_num_col}' not found in dataframe")

    # Validate day_of_week range
    day_min = df[day_num_col].min()
    day_max = df[day_num_col].max()

    if day_min < 0 or day_max > 6:
        raise ValueError(
            f"day_of_week values out of valid range [0, 6]. " f"Found: min={day_min}, max={day_max}"
        )

    # Create weekend indicator (Saturday=5, Sunday=6)
    df = df.with_columns((pl.col(day_num_col) >= 5).alias(output_col))

    return df


def create_cyclical_encoding(
    df: pl.DataFrame, col: str, period: int, sin_col: str = None, cos_col: str = None
) -> pl.DataFrame:
    """
    Create cyclical sin/cos encoding for periodic features.

    Cyclical encoding preserves the periodic nature of time-based features:
    - Hour 23 and Hour 0 are close (end/start of day)
    - Day 6 (Sunday) and Day 0 (Monday) are close (end/start of week)

    Formulas:
    - sin_feature = sin(2π * value / period)
    - cos_feature = cos(2π * value / period)

    Properties:
    - Values are bounded: [-1, 1]
    - Continuity: sin²(x) + cos²(x) = 1
    - Proximity: Similar values have similar encodings

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing the column to encode
    col : str
        Name of the column to encode cyclically
    period : int
        Period of the cyclic feature (24 for hours, 7 for days)
    sin_col : str, optional
        Name for sine column. Default: 'cyclical_{col}_sin'
    cos_col : str, optional
        Name for cosine column. Default: 'cyclical_{col}_cos'

    Returns
    -------
    pl.DataFrame
        DataFrame with new sin and cos columns

    Raises
    ------
    ValueError
        If col does not exist in dataframe
        If period <= 0

    Examples
    --------
    >>> df = pl.DataFrame({'hour': [0, 6, 12, 18, 23]})
    >>> df_cyclical = create_cyclical_encoding(df, 'hour', period=24)
    >>> df_cyclical.columns
    ['hour', 'cyclical_hour_sin', 'cyclical_hour_cos']

    >>> # Verify continuity: hour 23 ≈ hour 0
    >>> df_check = pl.DataFrame({'hour': [0, 23]})
    >>> df_check = create_cyclical_encoding(df_check, 'hour', 24)
    >>> # cos(0) ≈ cos(23) (both close to 1.0)
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe")

    if period <= 0:
        raise ValueError(f"Period must be positive, got: {period}")

    # Default column names
    if sin_col is None:
        sin_col = f"cyclical_{col}_sin"
    if cos_col is None:
        cos_col = f"cyclical_{col}_cos"

    # Create cyclical features using numpy trigonometric functions
    # Formula: 2π * value / period
    df = df.with_columns(
        [
            (pl.col(col) * 2 * np.pi / period).sin().alias(sin_col),
            (pl.col(col) * 2 * np.pi / period).cos().alias(cos_col),
        ]
    )

    return df


def create_all_temporal_features(
    df: pl.DataFrame, nsm_col: str = "NSM", day_name_col: str = "Day_of_week"
) -> pl.DataFrame:
    """
    Create all temporal features required for US-011 in one call.

    Creates 7 new features:
    1. hour (0-23): Hour of day from NSM
    2. day_of_week (0-6): Numeric day from day name
    3. is_weekend (bool): Weekend indicator
    4. cyclical_hour_sin: Sine encoding of hour
    5. cyclical_hour_cos: Cosine encoding of hour
    6. cyclical_day_sin: Sine encoding of day_of_week
    7. cyclical_day_cos: Cosine encoding of day_of_week

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with NSM and Day_of_week columns
    nsm_col : str, default='NSM'
        Name of the NSM column
    day_name_col : str, default='Day_of_week'
        Name of the day name column

    Returns
    -------
    pl.DataFrame
        DataFrame with 7 new temporal features added

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>> df_featured = create_all_temporal_features(df)
    >>> print(f"Original columns: {len(df.columns)}")
    >>> print(f"Featured columns: {len(df_featured.columns)}")
    >>> # Original columns: 11
    >>> # Featured columns: 18 (+7 new features)
    """
    # Step 1: Extract hour from NSM
    df = extract_hour_from_nsm(df, nsm_col=nsm_col, output_col="hour")

    # Step 2: Convert day name to numeric
    df = extract_day_of_week_numeric(df, day_col=day_name_col, output_col="day_of_week")

    # Step 3: Create weekend indicator
    df = create_is_weekend(df, day_num_col="day_of_week", output_col="is_weekend")

    # Step 4: Cyclical encoding for hour (period=24)
    df = create_cyclical_encoding(
        df, col="hour", period=24, sin_col="cyclical_hour_sin", cos_col="cyclical_hour_cos"
    )

    # Step 5: Cyclical encoding for day_of_week (period=7)
    df = create_cyclical_encoding(
        df, col="day_of_week", period=7, sin_col="cyclical_day_sin", cos_col="cyclical_day_cos"
    )

    return df


def validate_temporal_features(
    df: pl.DataFrame, required_features: list[str] = None
) -> dict[str, bool | list[str] | dict[str, any]]:
    """
    Validate that all temporal features are present and valid.

    Performs the following checks:
    1. All required features exist
    2. hour is in range [0, 23]
    3. day_of_week is in range [0, 6]
    4. is_weekend is boolean
    5. Cyclical features are in range [-1, 1]
    6. Cyclical orthogonality: sin² + cos² ≈ 1

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate
    required_features : List[str], optional
        List of required feature names. Default: US-011 features

    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool - Overall validation result
        - 'missing_features': List[str] - Missing required features
        - 'invalid_ranges': Dict[str, str] - Features with invalid ranges
        - 'stats': Dict[str, Dict] - Statistics for each feature

    Examples
    --------
    >>> df_featured = create_all_temporal_features(df)
    >>> validation = validate_temporal_features(df_featured)
    >>> if validation['valid']:
    ...     print("✅ All features valid")
    ... else:
    ...     print(f"❌ Issues: {validation}")
    """
    if required_features is None:
        required_features = [
            "hour",
            "day_of_week",
            "is_weekend",
            "cyclical_hour_sin",
            "cyclical_hour_cos",
            "cyclical_day_sin",
            "cyclical_day_cos",
        ]

    result = {"valid": True, "missing_features": [], "invalid_ranges": {}, "stats": {}}

    # Check 1: All features present
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        result["valid"] = False
        result["missing_features"] = missing
        return result

    # Check 2: hour range [0, 23]
    if "hour" in df.columns:
        hour_min, hour_max = df["hour"].min(), df["hour"].max()
        result["stats"]["hour"] = {"min": hour_min, "max": hour_max}

        if hour_min < 0 or hour_max > 23:
            result["valid"] = False
            result["invalid_ranges"]["hour"] = f"Expected [0, 23], got [{hour_min}, {hour_max}]"

    # Check 3: day_of_week range [0, 6]
    if "day_of_week" in df.columns:
        day_min, day_max = df["day_of_week"].min(), df["day_of_week"].max()
        result["stats"]["day_of_week"] = {"min": day_min, "max": day_max}

        if day_min < 0 or day_max > 6:
            result["valid"] = False
            result["invalid_ranges"]["day_of_week"] = f"Expected [0, 6], got [{day_min}, {day_max}]"

    # Check 4: is_weekend is boolean
    if "is_weekend" in df.columns:
        if df["is_weekend"].dtype != pl.Boolean:
            result["valid"] = False
            result["invalid_ranges"][
                "is_weekend"
            ] = f"Expected Boolean, got {df['is_weekend'].dtype}"

    # Check 5: Cyclical features in range [-1, 1]
    cyclical_features = [
        "cyclical_hour_sin",
        "cyclical_hour_cos",
        "cyclical_day_sin",
        "cyclical_day_cos",
    ]

    for feat in cyclical_features:
        if feat in df.columns:
            feat_min, feat_max = df[feat].min(), df[feat].max()
            result["stats"][feat] = {"min": feat_min, "max": feat_max}

            # Allow small floating point tolerance
            if feat_min < -1.01 or feat_max > 1.01:
                result["valid"] = False
                result["invalid_ranges"][
                    feat
                ] = f"Expected [-1, 1], got [{feat_min:.4f}, {feat_max:.4f}]"

    # Check 6: Cyclical orthogonality (sin² + cos² ≈ 1)
    if all(f in df.columns for f in ["cyclical_hour_sin", "cyclical_hour_cos"]):
        hour_norm = df["cyclical_hour_sin"] ** 2 + df["cyclical_hour_cos"] ** 2
        hour_norm_mean = hour_norm.mean()
        result["stats"]["hour_orthogonality"] = {"mean_norm": hour_norm_mean}

        # Check if mean is approximately 1.0 (tolerance: 0.01)
        if abs(hour_norm_mean - 1.0) > 0.01:
            result["valid"] = False
            result["invalid_ranges"][
                "hour_orthogonality"
            ] = f"Expected ≈1.0, got {hour_norm_mean:.6f}"

    if all(f in df.columns for f in ["cyclical_day_sin", "cyclical_day_cos"]):
        day_norm = df["cyclical_day_sin"] ** 2 + df["cyclical_day_cos"] ** 2
        day_norm_mean = day_norm.mean()
        result["stats"]["day_orthogonality"] = {"mean_norm": day_norm_mean}

        if abs(day_norm_mean - 1.0) > 0.01:
            result["valid"] = False
            result["invalid_ranges"][
                "day_orthogonality"
            ] = f"Expected ≈1.0, got {day_norm_mean:.6f}"

    return result
