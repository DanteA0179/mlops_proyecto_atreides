"""
Preprocessing Utilities.

Helper functions for feature preprocessing, type identification, validation,
and analysis.
"""

import polars as pl


def identify_feature_types(
    df: pl.DataFrame, exclude_cols: list[str] = None
) -> dict[str, list[str]]:
    """
    Automatically identify feature types in dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    exclude_cols : list of str, optional
        Columns to exclude from analysis (e.g., target, date, ID)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'numeric': Numeric features (Int, Float)
        - 'categorical': Categorical features (String with low cardinality)
        - 'boolean': Boolean features
        - 'datetime': Datetime features
        - 'excluded': Excluded columns

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/steel_featured.parquet")
    >>> types = identify_feature_types(df, exclude_cols=['date', 'Usage_kWh'])
    >>> print(types['numeric'])
    >>> print(types['categorical'])
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_types = {
        "numeric": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
        "excluded": exclude_cols,
    }

    for col in df.columns:
        if col in exclude_cols:
            continue

        dtype = df[col].dtype

        # Datetime
        if dtype in [pl.Datetime, pl.Date]:
            feature_types["datetime"].append(col)

        # Boolean
        elif dtype == pl.Boolean:
            feature_types["boolean"].append(col)

        # Numeric (Int or Float)
        elif dtype in [
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
        ]:
            feature_types["numeric"].append(col)

        # String (check cardinality for categorical)
        elif dtype == pl.Utf8:
            n_unique = df[col].n_unique()
            n_rows = len(df)

            # If cardinality < 10% of rows, consider categorical
            if n_unique / n_rows < 0.10:
                feature_types["categorical"].append(col)
            else:
                # High cardinality string, might be ID or text
                feature_types["excluded"].append(col)

    return feature_types


def validate_preprocessing_config(
    numeric_features: list[str], categorical_features: list[str], df: pl.DataFrame
) -> dict[str, bool | list[str]]:
    """
    Validate preprocessing configuration.

    Parameters
    ----------
    numeric_features : list of str
        List of numeric features to scale
    categorical_features : list of str
        List of categorical features to encode
    df : pl.DataFrame
        Dataframe to validate against

    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool - Overall validation
        - 'missing_features': List of features not in df
        - 'duplicate_features': Features in both numeric and categorical
        - 'wrong_types': Features with incorrect types

    Examples
    --------
    >>> validation = validate_preprocessing_config(
    ...     numeric_features=['CO2(tCO2)', 'Usage_kWh'],
    ...     categorical_features=['Load_Type'],
    ...     df=df
    ... )
    >>> assert validation['valid']
    """
    result = {
        "valid": True,
        "missing_features": [],
        "duplicate_features": [],
        "wrong_types": {},
    }

    all_features = numeric_features + categorical_features
    df_cols = set(df.columns)

    # Check 1: All features exist in df
    for feat in all_features:
        if feat not in df_cols:
            result["valid"] = False
            result["missing_features"].append(feat)

    # Check 2: No duplicates between numeric and categorical
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)
    duplicates = numeric_set.intersection(categorical_set)

    if duplicates:
        result["valid"] = False
        result["duplicate_features"] = list(duplicates)

    # Check 3: Numeric features have numeric types
    for feat in numeric_features:
        if feat in df_cols:
            dtype = df[feat].dtype
            if dtype not in [
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
            ]:
                result["valid"] = False
                result["wrong_types"][feat] = f"Expected numeric, got {dtype}"

    # Check 4: Categorical features have string or categorical types
    for feat in categorical_features:
        if feat in df_cols:
            dtype = df[feat].dtype
            if dtype not in [pl.Utf8, pl.Categorical]:
                result["valid"] = False
                result["wrong_types"][feat] = f"Expected string/categorical, got {dtype}"

    return result


def calculate_scaling_statistics(
    df: pl.DataFrame, features: list[str]
) -> dict[str, dict[str, float]]:
    """
    Calculate statistics for features before/after scaling.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe with features
    features : list of str
        Features to analyze

    Returns
    -------
    dict
        Statistics for each feature: mean, std, min, max, median

    Examples
    --------
    >>> stats = calculate_scaling_statistics(df_train, ['CO2(tCO2)', 'NSM'])
    >>> print(stats['CO2(tCO2)']['mean'])
    >>> print(stats['CO2(tCO2)']['std'])
    """
    statistics = {}

    for feat in features:
        if feat not in df.columns:
            continue

        statistics[feat] = {
            "mean": float(df[feat].mean()),
            "std": float(df[feat].std()),
            "min": float(df[feat].min()),
            "max": float(df[feat].max()),
            "median": float(df[feat].median()),
            "q25": float(df[feat].quantile(0.25)),
            "q75": float(df[feat].quantile(0.75)),
        }

    return statistics


def analyze_categorical_cardinality(
    df: pl.DataFrame, feature: str
) -> dict[str, int | list[str] | dict[str, int]]:
    """
    Analyze cardinality and distribution of categorical feature.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    feature : str
        Categorical feature name

    Returns
    -------
    dict
        Analysis results with keys:
        - 'n_categories': Number of unique categories
        - 'categories': List of category names
        - 'value_counts': Count per category
        - 'encoding_size': Dimensionality after OHE with drop='first'
        - 'most_common': Most frequent category
        - 'least_common': Least frequent category

    Examples
    --------
    >>> analysis = analyze_categorical_cardinality(df, 'Load_Type')
    >>> print(f"Categories: {analysis['n_categories']}")
    >>> print(f"OHE output dims: {analysis['encoding_size']}")
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe")

    # Get value counts
    value_counts_df = df[feature].value_counts().sort(feature)
    categories = [row[0] for row in value_counts_df.iter_rows()]
    counts = {row[0]: int(row[1]) for row in value_counts_df.iter_rows()}

    # Sort by count descending
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "n_categories": len(categories),
        "categories": categories,
        "value_counts": counts,
        "encoding_size": len(categories) - 1,  # drop='first'
        "most_common": sorted_counts[0][0] if sorted_counts else None,
        "least_common": sorted_counts[-1][0] if sorted_counts else None,
        "total_count": len(df),
        "distribution": {cat: count / len(df) for cat, count in counts.items()},
    }


def map_binary_feature(df: pl.DataFrame, feature: str, mapping: dict[str, int]) -> pl.DataFrame:
    """
    Map binary categorical feature to 0/1.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    feature : str
        Feature name to map
    mapping : dict
        Mapping dictionary {category: int}

    Returns
    -------
    pl.DataFrame
        Dataframe with mapped feature

    Examples
    --------
    >>> df = map_binary_feature(
    ...     df,
    ...     'WeekStatus',
    ...     mapping={'Weekday': 0, 'Weekend': 1}
    ... )
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe")

    # Create mapping expression
    mapping_expr = pl.col(feature)
    for i, (category, value) in enumerate(mapping.items()):
        if i == 0:
            mapping_expr = pl.when(pl.col(feature) == category).then(value)
        else:
            mapping_expr = mapping_expr.when(pl.col(feature) == category).then(value)

    # Otherwise keep original (or could raise error for unknown categories)
    mapping_expr = mapping_expr.otherwise(None)

    return df.with_columns(mapping_expr.cast(pl.Int32).alias(feature))


def get_feature_name_after_ohe(feature: str, category: str, prefix_sep: str = "_") -> str:
    """
    Generate feature name after OneHotEncoding.

    Parameters
    ----------
    feature : str
        Original feature name
    category : str
        Category value
    prefix_sep : str, default='_'
        Separator between feature name and category

    Returns
    -------
    str
        Encoded feature name

    Examples
    --------
    >>> get_feature_name_after_ohe('Load_Type', 'Medium_Load')
    'Load_Type_Medium_Load'
    >>> get_feature_name_after_ohe('Load_Type', 'Maximum_Load')
    'Load_Type_Maximum_Load'
    """
    # Clean category name (replace spaces with underscores)
    clean_category = category.replace(" ", "_")
    return f"{feature}{prefix_sep}{clean_category}"
