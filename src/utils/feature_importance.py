"""
Feature Importance Analysis Utilities

This module provides functions for analyzing feature importance in the Steel Industry
Energy Consumption dataset. It implements two complementary methods for identifying
the most predictive features:

1. Mutual Information: Captures non-linear relationships between features and target
2. Pearson Correlation: Measures linear relationships between features and target

The module supports:
- Calculation of mutual information scores using scikit-learn
- Calculation of Pearson correlation coefficients using Polars
- Visualization of top N features with horizontal bar charts
- Comparison of importance rankings between methods
- Export of results to CSV files

Example Usage:
    >>> import polars as pl
    >>> from src.utils.feature_importance import (
    ...     calculate_mutual_information,
    ...     calculate_pearson_correlation,
    ...     plot_feature_importance
    ... )
    >>>
    >>> # Load data
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>>
    >>> # Calculate mutual information scores
    >>> mi_scores = calculate_mutual_information(df, target_column="Usage_kWh")
    >>> print(mi_scores.head(10))
    >>>
    >>> # Calculate Pearson correlations
    >>> correlations = calculate_pearson_correlation(df, target_column="Usage_kWh")
    >>> print(correlations.head(10))
    >>>
    >>> # Visualize top 10 features
    >>> fig = plot_feature_importance(
    ...     mi_scores,
    ...     score_column="mi_score",
    ...     method_name="Mutual Information",
    ...     top_n=10,
    ...     output_path="reports/figures/mutual_information_top10.png"
    ... )

Requirements:
    - polars >= 0.20.0
    - scikit-learn >= 1.4.0
    - matplotlib >= 3.8.0
    - seaborn >= 0.13.0
    - numpy >= 1.26.0
"""

import matplotlib.pyplot as plt
import polars as pl
from sklearn.feature_selection import mutual_info_regression

# Module constants
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_NEIGHBORS = 3


def calculate_mutual_information(
    df: pl.DataFrame,
    target_column: str = "Usage_kWh",
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pl.DataFrame:
    """
    Calculate mutual information scores between features and target variable.

    Mutual information measures the dependency between each feature and the target,
    capturing both linear and non-linear relationships. Higher scores indicate
    stronger predictive power. Scores are always non-negative, with 0 indicating
    independence.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing features and target variable. Must include
        numeric columns for analysis.
    target_column : str, default="Usage_kWh"
        Name of the target variable column. Must exist in the dataframe.
    n_neighbors : int, default=3
        Number of neighbors to use for mutual information estimation. Higher
        values provide more stable estimates but may miss local patterns.
    random_state : int, default=42
        Random seed for reproducibility of mutual information calculations.

    Returns
    -------
    pl.DataFrame
        DataFrame with two columns:
        - 'feature': Feature name (str)
        - 'mi_score': Mutual information score (float, >= 0)
        Sorted by mi_score in descending order (highest importance first).

    Raises
    ------
    ValueError
        If target_column is not found in the dataframe.
        If no numeric features are available for analysis.
        If all rows contain null values after cleaning.

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_importance import calculate_mutual_information
    >>>
    >>> # Load data
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>>
    >>> # Calculate mutual information scores
    >>> mi_scores = calculate_mutual_information(df)
    >>> print(mi_scores.head(10))
    shape: (10, 2)
    ┌─────────────────────────────────────┬──────────┐
    │ feature                             ┆ mi_score │
    │ ---                                 ┆ ---      │
    │ str                                 ┆ f64      │
    ╞═════════════════════════════════════╪══════════╡
    │ Lagging_Current_Reactive_Power_kVarh┆ 2.456    │
    │ CO2(tCO2)                           ┆ 2.123    │
    │ ...                                 ┆ ...      │
    └─────────────────────────────────────┴──────────┘
    >>>
    >>> # Use custom parameters
    >>> mi_scores = calculate_mutual_information(
    ...     df,
    ...     target_column="Usage_kWh",
    ...     n_neighbors=5,
    ...     random_state=123
    ... )
    """
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Extract numeric columns only
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    # Remove target from feature list
    feature_cols = [col for col in numeric_cols if col != target_column]

    if len(feature_cols) == 0:
        raise ValueError("No numeric features found for analysis (excluding target)")

    # Handle missing values by dropping rows with nulls
    df_clean = df.select([target_column] + feature_cols).drop_nulls()

    if len(df_clean) == 0:
        raise ValueError("All rows contain null values after cleaning")

    # Convert to numpy arrays for sklearn compatibility
    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean.select(target_column).to_numpy().ravel()

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=random_state)

    # Create result DataFrame with feature names and scores
    result_df = pl.DataFrame({"feature": feature_cols, "mi_score": mi_scores})

    # Sort by mi_score descending
    result_df = result_df.sort("mi_score", descending=True)

    return result_df


def calculate_pearson_correlation(
    df: pl.DataFrame, target_column: str = "Usage_kWh"
) -> pl.DataFrame:
    """
    Calculate Pearson correlation coefficients between features and target variable.

    Pearson correlation measures the linear relationship between each feature and
    the target. Coefficients range from -1 (perfect negative correlation) to +1
    (perfect positive correlation), with 0 indicating no linear relationship.

    The function returns both the signed correlation (to identify direction of
    relationship) and absolute correlation (for ranking by strength regardless
    of direction).

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing features and target variable. Must include
        numeric columns for analysis.
    target_column : str, default="Usage_kWh"
        Name of the target variable column. Must exist in the dataframe.

    Returns
    -------
    pl.DataFrame
        DataFrame with three columns:
        - 'feature': Feature name (str)
        - 'correlation': Pearson correlation coefficient (float, -1 to 1)
        - 'abs_correlation': Absolute correlation value (float, 0 to 1)
        Sorted by abs_correlation in descending order (strongest relationship first).
        Target variable is excluded from results.

    Raises
    ------
    ValueError
        If target_column is not found in the dataframe.
        If no numeric features are available for analysis.

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_importance import calculate_pearson_correlation
    >>>
    >>> # Load data
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>>
    >>> # Calculate Pearson correlations
    >>> correlations = calculate_pearson_correlation(df)
    >>> print(correlations.head(10))
    shape: (10, 3)
    ┌─────────────────────────────────────┬─────────────┬─────────────────┐
    │ feature                             ┆ correlation ┆ abs_correlation │
    │ ---                                 ┆ ---         ┆ ---             │
    │ str                                 ┆ f64         ┆ f64             │
    ╞═════════════════════════════════════╪═════════════╪═════════════════╡
    │ Lagging_Current_Reactive_Power_kVarh┆ 0.856       ┆ 0.856           │
    │ CO2(tCO2)                           ┆ 0.823       ┆ 0.823           │
    │ ...                                 ┆ ...         ┆ ...             │
    └─────────────────────────────────────┴─────────────┴─────────────────┘
    >>>
    >>> # Identify positive vs negative correlations
    >>> positive_corr = correlations.filter(pl.col("correlation") > 0)
    >>> negative_corr = correlations.filter(pl.col("correlation") < 0)
    >>> print(f"Positive: {len(positive_corr)}, Negative: {len(negative_corr)}")
    """
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Select numeric columns only
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found for analysis")

    # Calculate correlation matrix using Polars
    corr_matrix = df.select(numeric_cols).corr()

    # Extract correlations with target column
    if target_column not in corr_matrix.columns:
        raise ValueError(f"Target column '{target_column}' not found in correlation matrix")

    # Get the target column correlations
    target_correlations = corr_matrix.select(target_column).to_series()

    # Create DataFrame with feature names and correlations
    result_df = pl.DataFrame({"feature": numeric_cols, "correlation": target_correlations})

    # Exclude target variable from results
    result_df = result_df.filter(pl.col("feature") != target_column)

    # Add absolute correlation column for ranking
    result_df = result_df.with_columns(pl.col("correlation").abs().alias("abs_correlation"))

    # Sort by absolute correlation descending (nulls last)
    result_df = result_df.sort("abs_correlation", descending=True, nulls_last=True)

    return result_df


def plot_feature_importance(
    importance_df: pl.DataFrame,
    score_column: str,
    method_name: str,
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6),
    output_path: str | None = None,
    color_by_sign: bool = False,
) -> plt.Figure:
    """
    Create horizontal bar chart of feature importance scores.

    This function visualizes the top N features ranked by their importance scores,
    using a horizontal bar chart for easy comparison. Supports color coding by
    sign (for correlation plots) to distinguish positive from negative relationships.

    Parameters
    ----------
    importance_df : pl.DataFrame
        DataFrame containing feature names and importance scores. Must have at
        least two columns: one for feature names and one for scores.
    score_column : str
        Name of the column containing importance scores (e.g., 'mi_score',
        'correlation', 'abs_correlation').
    method_name : str
        Name of the importance method for the plot title (e.g., 'Mutual Information',
        'Pearson Correlation').
    top_n : int, default=10
        Number of top features to display. If top_n exceeds the number of features,
        all features will be displayed.
    figsize : tuple[int, int], default=(10, 6)
        Figure size in inches as (width, height).
    output_path : str | None, default=None
        Path to save the figure. If None, figure is not saved to disk.
        Recommended paths: 'reports/figures/mutual_information_top10.png'
    color_by_sign : bool, default=False
        If True, colors bars by sign of the score column:
        - Blue for positive values
        - Red for negative values
        Useful for correlation plots to show direction of relationship.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object that can be further customized or displayed.

    Raises
    ------
    ValueError
        If score_column is not found in importance_df.
        If top_n is not a positive integer.

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_importance import (
    ...     calculate_mutual_information,
    ...     plot_feature_importance
    ... )
    >>>
    >>> # Load data and calculate MI scores
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>> mi_scores = calculate_mutual_information(df)
    >>>
    >>> # Plot top 10 features by mutual information
    >>> fig = plot_feature_importance(
    ...     mi_scores,
    ...     score_column='mi_score',
    ...     method_name='Mutual Information',
    ...     top_n=10,
    ...     output_path='reports/figures/mutual_information_top10.png'
    ... )
    >>> plt.show()
    >>>
    >>> # Plot correlations with color coding
    >>> correlations = calculate_pearson_correlation(df)
    >>> fig = plot_feature_importance(
    ...     correlations,
    ...     score_column='correlation',
    ...     method_name='Pearson Correlation',
    ...     top_n=10,
    ...     color_by_sign=True,
    ...     output_path='reports/figures/pearson_correlation_top10.png'
    ... )
    """
    # Validate score_column exists
    if score_column not in importance_df.columns:
        raise ValueError(f"Score column '{score_column}' not found in importance_df")

    # Validate top_n
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    # Adjust top_n if it exceeds number of features
    n_features = len(importance_df)
    if top_n > n_features:
        top_n = n_features

    # Select top N features
    top_features_df = importance_df.head(top_n)

    # Extract data for plotting (reverse order for horizontal bar chart)
    features = top_features_df["feature"].to_list()[::-1]
    scores = top_features_df[score_column].to_list()[::-1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if color_by_sign:
        # Color by sign: blue for positive, red for negative
        colors = ["#2E86AB" if score >= 0 else "#E63946" for score in scores]
    else:
        # Default color (seaborn blue)
        colors = "#2E86AB"

    # Create horizontal bar chart
    ax.barh(features, scores, color=colors, edgecolor="black", linewidth=0.5)

    # Add axis labels
    ax.set_xlabel(f"{method_name} Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")

    # Add title
    ax.set_title(f"Top {top_n} Features by {method_name}", fontsize=14, fontweight="bold", pad=20)

    # Add grid for readability
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure if output_path is provided
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def get_top_features(
    importance_df: pl.DataFrame,
    score_column: str,
    top_n: int = 10,
) -> list[str]:
    """
    Extract list of top N feature names from importance DataFrame.

    This utility function simplifies extracting feature names from importance
    analysis results, useful for feature selection pipelines or reporting.

    Parameters
    ----------
    importance_df : pl.DataFrame
        DataFrame containing feature importance scores. Must have a 'feature'
        column with feature names and a score column for ranking.
    score_column : str
        Name of the column containing importance scores used for ranking.
        The DataFrame should already be sorted by this column in descending order.
    top_n : int, default=10
        Number of top features to return. If top_n exceeds the number of features,
        all features will be returned.

    Returns
    -------
    list[str]
        List of feature names, ordered by importance (highest first).

    Raises
    ------
    ValueError
        If 'feature' column is not found in importance_df.
        If score_column is not found in importance_df.
        If top_n is not a positive integer.

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_importance import (
    ...     calculate_mutual_information,
    ...     get_top_features
    ... )
    >>>
    >>> # Load data and calculate MI scores
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>> mi_scores = calculate_mutual_information(df)
    >>>
    >>> # Get top 5 features
    >>> top_features = get_top_features(mi_scores, 'mi_score', top_n=5)
    >>> print(top_features)
    ['Lagging_Current_Reactive_Power_kVarh', 'CO2(tCO2)', ...]
    >>>
    >>> # Use for feature selection
    >>> selected_features = get_top_features(mi_scores, 'mi_score', top_n=10)
    >>> df_selected = df.select(selected_features + ['Usage_kWh'])
    """
    # Validate feature column exists
    if "feature" not in importance_df.columns:
        raise ValueError("'feature' column not found in importance_df")

    # Validate score_column exists
    if score_column not in importance_df.columns:
        raise ValueError(f"Score column '{score_column}' not found in importance_df")

    # Validate top_n
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    # Adjust top_n if it exceeds number of features
    n_features = len(importance_df)
    if top_n > n_features:
        top_n = n_features

    # Select top N rows and extract feature names as list
    top_features = importance_df.head(top_n)["feature"].to_list()

    return top_features


def compare_importance_methods(
    mi_df: pl.DataFrame,
    corr_df: pl.DataFrame,
    top_n: int = 10,
) -> dict[str, list[str] | float]:
    """
    Compare mutual information and correlation importance methods.

    This function identifies which features are consistently ranked highly by both
    methods (consensus features) versus features that are method-specific. High
    overlap suggests robust feature importance, while method-specific features
    may indicate non-linear relationships (MI-only) or purely linear relationships
    (correlation-only).

    Parameters
    ----------
    mi_df : pl.DataFrame
        Mutual information scores DataFrame with 'feature' and 'mi_score' columns.
        Should be sorted by mi_score in descending order.
    corr_df : pl.DataFrame
        Pearson correlation DataFrame with 'feature', 'correlation', and
        'abs_correlation' columns. Should be sorted by abs_correlation descending.
    top_n : int, default=10
        Number of top features to compare from each method.

    Returns
    -------
    dict[str, list[str] | float]
        Dictionary containing comparison metrics:
        - 'common_features': List of features appearing in both top N lists
        - 'mi_only': List of features only in MI top N (not in correlation top N)
        - 'corr_only': List of features only in correlation top N (not in MI top N)
        - 'overlap_percentage': Percentage of overlap (0-100)
        - 'n_common': Number of common features (int)
        - 'n_mi_only': Number of MI-only features (int)
        - 'n_corr_only': Number of correlation-only features (int)

    Raises
    ------
    ValueError
        If 'feature' column is not found in either DataFrame.
        If top_n is not a positive integer.

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_importance import (
    ...     calculate_mutual_information,
    ...     calculate_pearson_correlation,
    ...     compare_importance_methods
    ... )
    >>>
    >>> # Load data and calculate both importance methods
    >>> df = pl.read_parquet("data/processed/steel_cleaned.parquet")
    >>> mi_scores = calculate_mutual_information(df)
    >>> correlations = calculate_pearson_correlation(df)
    >>>
    >>> # Compare methods
    >>> comparison = compare_importance_methods(mi_scores, correlations, top_n=10)
    >>> print(f"Overlap: {comparison['overlap_percentage']:.1f}%")
    Overlap: 70.0%
    >>> print(f"Common features: {comparison['common_features']}")
    ['Lagging_Current_Reactive_Power_kVarh', 'CO2(tCO2)', ...]
    >>>
    >>> # Identify method-specific features
    >>> print(f"MI-only features: {comparison['mi_only']}")
    >>> print(f"Correlation-only features: {comparison['corr_only']}")
    >>>
    >>> # High overlap suggests robust feature importance
    >>> if comparison['overlap_percentage'] > 80:
    ...     print("Strong consensus between methods!")
    """
    # Validate feature column exists in both DataFrames
    if "feature" not in mi_df.columns:
        raise ValueError("'feature' column not found in mi_df")
    if "feature" not in corr_df.columns:
        raise ValueError("'feature' column not found in corr_df")

    # Validate top_n
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    # Extract top N features from MI DataFrame
    mi_top_features = set(mi_df.head(top_n)["feature"].to_list())

    # Extract top N features from correlation DataFrame
    corr_top_features = set(corr_df.head(top_n)["feature"].to_list())

    # Find common features using set intersection
    common_features = mi_top_features & corr_top_features

    # Find MI-only features using set difference
    mi_only_features = mi_top_features - corr_top_features

    # Find correlation-only features using set difference
    corr_only_features = corr_top_features - mi_top_features

    # Calculate overlap percentage
    # Overlap = (common features / top_n) * 100
    overlap_percentage = (len(common_features) / top_n) * 100

    # Return dictionary with comparison metrics
    return {
        "common_features": sorted(common_features),
        "mi_only": sorted(mi_only_features),
        "corr_only": sorted(corr_only_features),
        "overlap_percentage": overlap_percentage,
        "n_common": len(common_features),
        "n_mi_only": len(mi_only_features),
        "n_corr_only": len(corr_only_features),
    }
