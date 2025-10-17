"""
Time Series Analysis Utilities

Reusable functions for time series decomposition, ACF/PACF analysis, and seasonality detection.
Using Matplotlib/Seaborn for static visualizations that render well in GitHub.
Following project standards for code reusability and clean code.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd

logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def perform_stl_decomposition(
    df: pl.DataFrame,
    time_column: str,
    value_column: str,
    period: int,
    seasonal: int = 7,
    trend: Optional[int] = None,
    robust: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs STL (Seasonal-Trend decomposition using Loess) on time series data.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with time series data
    time_column : str
        Name of the time/date column
    value_column : str
        Name of the value column to decompose
    period : int
        Periodicity of the sequence (e.g., 24 for hourly data with daily seasonality)
    seasonal : int, default=7
        Length of the seasonal smoother. Must be odd
    trend : int, optional
        Length of the trend smoother. If None, defaults to the smallest odd integer
        greater than 1.5 * period / (1 - 1.5 / seasonal)
    robust : bool, default=True
        Flag indicating whether to use robust fitting

    Returns
    -------
    tuple
        - pd.DataFrame: DataFrame with original values and decomposition components
        - dict: Metadata about the decomposition

    Examples
    --------
    >>> from src.utils.time_series import perform_stl_decomposition
    >>> decomp_df, metadata = perform_stl_decomposition(
    ...     df, 'date', 'Usage_kWh', period=24, seasonal=7
    ... )
    """
    # Convert to pandas for statsmodels compatibility
    ts_data = df.select([time_column, value_column]).to_pandas()
    ts_data = ts_data.sort_values(time_column).set_index(time_column)

    # Ensure the series is numeric and handle missing values
    series = ts_data[value_column].astype(float)

    # Perform STL decomposition
    stl = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust
    )

    result = stl.fit()

    # Create result DataFrame
    decomp_df = pd.DataFrame({
        'observed': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid
    })

    # Calculate metadata
    metadata = {
        'period': period,
        'seasonal_strength': _calculate_seasonal_strength(result),
        'trend_strength': _calculate_trend_strength(result),
        'seasonal_peak_to_trough': result.seasonal.max() - result.seasonal.min(),
        'residual_std': result.resid.std(),
        'residual_mean': result.resid.mean()
    }

    logger.info(f"STL decomposition completed. Seasonal strength: {metadata['seasonal_strength']:.4f}")

    return decomp_df, metadata


def _calculate_seasonal_strength(stl_result) -> float:
    """
    Calculate the strength of seasonality (0 to 1).

    Parameters
    ----------
    stl_result : STLResult
        Result from STL decomposition

    Returns
    -------
    float
        Seasonal strength metric
    """
    var_resid = np.var(stl_result.resid)
    var_deseasonalized = np.var(stl_result.resid + stl_result.seasonal)

    if var_deseasonalized == 0:
        return 0.0

    return max(0, 1 - (var_resid / var_deseasonalized))


def _calculate_trend_strength(stl_result) -> float:
    """
    Calculate the strength of trend (0 to 1).

    Parameters
    ----------
    stl_result : STLResult
        Result from STL decomposition

    Returns
    -------
    float
        Trend strength metric
    """
    var_resid = np.var(stl_result.resid)
    var_detrended = np.var(stl_result.resid + stl_result.trend)

    if var_detrended == 0:
        return 0.0

    return max(0, 1 - (var_resid / var_detrended))


def _calculate_seasonal_strength(stl_result) -> float:
    """
    Calculate the strength of seasonality (0 to 1).

    Parameters
    ----------
    stl_result : STLResult
        Result from STL decomposition

    Returns
    -------
    float
        Seasonal strength metric
    """
    var_resid = np.var(stl_result.resid)
    var_deseasonalized = np.var(stl_result.resid + stl_result.seasonal)

    if var_deseasonalized == 0:
        return 0.0

    return max(0, 1 - (var_resid / var_deseasonalized))


def _calculate_trend_strength(stl_result) -> float:
    """
    Calculate the strength of trend (0 to 1).

    Parameters
    ----------
    stl_result : STLResult
        Result from STL decomposition

    Returns
    -------
    float
        Trend strength metric
    """
    var_resid = np.var(stl_result.resid)
    var_detrended = np.var(stl_result.resid + stl_result.trend)

    if var_detrended == 0:
        return 0.0

    return max(0, 1 - (var_resid / var_detrended))


def plot_stl_components(
    decomp_df: pd.DataFrame,
    title: str = "STL Decomposition",
    figsize: Tuple[int, int] = (14, 10),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates a plot of STL decomposition components using Matplotlib.

    Parameters
    ----------
    decomp_df : pd.DataFrame
        DataFrame with decomposition components (from perform_stl_decomposition)
    title : str, default='STL Decomposition'
        Main title for the plot
    figsize : tuple, default=(14, 10)
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure object with 4 subplots (observed, trend, seasonal, residual)

    Examples
    --------
    >>> from src.utils.time_series import plot_stl_components
    >>> fig = plot_stl_components(decomp_df, 'Energy Consumption STL Decomposition')
    >>> plt.show()
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Observed
    axes[0].plot(decomp_df.index, decomp_df['observed'], color='#3498db', linewidth=1)
    axes[0].set_ylabel('Observed', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(decomp_df.index, decomp_df['trend'], color='#e74c3c', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(decomp_df.index, decomp_df['seasonal'], color='#2ecc71', linewidth=1)
    axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].scatter(decomp_df.index, decomp_df['resid'], color='#95a5a6', s=10, alpha=0.5)
    axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    # Add title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def calculate_acf_pacf(
    series: pl.Series,
    nlags: int = 40,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates ACF and PACF for a time series.

    Parameters
    ----------
    series : pl.Series
        Time series data
    nlags : int, default=40
        Number of lags to calculate
    alpha : float, default=0.05
        Significance level for confidence intervals

    Returns
    -------
    tuple
        - acf_values: ACF values
        - acf_confint: ACF confidence intervals
        - pacf_values: PACF values
        - pacf_confint: PACF confidence intervals

    Examples
    --------
    >>> from src.utils.time_series import calculate_acf_pacf
    >>> acf_vals, acf_ci, pacf_vals, pacf_ci = calculate_acf_pacf(df['Usage_kWh'], nlags=48)
    """
    # Convert to numpy array
    data = series.to_numpy()

    # Remove NaN values
    data = data[~np.isnan(data)]

    # Calculate ACF
    acf_values = acf(data, nlags=nlags, alpha=alpha)
    if isinstance(acf_values, tuple):
        acf_vals, acf_confint = acf_values
    else:
        acf_vals = acf_values
        # Calculate confidence intervals manually
        n = len(data)
        acf_confint = np.array([
            [-1.96/np.sqrt(n), 1.96/np.sqrt(n)] for _ in range(nlags + 1)
        ])

    # Calculate PACF
    pacf_values = pacf(data, nlags=nlags, alpha=alpha)
    if isinstance(pacf_values, tuple):
        pacf_vals, pacf_confint = pacf_values
    else:
        pacf_vals = pacf_values
        # Calculate confidence intervals manually
        n = len(data)
        pacf_confint = np.array([
            [-1.96/np.sqrt(n), 1.96/np.sqrt(n)] for _ in range(nlags + 1)
        ])

    return acf_vals, acf_confint, pacf_vals, pacf_confint


def plot_acf_pacf(
    series: pl.Series,
    nlags: int = 40,
    title: str = "ACF and PACF Analysis",
    figsize: Tuple[int, int] = (14, 5),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates ACF and PACF plots using statsmodels visualization.

    Parameters
    ----------
    series : pl.Series
        Time series data
    nlags : int, default=40
        Number of lags to plot
    title : str, default='ACF and PACF Analysis'
        Main title for the plot
    figsize : tuple, default=(14, 5)
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure with ACF and PACF plots

    Examples
    --------
    >>> from src.utils.time_series import plot_acf_pacf
    >>> fig = plot_acf_pacf(df['Usage_kWh'], nlags=48)
    >>> plt.show()
    """
    # Convert to pandas Series for statsmodels
    data = series.to_pandas()

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF plot
    plot_acf(data, lags=nlags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lag', fontsize=11)
    axes[0].set_ylabel('Correlation', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # PACF plot
    plot_pacf(data, lags=nlags, ax=axes[1], alpha=0.05)
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lag', fontsize=11)
    axes[1].set_ylabel('Correlation', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Add main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def analyze_seasonality_by_group(
    df: pl.DataFrame,
    group_column: str,
    value_column: str,
    time_column: str,
    period: int,
    seasonal: int = 7
) -> Dict[str, Dict]:
    """
    Analyzes seasonality patterns separately for each group in a categorical variable.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    group_column : str
        Categorical column to group by (e.g., 'Load_Type')
    value_column : str
        Value column to analyze
    time_column : str
        Time/date column
    period : int
        Periodicity of the sequence
    seasonal : int, default=7
        Length of the seasonal smoother

    Returns
    -------
    dict
        Dictionary with group names as keys and decomposition metadata as values

    Examples
    --------
    >>> from src.utils.time_series import analyze_seasonality_by_group
    >>> results = analyze_seasonality_by_group(
    ...     df, 'Load_Type', 'Usage_kWh', 'date', period=24
    ... )
    >>> print(results['Light_Load']['seasonal_strength'])
    """
    results = {}

    # Get unique groups
    groups = df[group_column].unique().to_list()

    for group in groups:
        if group is None:
            continue

        # Filter data for this group
        group_df = df.filter(pl.col(group_column) == group)

        try:
            # Perform STL decomposition
            _, metadata = perform_stl_decomposition(
                group_df,
                time_column=time_column,
                value_column=value_column,
                period=period,
                seasonal=seasonal
            )

            results[group] = metadata
            logger.info(f"Seasonality analysis completed for group: {group}")

        except Exception as e:
            logger.warning(f"Failed to analyze group {group}: {e}")
            results[group] = {'error': str(e)}

    return results


def plot_seasonality_comparison(
    seasonality_results: Dict[str, Dict],
    metric: str = 'seasonal_strength',
    title: str = "Seasonality Comparison by Group",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates a bar chart comparing seasonality metrics across groups.

    Parameters
    ----------
    seasonality_results : dict
        Results from analyze_seasonality_by_group
    metric : str, default='seasonal_strength'
        Metric to compare ('seasonal_strength', 'trend_strength', etc.)
    title : str
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib bar chart

    Examples
    --------
    >>> from src.utils.time_series import plot_seasonality_comparison
    >>> fig = plot_seasonality_comparison(results, 'seasonal_strength')
    >>> plt.show()
    """
    groups = []
    values = []

    for group, data in seasonality_results.items():
        if 'error' not in data and metric in data:
            groups.append(group)
            values.append(data[metric])

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(groups, values, color='#3498db', alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def extract_seasonal_pattern(
    decomp_df: pd.DataFrame,
    period: int
) -> pd.DataFrame:
    """
    Extracts the average seasonal pattern from decomposition.

    Parameters
    ----------
    decomp_df : pd.DataFrame
        DataFrame with decomposition components
    period : int
        Period of seasonality

    Returns
    -------
    pd.DataFrame
        Average seasonal pattern

    Examples
    --------
    >>> pattern = extract_seasonal_pattern(decomp_df, period=24)
    """
    seasonal = decomp_df['seasonal'].values
    n_periods = len(seasonal) // period

    # Reshape to get all periods
    seasonal_matrix = seasonal[:n_periods * period].reshape(n_periods, period)

    # Calculate average pattern
    avg_pattern = seasonal_matrix.mean(axis=0)

    return pd.DataFrame({
        'period_index': np.arange(period),
        'seasonal_value': avg_pattern
    })


def plot_seasonal_pattern(
    seasonal_pattern: pd.DataFrame,
    period_label: str = "Hour of Day",
    title: str = "Average Seasonal Pattern",
    figsize: Tuple[int, int] = (12, 5),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the average seasonal pattern.

    Parameters
    ----------
    seasonal_pattern : pd.DataFrame
        Average seasonal pattern from extract_seasonal_pattern
    period_label : str, default='Hour of Day'
        Label for the x-axis
    title : str
        Plot title
    figsize : tuple, default=(12, 5)
        Figure size (width, height) in inches
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> pattern = extract_seasonal_pattern(decomp_df, period=24)
    >>> fig = plot_seasonal_pattern(pattern, 'Hour of Day')
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(seasonal_pattern['period_index'],
            seasonal_pattern['seasonal_value'],
            marker='o', linewidth=2, markersize=6, color='#2ecc71')

    ax.fill_between(seasonal_pattern['period_index'],
                     seasonal_pattern['seasonal_value'],
                     alpha=0.3, color='#2ecc71')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(period_label, fontsize=12)
    ax.set_ylabel('Seasonal Component', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """
    Saves a Matplotlib figure to file.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save
    output_path : str
        Path where to save the figure
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Figure saved to: {output_path}")
