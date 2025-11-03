"""
EDA Plotting Utilities

Reusable interactive plotting functions using Plotly for exploratory data analysis.

"""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_distribution(
    df: pl.DataFrame,
    column: str,
    title: Optional[str] = None,
    nbins: int = 50,
    show_kde: bool = True,
    color: str = "#636EFA",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Creates a distribution plot (histogram + KDE) for a single variable.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    title : str, optional
        Plot title. If None, auto-generated
    nbins : int, default=50
        Number of histogram bins
    show_kde : bool, default=True
        Whether to show KDE curve overlay
    color : str, default='#636EFA'
        Bar color (hex or named color)
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> from src.utils.eda_plots import plot_distribution
    >>> fig = plot_distribution(df, 'Usage_kWh', output_path='reports/figures/usage_dist.html')
    >>> fig.show()
    """
    if title is None:
        title = f"Distribution of {column}"

    # Convert to pandas for plotly compatibility
    data = df.select(column).to_pandas()

    fig = px.histogram(
        data,
        x=column,
        nbins=nbins,
        title=title,
        labels={column: column.replace('_', ' ')},
        marginal="box" if show_kde else None,
        color_discrete_sequence=[color]
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=500,
        font=dict(size=12)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_correlation_heatmap(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    colorscale: str = "RdBu_r",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Creates an interactive correlation heatmap.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    columns : list of str, optional
        Columns to include. If None, uses all numeric columns
    title : str, default='Correlation Matrix'
        Plot title
    colorscale : str, default='RdBu_r'
        Plotly colorscale name
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> from src.utils.eda_plots import plot_correlation_heatmap
    >>> fig = plot_correlation_heatmap(df, output_path='reports/figures/correlation.html')
    """
    # Select numeric columns
    if columns is None:
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    else:
        numeric_cols = columns

    # Calculate correlation matrix using polars
    data = df.select(numeric_cols).to_pandas()
    corr_matrix = data.corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=colorscale,
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700,
        width=800,
        xaxis=dict(tickangle=-45),
        font=dict(size=11)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_time_series(
    df: pl.DataFrame,
    time_column: str,
    value_column: str,
    title: Optional[str] = None,
    aggregation: Optional[str] = None,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Creates a time series line plot.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    time_column : str
        Column with time/hour values
    value_column : str
        Column with values to plot
    title : str, optional
        Plot title
    aggregation : str, optional
        Aggregation method: 'mean', 'sum', 'median'
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> from src.utils.eda_plots import plot_time_series
    >>> fig = plot_time_series(df, 'hour', 'Usage_kWh', aggregation='mean')
    """
    if title is None:
        title = f"{value_column} over {time_column}"

    # Aggregate if requested
    if aggregation:
        if aggregation == 'mean':
            plot_df = df.group_by(time_column).agg(pl.col(value_column).mean().alias(value_column))
        elif aggregation == 'sum':
            plot_df = df.group_by(time_column).agg(pl.col(value_column).sum().alias(value_column))
        elif aggregation == 'median':
            plot_df = df.group_by(time_column).agg(pl.col(value_column).median().alias(value_column))
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")
        plot_df = plot_df.sort(time_column)
    else:
        plot_df = df

    data = plot_df.to_pandas()

    fig = px.line(
        data,
        x=time_column,
        y=value_column,
        title=title,
        labels={
            time_column: time_column.replace('_', ' ').title(),
            value_column: value_column.replace('_', ' ')
        }
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=500,
        font=dict(size=12)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_box_by_category(
    df: pl.DataFrame,
    category_column: str,
    value_column: str,
    title: Optional[str] = None,
    color_discrete_sequence: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Creates box plots grouped by a categorical variable.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    category_column : str
        Categorical column for grouping
    value_column : str
        Numeric column for box plots
    title : str, optional
        Plot title
    color_discrete_sequence : list of str, optional
        Custom color palette
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> from src.utils.eda_plots import plot_box_by_category
    >>> fig = plot_box_by_category(df, 'Load_Type', 'Usage_kWh')
    """
    if title is None:
        title = f"{value_column} by {category_column}"

    data = df.to_pandas()

    fig = px.box(
        data,
        x=category_column,
        y=value_column,
        title=title,
        labels={
            category_column: category_column.replace('_', ' ').title(),
            value_column: value_column.replace('_', ' ')
        },
        color=category_column,
        color_discrete_sequence=color_discrete_sequence
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=500,
        font=dict(size=12)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_scatter(
    df: pl.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    trendline: bool = False,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Creates a scatter plot with optional color coding and trendline.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    x_column : str
        Column for x-axis
    y_column : str
        Column for y-axis
    color_column : str, optional
        Column for color coding
    title : str, optional
        Plot title
    trendline : bool, default=False
        Whether to add OLS trendline
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> from src.utils.eda_plots import plot_scatter
    >>> fig = plot_scatter(df, 'CO2(tCO2)', 'Usage_kWh', trendline=True)
    """
    if title is None:
        title = f"{y_column} vs {x_column}"

    data = df.to_pandas()

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        title=title,
        labels={
            x_column: x_column.replace('_', ' '),
            y_column: y_column.replace('_', ' ')
        },
        trendline="ols" if trendline else None,
        opacity=0.6
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        font=dict(size=12)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_scatter_matrix(
    df: pl.DataFrame,
    columns: List[str],
    color_column: Optional[str] = None,
    title: str = "Scatter Matrix",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Creates a scatter matrix (pairplot) for multiple variables.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    columns : list of str
        Columns to include in the matrix
    color_column : str, optional
        Column for color coding
    title : str, default='Scatter Matrix'
        Plot title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    >>> from src.utils.eda_plots import plot_scatter_matrix
    >>> top_features = ['Usage_kWh', 'CO2(tCO2)', 'Lagging_Current_Reactive.Power_kVarh']
    >>> fig = plot_scatter_matrix(df, top_features, color_column='Load_Type')
    """
    data = df.select(columns + ([color_column] if color_column else [])).to_pandas()

    fig = px.scatter_matrix(
        data,
        dimensions=columns,
        color=color_column,
        title=title,
        opacity=0.5
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(
        template="plotly_white",
        height=800,
        width=1000,
        font=dict(size=10)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def create_multi_plot(
    figures: List[go.Figure],
    rows: int,
    cols: int,
    subplot_titles: Optional[List[str]] = None,
    main_title: str = "Dashboard",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Combines multiple figures into a single subplot layout.

    Parameters
    ----------
    figures : list of go.Figure
        List of Plotly figures to combine
    rows : int
        Number of rows
    cols : int
        Number of columns
    subplot_titles : list of str, optional
        Titles for each subplot
    main_title : str, default='Dashboard'
        Overall title
    output_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    go.Figure
        Combined Plotly figure

    Examples
    --------
    >>> from src.utils.eda_plots import create_multi_plot
    >>> figs = [fig1, fig2, fig3, fig4]
    >>> combined = create_multi_plot(figs, rows=2, cols=2)
    """
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles
    )

    for idx, single_fig in enumerate(figures):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        for trace in single_fig.data:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title_text=main_title,
        template="plotly_white",
        showlegend=False,
        height=400 * rows,
        font=dict(size=11)
    )

    if output_path:
        _save_figure(fig, output_path)

    return fig


def _save_figure(fig: go.Figure, output_path: str) -> None:
    """
    Saves a Plotly figure to HTML file.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to save
    output_path : str
        Path where to save the figure
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_path)
    logger.info(f"📊 Figure saved to: {output_path}")


def get_top_correlated_features(
    df: pl.DataFrame,
    target_column: str,
    n: int = 5,
    exclude_columns: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Finds the top N features most correlated with the target.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    target_column : str
        Target column name
    n : int, default=5
        Number of top features to return
    exclude_columns : list of str, optional
        Columns to exclude from analysis

    Returns
    -------
    list of tuple
        List of (column_name, correlation) tuples, sorted by absolute correlation

    Examples
    --------
    >>> from src.utils.eda_plots import get_top_correlated_features
    >>> top_features = get_top_correlated_features(df, 'Usage_kWh', n=5)
    >>> print(top_features)
    [('CO2(tCO2)', 0.98), ('Lagging_Current_Reactive.Power_kVarh', 0.75), ...]
    """
    # Get numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    # Remove target and excluded columns
    numeric_cols = [col for col in numeric_cols if col != target_column]
    if exclude_columns:
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    # Calculate correlations
    data = df.select([target_column] + numeric_cols).to_pandas()
    correlations = data.corr()[target_column].drop(target_column)

    # Sort by absolute value and get top N
    top_corr = correlations.abs().sort_values(ascending=False).head(n)

    # Return as list of tuples with original (signed) correlation values
    return [(col, correlations[col]) for col in top_corr.index]
