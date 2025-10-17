"""
Tests for Time Series Analysis Utilities

Unit tests for src/utils/time_series.py functions.
"""

# Configure matplotlib backend FIRST, before any other imports
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for testing

import pytest
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.time_series import (
    perform_stl_decomposition,
    plot_stl_components,
    calculate_acf_pacf,
    plot_acf_pacf,
    analyze_seasonality_by_group,
    plot_seasonality_comparison,
    extract_seasonal_pattern,
    plot_seasonal_pattern
)


@pytest.fixture
def sample_time_series_df():
    """
    Create a sample time series DataFrame for testing.

    Returns
    -------
    pl.DataFrame
        Sample time series with synthetic seasonality
    """
    # Create synthetic time series with trend and seasonality
    n_points = 500
    t = np.arange(n_points)

    # Trend
    trend = 0.05 * t + 50

    # Daily seasonality (period=24)
    seasonal = 10 * np.sin(2 * np.pi * t / 24)

    # Random noise
    noise = np.random.normal(0, 2, n_points)

    # Combined signal
    values = trend + seasonal + noise

    # Create date range
    dates = pd.date_range('2024-01-01', periods=n_points, freq='h')

    return pl.DataFrame({
        'date': dates,
        'value': values,
        'category': ['A' if i % 2 == 0 else 'B' for i in range(n_points)]
    })


@pytest.fixture
def sample_series():
    """
    Create a simple time series for ACF/PACF testing.

    Returns
    -------
    pl.Series
        Sample time series
    """
    np.random.seed(42)
    n = 200
    # AR(1) process
    data = np.zeros(n)
    data[0] = np.random.normal(0, 1)
    for i in range(1, n):
        data[i] = 0.7 * data[i-1] + np.random.normal(0, 1)

    return pl.Series('value', data)


class TestPerformSTLDecomposition:
    """Tests for perform_stl_decomposition function."""

    def test_basic_decomposition(self, sample_time_series_df):
        """Test basic STL decomposition."""
        decomp_df, metadata = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24,
            seasonal=7
        )

        # Check output types
        assert isinstance(decomp_df, pd.DataFrame)
        assert isinstance(metadata, dict)

        # Check DataFrame columns
        assert 'observed' in decomp_df.columns
        assert 'trend' in decomp_df.columns
        assert 'seasonal' in decomp_df.columns
        assert 'resid' in decomp_df.columns

        # Check metadata keys
        assert 'period' in metadata
        assert 'seasonal_strength' in metadata
        assert 'trend_strength' in metadata

    def test_decomposition_shape(self, sample_time_series_df):
        """Test that decomposition preserves data length."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        assert len(decomp_df) == len(sample_time_series_df)

    def test_decomposition_reconstruction(self, sample_time_series_df):
        """Test that decomposition components sum to original."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        # Trend + Seasonal + Residual should equal Observed
        reconstructed = decomp_df['trend'] + decomp_df['seasonal'] + decomp_df['resid']
        np.testing.assert_array_almost_equal(
            reconstructed.values,
            decomp_df['observed'].values,
            decimal=10
        )

    def test_metadata_values(self, sample_time_series_df):
        """Test that metadata values are in valid ranges."""
        _, metadata = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        # Seasonal and trend strength should be between 0 and 1
        assert 0 <= metadata['seasonal_strength'] <= 1
        assert 0 <= metadata['trend_strength'] <= 1

        # Period should match input
        assert metadata['period'] == 24


class TestPlotSTLComponents:
    """Tests for plot_stl_components function."""

    def test_basic_plot(self, sample_time_series_df):
        """Test basic STL plot creation."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        fig = plot_stl_components(decomp_df)

        assert isinstance(fig, plt.Figure)
        # Check that we have 4 subplots
        assert len(fig.axes) == 4

        plt.close(fig)

    def test_custom_title(self, sample_time_series_df):
        """Test STL plot with custom title."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        custom_title = "Custom STL Decomposition"
        fig = plot_stl_components(decomp_df, title=custom_title)

        assert custom_title in fig._suptitle.get_text()
        plt.close(fig)


class TestCalculateACFPACF:
    """Tests for calculate_acf_pacf function."""

    def test_basic_calculation(self, sample_series):
        """Test basic ACF/PACF calculation."""
        acf_vals, acf_ci, pacf_vals, pacf_ci = calculate_acf_pacf(
            sample_series,
            nlags=20
        )

        # Check output shapes
        assert len(acf_vals) == 21  # nlags + 1
        assert len(pacf_vals) == 21

        # ACF at lag 0 should be 1
        assert np.isclose(acf_vals[0], 1.0)

    def test_custom_nlags(self, sample_series):
        """Test ACF/PACF with custom number of lags."""
        nlags = 30
        acf_vals, acf_ci, pacf_vals, pacf_ci = calculate_acf_pacf(
            sample_series,
            nlags=nlags
        )

        assert len(acf_vals) == nlags + 1
        assert len(pacf_vals) == nlags + 1

    def test_confidence_intervals(self, sample_series):
        """Test that confidence intervals are returned."""
        acf_vals, acf_ci, pacf_vals, pacf_ci = calculate_acf_pacf(
            sample_series,
            nlags=10
        )

        # Confidence intervals should be 2D arrays
        assert acf_ci.ndim == 2
        assert pacf_ci.ndim == 2


class TestPlotACFPACF:
    """Tests for plot_acf_pacf function."""

    def test_basic_plot(self, sample_series):
        """Test basic ACF/PACF plot creation."""
        fig = plot_acf_pacf(sample_series, nlags=20)

        assert isinstance(fig, plt.Figure)
        # Should have 2 subplots (ACF and PACF)
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_custom_nlags(self, sample_series):
        """Test ACF/PACF plot with custom lags."""
        fig = plot_acf_pacf(sample_series, nlags=30)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAnalyzeSeasonalityByGroup:
    """Tests for analyze_seasonality_by_group function."""

    def test_basic_analysis(self, sample_time_series_df):
        """Test basic seasonality analysis by group."""
        results = analyze_seasonality_by_group(
            sample_time_series_df,
            group_column='category',
            value_column='value',
            time_column='date',
            period=24
        )

        # Should have results for both categories
        assert 'A' in results
        assert 'B' in results

        # Each result should have metadata
        for group, metadata in results.items():
            if 'error' not in metadata:
                assert 'seasonal_strength' in metadata
                assert 'trend_strength' in metadata

    def test_group_metadata(self, sample_time_series_df):
        """Test that group metadata is valid."""
        results = analyze_seasonality_by_group(
            sample_time_series_df,
            group_column='category',
            value_column='value',
            time_column='date',
            period=24
        )

        for group, metadata in results.items():
            if 'error' not in metadata:
                # Check value ranges
                assert 0 <= metadata['seasonal_strength'] <= 1
                assert 0 <= metadata['trend_strength'] <= 1


class TestPlotSeasonalityComparison:
    """Tests for plot_seasonality_comparison function."""

    def test_basic_plot(self, sample_time_series_df):
        """Test basic seasonality comparison plot."""
        # First get seasonality results
        results = analyze_seasonality_by_group(
            sample_time_series_df,
            group_column='category',
            value_column='value',
            time_column='date',
            period=24
        )

        fig = plot_seasonality_comparison(results, metric='seasonal_strength')

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_metric(self, sample_time_series_df):
        """Test comparison plot with custom metric."""
        results = analyze_seasonality_by_group(
            sample_time_series_df,
            group_column='category',
            value_column='value',
            time_column='date',
            period=24
        )

        fig = plot_seasonality_comparison(results, metric='trend_strength')

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestExtractSeasonalPattern:
    """Tests for extract_seasonal_pattern function."""

    def test_basic_extraction(self, sample_time_series_df):
        """Test basic seasonal pattern extraction."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        pattern = extract_seasonal_pattern(decomp_df, period=24)

        # Check output
        assert isinstance(pattern, pd.DataFrame)
        assert 'period_index' in pattern.columns
        assert 'seasonal_value' in pattern.columns
        assert len(pattern) == 24

    def test_pattern_values(self, sample_time_series_df):
        """Test that extracted pattern has valid values."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        pattern = extract_seasonal_pattern(decomp_df, period=24)

        # All values should be finite
        assert np.all(np.isfinite(pattern['seasonal_value']))


class TestPlotSeasonalPattern:
    """Tests for plot_seasonal_pattern function."""

    def test_basic_plot(self, sample_time_series_df):
        """Test basic seasonal pattern plot."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        pattern = extract_seasonal_pattern(decomp_df, period=24)
        fig = plot_seasonal_pattern(pattern, period_label='Hour of Day')

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels(self, sample_time_series_df):
        """Test pattern plot with custom labels."""
        decomp_df, _ = perform_stl_decomposition(
            sample_time_series_df,
            time_column='date',
            value_column='value',
            period=24
        )

        pattern = extract_seasonal_pattern(decomp_df, period=24)
        fig = plot_seasonal_pattern(
            pattern,
            period_label='Hour',
            title='Custom Title'
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
