"""
Tests for EDA plotting utilities

Unit tests for src/utils/eda_plots.py functions.
"""

import plotly.graph_objects as go
import polars as pl
import pytest

from src.utils.eda_plots import (
    get_top_correlated_features,
    plot_box_by_category,
    plot_correlation_heatmap,
    plot_distribution,
    plot_scatter,
    plot_scatter_matrix,
    plot_time_series,
)


@pytest.fixture
def sample_df():
    """
    Create a sample DataFrame for testing.

    Returns
    -------
    pl.DataFrame
        Sample DataFrame with numeric and categorical columns
    """
    return pl.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "feature3": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "hour": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )


class TestPlotDistribution:
    """Tests for plot_distribution function."""

    def test_basic_distribution(self, sample_df):
        """Test basic distribution plot creation."""
        fig = plot_distribution(sample_df, "feature1")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_title(self, sample_df):
        """Test distribution plot with custom title."""
        custom_title = "Custom Distribution"
        fig = plot_distribution(sample_df, "feature1", title=custom_title)

        assert custom_title in fig.layout.title.text

    def test_custom_nbins(self, sample_df):
        """Test distribution plot with custom number of bins."""
        fig = plot_distribution(sample_df, "feature1", nbins=20)

        assert isinstance(fig, go.Figure)

    def test_plot_distribution_missing_column(self, sample_df):
        """Prueba que la función falla si la columna no existe."""
        with pytest.raises(pl.ColumnNotFoundError):
            plot_distribution(sample_df, "columna_fantasma")

    def test_plot_distribution_empty_df(self):
        """Prueba que la función maneja un DataFrame vacío sin fallar."""
        empty_df = pl.DataFrame({"feature1": []})
        fig = plot_distribution(empty_df, "feature1")

        assert isinstance(fig, go.Figure)
        # Los datos del gráfico estarán vacíos
        assert len(fig.data[0].x) == 0

    def test_plot_distribution_categorical_column(self, sample_df):
        """
        Prueba cómo la función maneja una columna categórica.
        (Idealmente, debería crear un histograma de conteo de categorías)
        """
        fig = plot_distribution(sample_df, "category")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Verifica que el eje x tenga las categorías A y B
        assert "A" in fig.data[0].x
        assert "B" in fig.data[0].x


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap function."""

    def test_basic_heatmap(self, sample_df):
        """Test basic correlation heatmap creation."""
        fig = plot_correlation_heatmap(sample_df)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_columns(self, sample_df):
        """Test heatmap with specific columns."""
        columns = ["feature1", "feature2", "target"]
        fig = plot_correlation_heatmap(sample_df, columns=columns)

        assert isinstance(fig, go.Figure)

    def test_custom_colorscale(self, sample_df):
        """Test heatmap with custom colorscale."""
        fig = plot_correlation_heatmap(sample_df, colorscale="Viridis")

        assert isinstance(fig, go.Figure)

    def test_plot_heatmap_empty_df(self):
        """Prueba que el heatmap maneja un DataFrame vacío."""
        empty_df = pl.DataFrame({"feature1": [], "feature2": []})
        fig = plot_correlation_heatmap(empty_df)

        assert isinstance(fig, go.Figure)

    def test_plot_heatmap_all_categorical(self, sample_df):
        """
        Prueba que el heatmap solo usa columnas numéricas
        (ignorando 'category').
        """
        # Creamos un DF que solo tiene 'category' y 'feature1'
        subset_df = sample_df.select(["category", "feature1"])
        fig = plot_correlation_heatmap(subset_df)

        assert isinstance(fig, go.Figure)
        # El heatmap resultante solo debe tener 'feature1'
        assert fig.data[0].x == ["feature1"]
        assert fig.data[0].y == ["feature1"]


class TestPlotTimeSeries:
    """Tests for plot_time_series function."""

    def test_basic_time_series(self, sample_df):
        """Test basic time series plot creation."""
        fig = plot_time_series(sample_df, "hour", "target")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_aggregation_mean(self, sample_df):
        """Test time series with mean aggregation."""
        fig = plot_time_series(sample_df, "hour", "target", aggregation="mean")

        assert isinstance(fig, go.Figure)

    def test_aggregation_sum(self, sample_df):
        """Test time series with sum aggregation."""
        fig = plot_time_series(sample_df, "hour", "target", aggregation="sum")

        assert isinstance(fig, go.Figure)


class TestPlotBoxByCategory:
    """Tests for plot_box_by_category function."""

    def test_basic_box_plot(self, sample_df):
        """Test basic box plot creation."""
        fig = plot_box_by_category(sample_df, "category", "target")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_custom_colors(self, sample_df):
        """Test box plot with custom colors."""
        colors = ["#FF0000", "#00FF00"]
        fig = plot_box_by_category(sample_df, "category", "target", color_discrete_sequence=colors)

        assert isinstance(fig, go.Figure)

        # Puedes añadir esto a TestPlotBoxByCategory

    def test_plot_box_axis_labels(self, sample_df):
        """Prueba que los títulos de los ejes X e Y son correctos."""
        fig = plot_box_by_category(sample_df, "category", "target")

        assert fig.layout.xaxis.title.text == "category"
        assert fig.layout.yaxis.title.text == "target"


class TestPlotScatter:
    """Tests for plot_scatter function."""

    def test_basic_scatter(self, sample_df):
        """Test basic scatter plot creation."""
        fig = plot_scatter(sample_df, "feature1", "target")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_scatter_with_color(self, sample_df):
        """Test scatter plot with color coding."""
        fig = plot_scatter(sample_df, "feature1", "target", color_column="category")

        assert isinstance(fig, go.Figure)

    def test_scatter_with_trendline(self, sample_df):
        """Test scatter plot with trendline."""
        fig = plot_scatter(sample_df, "feature1", "target", trendline=True)

        assert isinstance(fig, go.Figure)
        # Trendline adds an extra trace
        assert len(fig.data) >= 1

    def test_plot_scatter_missing_column(self, sample_df):
        """Prueba que la función falla si una columna no existe."""
        with pytest.raises(pl.ColumnNotFoundError):
            plot_scatter(sample_df, "feature1", "columna_fantasma")

    def test_plot_scatter_axis_labels(self, sample_df):
        """Prueba que los títulos de los ejes X e Y son correctos."""
        fig = plot_scatter(sample_df, "feature1", "target")

        assert fig.layout.xaxis.title.text == "feature1"
        assert fig.layout.yaxis.title.text == "target"


class TestPlotScatterMatrix:
    """Tests for plot_scatter_matrix function."""

    def test_basic_scatter_matrix(self, sample_df):
        """Test basic scatter matrix creation."""
        columns = ["feature1", "feature2", "target"]
        fig = plot_scatter_matrix(sample_df, columns)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_scatter_matrix_with_color(self, sample_df):
        """Test scatter matrix with color coding."""
        columns = ["feature1", "feature2", "target"]
        fig = plot_scatter_matrix(sample_df, columns, color_column="category")

        assert isinstance(fig, go.Figure)


class TestGetTopCorrelatedFeatures:
    """Tests for get_top_correlated_features function."""

    def test_basic_correlation(self, sample_df):
        """Test getting top correlated features."""
        top_features = get_top_correlated_features(sample_df, "target", n=2)

        assert isinstance(top_features, list)
        assert len(top_features) == 2
        assert all(isinstance(item, tuple) for item in top_features)
        assert all(len(item) == 2 for item in top_features)

    def test_correlation_values(self, sample_df):
        """Test that correlation values are valid."""
        top_features = get_top_correlated_features(sample_df, "target", n=3)

        for feature, corr in top_features:
            assert isinstance(feature, str)
            assert isinstance(corr, float)
            assert -1.0 <= corr <= 1.0

    def test_exclude_columns(self, sample_df):
        """Test excluding specific columns."""
        top_features = get_top_correlated_features(
            sample_df, "target", n=2, exclude_columns=["feature3"]
        )

        # Verify feature3 is not in results
        feature_names = [feat[0] for feat in top_features]
        assert "feature3" not in feature_names

    def test_correlation_exact_values(self, sample_df):
        """Prueba que los valores de correlación calculados son correctos."""
        # Obtenemos las 2 mejores correlaciones con 'target'
        top_features = get_top_correlated_features(sample_df, "target", n=2)

        # Sabemos que 'feature1' y 'feature2' están perfectamente correlacionados
        # con 'target' en los datos de muestra.

        # Convertimos la lista de tuplas a un diccionario para facilitar la comprobación
        top_dict = dict(top_features)

        # Comprobamos los valores exactos
        assert "feature1" in top_dict
        assert top_dict["feature1"] == 1.0

        assert "feature2" in top_dict
        assert top_dict["feature2"] == 1.0


# Run tests with: pytest tests/test_eda_plots.py -v
