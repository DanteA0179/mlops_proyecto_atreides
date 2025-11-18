"""
Unit Tests for Feature Importance Analysis

Tests para las funciones de análisis de importancia de features en
src/utils/feature_importance.py
"""

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for tests
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.feature_importance import (
    calculate_mutual_information,
    calculate_pearson_correlation,
    compare_importance_methods,
    get_top_features,
    plot_feature_importance,
)


@pytest.fixture
def sample_df():
    """Create sample dataframe for testing with 5 rows and 3 features + target."""
    return pl.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "feature3": [2.0, 2.5, 3.0, 3.5, 4.0],
            "Usage_kWh": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture
def sample_mi_scores():
    """Create sample mutual information scores for testing."""
    return pl.DataFrame(
        {"feature": ["feature1", "feature3", "feature2"], "mi_score": [2.5, 1.8, 0.5]}
    )


@pytest.fixture
def sample_correlations():
    """Create sample Pearson correlations for testing."""
    return pl.DataFrame(
        {
            "feature": ["feature1", "feature2", "feature3"],
            "correlation": [0.95, -0.85, 0.75],
            "abs_correlation": [0.95, 0.85, 0.75],
        }
    )


class TestMutualInformation:
    """Tests for mutual information calculation."""

    def test_calculate_mi_basic(self, sample_df):
        """Test basic MI calculation functionality."""
        result = calculate_mutual_information(sample_df, target_column="Usage_kWh")

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "mi_score" in result.columns
        assert len(result) == 3  # 3 features (excluding target)

    def test_calculate_mi_scores_non_negative(self, sample_df):
        """Test that MI scores are non-negative."""
        result = calculate_mutual_information(sample_df, target_column="Usage_kWh")

        # All MI scores should be >= 0
        assert all(result["mi_score"] >= 0)

    def test_calculate_mi_sorted_descending(self, sample_df):
        """Test that results are sorted by MI score descending."""
        result = calculate_mutual_information(sample_df, target_column="Usage_kWh")

        mi_scores = result["mi_score"].to_list()
        # Check that each score is >= the next one
        for i in range(len(mi_scores) - 1):
            assert mi_scores[i] >= mi_scores[i + 1]

    def test_calculate_mi_handles_nulls(self):
        """Test null handling by dropping rows with nulls."""
        df_with_nulls = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, None, 4.0, 5.0],
                "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "Usage_kWh": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        result = calculate_mutual_information(df_with_nulls, target_column="Usage_kWh")

        # Should complete without error
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2  # 2 features

    def test_calculate_mi_reproducible(self, sample_df):
        """Test that same random_state gives same results."""
        result1 = calculate_mutual_information(
            sample_df, target_column="Usage_kWh", random_state=42
        )
        result2 = calculate_mutual_information(
            sample_df, target_column="Usage_kWh", random_state=42
        )

        # Results should be identical
        assert result1["feature"].to_list() == result2["feature"].to_list()
        assert result1["mi_score"].to_list() == result2["mi_score"].to_list()

    def test_calculate_mi_excludes_target(self, sample_df):
        """Test that target variable is not in results."""
        result = calculate_mutual_information(sample_df, target_column="Usage_kWh")

        # Target should not appear in feature list
        assert "Usage_kWh" not in result["feature"].to_list()


class TestPearsonCorrelation:
    """Tests for Pearson correlation calculation."""

    def test_calculate_corr_basic(self, sample_df):
        """Test basic correlation calculation functionality."""
        result = calculate_pearson_correlation(sample_df, target_column="Usage_kWh")

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "correlation" in result.columns
        assert "abs_correlation" in result.columns
        assert len(result) == 3  # 3 features (excluding target)

    def test_calculate_corr_range(self, sample_df):
        """Test that correlations are between -1 and 1."""
        result = calculate_pearson_correlation(sample_df, target_column="Usage_kWh")

        # All correlations should be between -1 and 1
        correlations = result["correlation"].to_list()
        for corr in correlations:
            assert -1 <= corr <= 1

    def test_calculate_corr_excludes_target(self, sample_df):
        """Test that target variable is not in results."""
        result = calculate_pearson_correlation(sample_df, target_column="Usage_kWh")

        # Target should not appear in feature list
        assert "Usage_kWh" not in result["feature"].to_list()

    def test_calculate_corr_sorted_by_abs(self, sample_df):
        """Test that results are sorted by absolute correlation descending."""
        result = calculate_pearson_correlation(sample_df, target_column="Usage_kWh")

        abs_correlations = result["abs_correlation"].to_list()
        # Check that each abs correlation is >= the next one
        for i in range(len(abs_correlations) - 1):
            assert abs_correlations[i] >= abs_correlations[i + 1]

    def test_calculate_corr_has_abs_column(self, sample_df):
        """Test that abs_correlation column exists and is correct."""
        result = calculate_pearson_correlation(sample_df, target_column="Usage_kWh")

        # Verify abs_correlation is absolute value of correlation
        for i in range(len(result)):
            corr = result["correlation"][i]
            abs_corr = result["abs_correlation"][i]
            assert abs_corr == abs(corr)


class TestVisualization:
    """Tests for plotting functions."""

    def test_plot_returns_figure(self, sample_mi_scores):
        """Test that function returns matplotlib Figure."""
        fig = plot_feature_importance(
            sample_mi_scores, score_column="mi_score", method_name="Mutual Information", top_n=3
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_to_file(self, sample_mi_scores):
        """Test that file is created at output_path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_plot.png"

            fig = plot_feature_importance(
                sample_mi_scores,
                score_column="mi_score",
                method_name="Mutual Information",
                top_n=3,
                output_path=str(output_path),
            )

            # Verify file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)

    def test_plot_color_by_sign(self, sample_correlations):
        """Test that color coding works for correlations."""
        fig = plot_feature_importance(
            sample_correlations,
            score_column="correlation",
            method_name="Pearson Correlation",
            top_n=3,
            color_by_sign=True,
        )

        # Should complete without error
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_top_n_limit(self, sample_mi_scores):
        """Test that plot respects top_n parameter."""
        fig = plot_feature_importance(
            sample_mi_scores, score_column="mi_score", method_name="Mutual Information", top_n=2
        )

        # Get the axes and check number of bars
        ax = fig.axes[0]
        # Number of bars should be 2
        assert len(ax.patches) == 2
        plt.close(fig)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_top_features(self, sample_mi_scores):
        """Test that function returns correct list of feature names."""
        top_features = get_top_features(sample_mi_scores, score_column="mi_score", top_n=2)

        assert isinstance(top_features, list)
        assert len(top_features) == 2
        assert top_features[0] == "feature1"  # Highest MI score
        assert top_features[1] == "feature3"  # Second highest

    def test_get_top_features_limit(self, sample_mi_scores):
        """Test that function respects top_n parameter."""
        top_features = get_top_features(sample_mi_scores, score_column="mi_score", top_n=1)

        assert len(top_features) == 1
        assert top_features[0] == "feature1"

    def test_compare_methods(self, sample_mi_scores, sample_correlations):
        """Test that comparison returns correct dictionary structure."""
        comparison = compare_importance_methods(sample_mi_scores, sample_correlations, top_n=2)

        assert isinstance(comparison, dict)
        assert "common_features" in comparison
        assert "mi_only" in comparison
        assert "corr_only" in comparison
        assert "overlap_percentage" in comparison
        assert "n_common" in comparison
        assert "n_mi_only" in comparison
        assert "n_corr_only" in comparison

    def test_compare_methods_overlap(self, sample_mi_scores, sample_correlations):
        """Test that overlap percentage is calculated correctly."""
        comparison = compare_importance_methods(sample_mi_scores, sample_correlations, top_n=2)

        # Top 2 MI: feature1, feature3
        # Top 2 Corr: feature1, feature2
        # Common: feature1 (1 out of 2 = 50%)
        assert comparison["n_common"] == 1
        assert comparison["overlap_percentage"] == 50.0
        assert "feature1" in comparison["common_features"]
        assert "feature3" in comparison["mi_only"]
        assert "feature2" in comparison["corr_only"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pl.DataFrame({"feature1": [], "Usage_kWh": []})

        with pytest.raises(ValueError):
            calculate_mutual_information(empty_df, target_column="Usage_kWh")

    def test_single_feature(self):
        """Test that function works with single feature."""
        single_feature_df = pl.DataFrame(
            {"feature1": [1.0, 2.0, 3.0, 4.0, 5.0], "Usage_kWh": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )

        result = calculate_mutual_information(single_feature_df, target_column="Usage_kWh")

        assert len(result) == 1
        assert result["feature"][0] == "feature1"

    def test_all_nulls(self):
        """Test handling of all null values."""
        all_nulls_df = pl.DataFrame(
            {"feature1": [None, None, None], "Usage_kWh": [10.0, 20.0, 30.0]}
        )

        # Null columns are not considered numeric, so this raises "No numeric features"
        with pytest.raises(ValueError, match="No numeric features found"):
            calculate_mutual_information(all_nulls_df, target_column="Usage_kWh")

    def test_missing_target_column(self):
        """Test that appropriate error is raised for missing target."""
        df = pl.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="Target column 'Usage_kWh' not found"):
            calculate_mutual_information(df, target_column="Usage_kWh")

        with pytest.raises(ValueError, match="Target column 'Usage_kWh' not found"):
            calculate_pearson_correlation(df, target_column="Usage_kWh")

    def test_no_numeric_features(self):
        """Test handling of no numeric columns."""
        non_numeric_df = pl.DataFrame(
            {
                "feature1": ["a", "b", "c"],
                "feature2": ["x", "y", "z"],
                "Usage_kWh": [10.0, 20.0, 30.0],
            }
        )

        with pytest.raises(ValueError, match="No numeric features found"):
            calculate_mutual_information(non_numeric_df, target_column="Usage_kWh")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
