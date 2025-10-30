"""
Tests for temporal feature transformers (POO approach).

This module tests sklearn-compatible transformers for temporal feature engineering.
"""

import numpy as np
import polars as pl
import pytest

from src.features.temporal_transformers import (
    CyclicalEncoder,
    DayOfWeekEncoder,
    HourExtractor,
    TemporalFeatureEngineer,
    WeekendIndicator,
)


class TestHourExtractor:
    """Tests for HourExtractor transformer."""

    def test_fit_transform_basic(self):
        """Test basic hour extraction."""
        df = pl.DataFrame({"NSM": [0, 3600, 7200, 43200]})

        transformer = HourExtractor()
        df_result = transformer.fit_transform(df)

        assert "hour" in df_result.columns
        assert df_result["hour"].to_list() == [0, 1, 2, 12]

    def test_edge_case_nsm_86400(self):
        """Test NSM=86400 maps to hour 23, not 24."""
        df = pl.DataFrame({"NSM": [0, 86399, 86400]})

        transformer = HourExtractor()
        df_result = transformer.fit_transform(df)

        assert df_result["hour"].to_list() == [0, 23, 23]

    def test_custom_column_names(self):
        """Test custom column names."""
        df = pl.DataFrame({"seconds": [0, 3600]})

        transformer = HourExtractor(nsm_col="seconds", output_col="hora")
        df_result = transformer.fit_transform(df)

        assert "hora" in df_result.columns
        assert df_result["hora"].to_list() == [0, 1]

    def test_drop_nsm_option(self):
        """Test drop_nsm option."""
        df = pl.DataFrame({"NSM": [0, 3600]})

        transformer = HourExtractor(drop_nsm=True)
        df_result = transformer.fit_transform(df)

        assert "NSM" not in df_result.columns
        assert "hour" in df_result.columns

    def test_missing_column_raises(self):
        """Test error when NSM column doesn't exist."""
        df = pl.DataFrame({"other": [0, 1]})

        transformer = HourExtractor()
        with pytest.raises(ValueError, match="Column 'NSM' not found"):
            transformer.fit(df)


class TestDayOfWeekEncoder:
    """Tests for DayOfWeekEncoder transformer."""

    def test_fit_transform_all_days(self):
        """Test encoding all days of week."""
        df = pl.DataFrame(
            {
                "Day_of_week": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
            }
        )

        transformer = DayOfWeekEncoder()
        df_result = transformer.fit_transform(df)

        assert df_result["day_of_week"].to_list() == [0, 1, 2, 3, 4, 5, 6]

    def test_custom_column_names(self):
        """Test custom column names."""
        df = pl.DataFrame({"dia": ["Monday", "Friday"]})

        transformer = DayOfWeekEncoder(day_col="dia", output_col="dia_num")
        df_result = transformer.fit_transform(df)

        assert "dia_num" in df_result.columns
        assert df_result["dia_num"].to_list() == [0, 4]

    def test_drop_original_option(self):
        """Test drop_original option."""
        df = pl.DataFrame({"Day_of_week": ["Monday", "Friday"]})

        transformer = DayOfWeekEncoder(drop_original=True)
        df_result = transformer.fit_transform(df)

        assert "Day_of_week" not in df_result.columns
        assert "day_of_week" in df_result.columns

    def test_missing_column_raises(self):
        """Test error when column doesn't exist."""
        df = pl.DataFrame({"other": ["Monday"]})

        transformer = DayOfWeekEncoder()
        with pytest.raises(ValueError, match="Column 'Day_of_week' not found"):
            transformer.fit(df)


class TestWeekendIndicator:
    """Tests for WeekendIndicator transformer."""

    def test_fit_transform_default_weekend(self):
        """Test default weekend days (5=Saturday, 6=Sunday)."""
        df = pl.DataFrame({"day_of_week": [0, 1, 5, 6]})  # Mon, Tue, Sat, Sun

        transformer = WeekendIndicator()
        df_result = transformer.fit_transform(df)

        assert df_result["is_weekend"].to_list() == [False, False, True, True]

    def test_custom_weekend_days(self):
        """Test custom weekend days."""
        df = pl.DataFrame({"day_of_week": [0, 4, 5, 6]})

        # Friday-Saturday as weekend
        transformer = WeekendIndicator(weekend_days=[4, 5])
        df_result = transformer.fit_transform(df)

        assert df_result["is_weekend"].to_list() == [False, True, True, False]

    def test_custom_column_names(self):
        """Test custom column names."""
        df = pl.DataFrame({"dia_num": [0, 5]})

        transformer = WeekendIndicator(day_col="dia_num", output_col="fin_semana")
        df_result = transformer.fit_transform(df)

        assert "fin_semana" in df_result.columns


class TestCyclicalEncoder:
    """Tests for CyclicalEncoder transformer."""

    def test_fit_transform_hour(self):
        """Test cyclical encoding for hours."""
        df = pl.DataFrame({"hour": [0, 6, 12, 18]})

        transformer = CyclicalEncoder(column="hour", period=24)
        df_result = transformer.fit_transform(df)

        assert "cyclical_hour_sin" in df_result.columns
        assert "cyclical_hour_cos" in df_result.columns

        # Check hour 0
        assert np.isclose(df_result["cyclical_hour_sin"][0], 0, atol=1e-10)
        assert np.isclose(df_result["cyclical_hour_cos"][0], 1, atol=1e-10)

        # Check hour 6 (90 degrees)
        assert np.isclose(df_result["cyclical_hour_sin"][1], 1, atol=1e-10)
        assert np.isclose(df_result["cyclical_hour_cos"][1], 0, atol=1e-10)

    def test_cyclical_orthogonality(self):
        """Test sin² + cos² = 1 for all values."""
        df = pl.DataFrame({"hour": list(range(24))})

        transformer = CyclicalEncoder(column="hour", period=24)
        df_result = transformer.fit_transform(df)

        # Calculate sin² + cos²
        norm = df_result["cyclical_hour_sin"] ** 2 + df_result["cyclical_hour_cos"] ** 2

        # All values should be approximately 1.0
        assert all(np.isclose(norm.to_list(), 1.0, atol=1e-10))

    def test_cyclical_continuity(self):
        """Test hour 23 is close to hour 0."""
        df = pl.DataFrame({"hour": [0, 23]})

        transformer = CyclicalEncoder(column="hour", period=24)
        df_result = transformer.fit_transform(df)

        # Hour 0 and hour 23 should have similar encoded values
        sin_0, cos_0 = df_result["cyclical_hour_sin"][0], df_result["cyclical_hour_cos"][0]
        sin_23, cos_23 = df_result["cyclical_hour_sin"][1], df_result["cyclical_hour_cos"][1]

        # Euclidean distance should be small
        distance = np.sqrt((sin_0 - sin_23) ** 2 + (cos_0 - cos_23) ** 2)
        assert distance < 0.5  # Much smaller than distance to hour 12

    def test_generic_usage_month(self):
        """Test generic usage for months."""
        df = pl.DataFrame({"month": [1, 3, 6, 9, 12]})

        transformer = CyclicalEncoder(column="month", period=12)
        df_result = transformer.fit_transform(df)

        assert "cyclical_month_sin" in df_result.columns
        assert "cyclical_month_cos" in df_result.columns

    def test_generic_usage_angle(self):
        """Test generic usage for compass directions."""
        df = pl.DataFrame({"wind_direction": [0, 90, 180, 270]})

        transformer = CyclicalEncoder(
            column="wind_direction", period=360, sin_col="wind_sin", cos_col="wind_cos"
        )
        df_result = transformer.fit_transform(df)

        assert "wind_sin" in df_result.columns
        assert "wind_cos" in df_result.columns

    def test_drop_original_option(self):
        """Test drop_original option."""
        df = pl.DataFrame({"hour": [0, 6]})

        transformer = CyclicalEncoder(column="hour", period=24, drop_original=True)
        df_result = transformer.fit_transform(df)

        assert "hour" not in df_result.columns

    def test_invalid_period_raises(self):
        """Test error when period is invalid."""
        df = pl.DataFrame({"hour": [0]})

        transformer = CyclicalEncoder(column="hour", period=0)
        with pytest.raises(ValueError, match="Period must be positive"):
            transformer.fit(df)


class TestTemporalFeatureEngineer:
    """Tests for TemporalFeatureEngineer (complete pipeline)."""

    def test_fit_transform_all_features(self):
        """Test creating all 7 temporal features."""
        df = pl.DataFrame(
            {
                "NSM": [0, 43200, 86399],
                "Day_of_week": ["Monday", "Wednesday", "Sunday"],
                "Usage_kWh": [10, 20, 15],
            }
        )

        engineer = TemporalFeatureEngineer()
        df_result = engineer.fit_transform(df)

        # Check all features created
        expected_features = [
            "hour",
            "day_of_week",
            "is_weekend",
            "cyclical_hour_sin",
            "cyclical_hour_cos",
            "cyclical_day_sin",
            "cyclical_day_cos",
        ]

        for feat in expected_features:
            assert feat in df_result.columns

        # Check total columns
        assert df_result.shape[1] == 3 + 7  # Original 3 + 7 new features

    def test_partial_feature_creation(self):
        """Test creating only some features."""
        df = pl.DataFrame({"NSM": [0, 43200], "Day_of_week": ["Monday", "Wednesday"]})

        # Only hour and cyclical, no weekend
        engineer = TemporalFeatureEngineer(
            create_hour=True, create_day=False, create_weekend=False, create_cyclical=True
        )
        df_result = engineer.fit_transform(df)

        assert "hour" in df_result.columns
        assert "day_of_week" not in df_result.columns
        assert "is_weekend" not in df_result.columns
        assert "cyclical_hour_sin" in df_result.columns
        assert "cyclical_hour_cos" in df_result.columns

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        engineer = TemporalFeatureEngineer()

        feature_names = engineer.get_feature_names_out()

        assert feature_names == [
            "hour",
            "day_of_week",
            "is_weekend",
            "cyclical_hour_sin",
            "cyclical_hour_cos",
            "cyclical_day_sin",
            "cyclical_day_cos",
        ]

    def test_sklearn_pipeline_compatible(self):
        """Test compatibility with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        df = pl.DataFrame(
            {"NSM": [0, 43200, 86399], "Day_of_week": ["Monday", "Wednesday", "Sunday"]}
        )

        # Create pipeline
        pipeline = Pipeline([("temporal", TemporalFeatureEngineer())])

        # Fit and transform
        df_result = pipeline.fit_transform(df)

        assert df_result.shape[1] == 2 + 7  # Original 2 + 7 new features

    def test_preserves_original_data(self):
        """Test that original data is preserved."""
        df = pl.DataFrame(
            {"NSM": [0, 43200], "Day_of_week": ["Monday", "Wednesday"], "Usage_kWh": [10, 20]}
        )

        engineer = TemporalFeatureEngineer()
        df_result = engineer.fit_transform(df)

        # Original columns should still exist
        assert "NSM" in df_result.columns
        assert "Day_of_week" in df_result.columns
        assert "Usage_kWh" in df_result.columns

        # Original values should be unchanged
        assert df_result["Usage_kWh"].to_list() == [10, 20]

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        n = 10000
        df = pl.DataFrame(
            {
                "NSM": np.random.randint(0, 86400, n),
                "Day_of_week": np.random.choice(
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    n,
                ),
            }
        )

        engineer = TemporalFeatureEngineer()
        df_result = engineer.fit_transform(df)

        assert df_result.shape[0] == n
        assert df_result.shape[1] == 2 + 7

    def test_output_column_names_match_expected(self):
        """Test that output column names match validation expectations."""
        df = pl.DataFrame({"NSM": [0, 43200], "Day_of_week": ["Monday", "Wednesday"]})

        engineer = TemporalFeatureEngineer()
        df_result = engineer.fit_transform(df)

        # These are the exact names expected by validation
        expected_names = [
            "cyclical_hour_sin",
            "cyclical_hour_cos",
            "cyclical_day_sin",
            "cyclical_day_cos",
        ]

        for name in expected_names:
            assert name in df_result.columns, f"Missing column: {name}"
