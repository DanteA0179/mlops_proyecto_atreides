"""
Unit Tests for Temporal Feature Engineering

Tests for functions in src/utils/temporal_features.py
"""

import numpy as np
import polars as pl
import pytest

from src.utils.temporal_features import (
    DAY_NAME_TO_NUMBER,
    create_all_temporal_features,
    create_cyclical_encoding,
    create_is_weekend,
    extract_day_of_week_numeric,
    extract_hour_from_nsm,
    validate_temporal_features,
)


@pytest.fixture
def sample_df_with_nsm():
    """Create sample dataframe with NSM column."""
    return pl.DataFrame(
        {
            "NSM": [0, 3600, 7200, 43200, 86399],  # 0h, 1h, 2h, 12h, 23h59m59s
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def sample_df_with_days():
    """Create sample dataframe with Day_of_week column."""
    return pl.DataFrame(
        {
            "Day_of_week": ["Monday", "Tuesday", "Friday", "Saturday", "Sunday"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def sample_df_complete():
    """Create complete sample dataframe for integration tests."""
    return pl.DataFrame(
        {
            "NSM": [3600, 43200, 72000],  # 1h, 12h, 20h
            "Day_of_week": ["Monday", "Friday", "Saturday"],
            "Usage_kWh": [10.0, 20.0, 30.0],
        }
    )


class TestExtractHour:
    """Tests for extract_hour_from_nsm function."""

    def test_extract_hour_basic(self, sample_df_with_nsm):
        """Test basic hour extraction."""
        result = extract_hour_from_nsm(sample_df_with_nsm)

        assert "hour" in result.columns
        assert result["hour"].to_list() == [0, 1, 2, 12, 23]

    def test_extract_hour_range_validation(self, sample_df_with_nsm):
        """Test that extracted hours are in valid range [0, 23]."""
        result = extract_hour_from_nsm(sample_df_with_nsm)

        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23

    def test_extract_hour_edge_cases(self):
        """Test edge cases: midnight and end of day."""
        df = pl.DataFrame({"NSM": [0, 86399]})  # 00:00:00 and 23:59:59
        result = extract_hour_from_nsm(df)

        assert result["hour"].to_list() == [0, 23]

    def test_extract_hour_with_nulls(self):
        """Test handling of null values."""
        df = pl.DataFrame({"NSM": [0, None, 7200]})
        result = extract_hour_from_nsm(df)

        assert result["hour"][1] is None
        assert result["hour"].to_list() == [0, None, 2]

    def test_extract_hour_invalid_column(self):
        """Test error when NSM column doesn't exist."""
        df = pl.DataFrame({"other_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'NSM' not found"):
            extract_hour_from_nsm(df)

    def test_extract_hour_invalid_range(self):
        """Test error when NSM values are out of range."""
        df = pl.DataFrame({"NSM": [0, 86401]})  # One value exceeds 86400

        with pytest.raises(ValueError, match="NSM values out of valid range"):
            extract_hour_from_nsm(df)


class TestDayOfWeekConversion:
    """Tests for extract_day_of_week_numeric function."""

    def test_day_of_week_mapping(self, sample_df_with_days):
        """Test correct mapping of day names to numbers."""
        result = extract_day_of_week_numeric(sample_df_with_days)

        assert "day_of_week" in result.columns
        assert result["day_of_week"].to_list() == [0, 1, 4, 5, 6]  # Mon, Tue, Fri, Sat, Sun

    def test_day_of_week_all_days(self):
        """Test mapping for all 7 days of week."""
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
        result = extract_day_of_week_numeric(df)

        assert result["day_of_week"].to_list() == [0, 1, 2, 3, 4, 5, 6]

    def test_day_of_week_invalid_input(self):
        """Test error with invalid day name."""
        df = pl.DataFrame({"Day_of_week": ["Monday", "InvalidDay"]})

        with pytest.raises(ValueError, match="Invalid day names found"):
            extract_day_of_week_numeric(df)

    def test_day_of_week_missing_column(self):
        """Test error when Day_of_week column doesn't exist."""
        df = pl.DataFrame({"other_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'Day_of_week' not found"):
            extract_day_of_week_numeric(df)


class TestIsWeekend:
    """Tests for create_is_weekend function."""

    def test_is_weekend_saturday_sunday(self):
        """Test that Saturday and Sunday are marked as weekend."""
        df = pl.DataFrame({"day_of_week": [5, 6]})  # Saturday, Sunday
        result = create_is_weekend(df)

        assert result["is_weekend"].to_list() == [True, True]

    def test_is_weekend_weekdays(self):
        """Test that Monday-Friday are not weekend."""
        df = pl.DataFrame({"day_of_week": [0, 1, 2, 3, 4]})  # Mon-Fri
        result = create_is_weekend(df)

        assert result["is_weekend"].to_list() == [False, False, False, False, False]

    def test_is_weekend_boolean_dtype(self):
        """Test that is_weekend column is Boolean type."""
        df = pl.DataFrame({"day_of_week": [0, 5, 6]})
        result = create_is_weekend(df)

        assert result["is_weekend"].dtype == pl.Boolean

    def test_is_weekend_invalid_range(self):
        """Test error when day_of_week is out of range."""
        df = pl.DataFrame({"day_of_week": [0, 7]})  # 7 is invalid

        with pytest.raises(ValueError, match="day_of_week values out of valid range"):
            create_is_weekend(df)


class TestCyclicalEncoding:
    """Tests for create_cyclical_encoding function."""

    def test_cyclical_hour_sin_cos(self):
        """Test cyclical encoding creates sin and cos columns."""
        df = pl.DataFrame({"hour": [0, 6, 12, 18]})
        result = create_cyclical_encoding(df, "hour", period=24)

        assert "cyclical_hour_sin" in result.columns
        assert "cyclical_hour_cos" in result.columns

    def test_cyclical_day_sin_cos(self):
        """Test cyclical encoding for day_of_week."""
        df = pl.DataFrame({"day_of_week": [0, 3, 6]})
        result = create_cyclical_encoding(df, "day_of_week", period=7)

        assert "cyclical_day_of_week_sin" in result.columns
        assert "cyclical_day_of_week_cos" in result.columns

    def test_cyclical_range_validation(self):
        """Test that cyclical values are in range [-1, 1]."""
        df = pl.DataFrame({"hour": list(range(24))})
        result = create_cyclical_encoding(df, "hour", period=24)

        assert result["cyclical_hour_sin"].min() >= -1.0
        assert result["cyclical_hour_sin"].max() <= 1.0
        assert result["cyclical_hour_cos"].min() >= -1.0
        assert result["cyclical_hour_cos"].max() <= 1.0

    def test_cyclical_continuity(self):
        """Test that hour 23 and hour 0 have similar encodings (continuity)."""
        df = pl.DataFrame({"hour": [0, 23]})
        result = create_cyclical_encoding(df, "hour", period=24)

        cos_0 = result["cyclical_hour_cos"][0]
        cos_23 = result["cyclical_hour_cos"][1]

        # cos(0) and cos(23) should be close (both near 1.0)
        assert abs(cos_0 - cos_23) < 0.3  # Allow some tolerance

    def test_cyclical_orthogonality(self):
        """Test that sin² + cos² = 1 for all values."""
        df = pl.DataFrame({"hour": list(range(24))})
        result = create_cyclical_encoding(df, "hour", period=24)

        # Calculate sin² + cos²
        norm = result["cyclical_hour_sin"] ** 2 + result["cyclical_hour_cos"] ** 2

        # All values should be approximately 1.0
        assert (norm - 1.0).abs().max() < 0.0001

    def test_cyclical_invalid_period(self):
        """Test error with invalid period."""
        df = pl.DataFrame({"hour": [0, 1, 2]})

        with pytest.raises(ValueError, match="Period must be positive"):
            create_cyclical_encoding(df, "hour", period=0)

    def test_cyclical_missing_column(self):
        """Test error when column doesn't exist."""
        df = pl.DataFrame({"other": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'hour' not found"):
            create_cyclical_encoding(df, "hour", period=24)


class TestFeatureValidation:
    """Tests for validate_temporal_features function."""

    def test_validate_all_features_present(self, sample_df_complete):
        """Test validation passes when all features are present."""
        df_featured = create_all_temporal_features(sample_df_complete)
        validation = validate_temporal_features(df_featured)

        assert validation["valid"] is True
        assert len(validation["missing_features"]) == 0

    def test_validate_feature_types(self, sample_df_complete):
        """Test validation checks correct data types."""
        df_featured = create_all_temporal_features(sample_df_complete)
        validation = validate_temporal_features(df_featured)

        assert validation["valid"] is True
        assert df_featured["is_weekend"].dtype == pl.Boolean

    def test_validate_missing_features_raises(self):
        """Test validation fails when features are missing."""
        df = pl.DataFrame({"some_col": [1, 2, 3]})
        validation = validate_temporal_features(df)

        assert validation["valid"] is False
        assert len(validation["missing_features"]) > 0

    def test_validate_invalid_ranges(self):
        """Test validation detects invalid ranges."""
        df = pl.DataFrame(
            {
                "hour": [0, 25],  # 25 is invalid
                "day_of_week": [0, 1],
                "is_weekend": [False, False],
                "cyclical_hour_sin": [0.0, 0.0],
                "cyclical_hour_cos": [1.0, 1.0],
                "cyclical_day_sin": [0.0, 0.0],
                "cyclical_day_cos": [1.0, 1.0],
            }
        )
        validation = validate_temporal_features(df)

        assert validation["valid"] is False
        assert "hour" in validation["invalid_ranges"]


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_on_sample_data(self, sample_df_complete):
        """Test complete pipeline creates all 7 features."""
        result = create_all_temporal_features(sample_df_complete)

        # Check all 7 new features are created
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
            assert feat in result.columns

        # Check total columns: 3 original + 7 new = 10
        assert len(result.columns) == 10

    def test_pipeline_preserves_original_columns(self, sample_df_complete):
        """Test that original columns are preserved."""
        result = create_all_temporal_features(sample_df_complete)

        assert "NSM" in result.columns
        assert "Day_of_week" in result.columns
        assert "Usage_kWh" in result.columns

    def test_pipeline_output_schema(self, sample_df_complete):
        """Test output schema matches expected types."""
        result = create_all_temporal_features(sample_df_complete)

        assert result["hour"].dtype == pl.Int32
        assert result["day_of_week"].dtype == pl.Int32
        assert result["is_weekend"].dtype == pl.Boolean
        assert result["cyclical_hour_sin"].dtype == pl.Float64
        assert result["cyclical_hour_cos"].dtype == pl.Float64
        assert result["cyclical_day_sin"].dtype == pl.Float64
        assert result["cyclical_day_cos"].dtype == pl.Float64

    def test_pipeline_validation_passes(self, sample_df_complete):
        """Test that pipeline output passes validation."""
        result = create_all_temporal_features(sample_df_complete)
        validation = validate_temporal_features(result)

        assert validation["valid"] is True
        assert len(validation["missing_features"]) == 0
        assert len(validation["invalid_ranges"]) == 0

    def test_pipeline_with_large_dataset(self):
        """Test pipeline performance with larger dataset."""
        # Create dataset with 1000 rows
        df = pl.DataFrame(
            {
                "NSM": np.random.randint(0, 86400, 1000),
                "Day_of_week": np.random.choice(list(DAY_NAME_TO_NUMBER.keys()), 1000),
                "Usage_kWh": np.random.uniform(5, 50, 1000),
            }
        )

        result = create_all_temporal_features(df)

        assert len(result) == 1000
        assert len(result.columns) == 10

        # Validate output
        validation = validate_temporal_features(result)
        assert validation["valid"] is True
