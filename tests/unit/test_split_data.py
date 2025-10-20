"""
Unit tests for split_data.py.

Tests for data splitting utilities including stratified and temporal splits.
"""

import numpy as np
import polars as pl
import pytest

from src.utils.split_data import (
    get_split_statistics,
    stratified_train_val_test_split,
    temporal_train_val_test_split,
    validate_splits,
)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    n_rows = 1000
    return pl.DataFrame(
        {
            "id": range(n_rows),
            "feature1": np.random.randn(n_rows),
            "feature2": np.random.rand(n_rows) * 100,
            "category": np.random.choice(["A", "B", "C"], n_rows),
            "target": np.random.rand(n_rows) * 50 + 10,  # Target between 10-60
            "date": pl.date_range(
                pl.date(2020, 1, 1), pl.date(2022, 9, 26), interval="1d", eager=True
            )[:n_rows],
        }
    )


class TestStratifiedTrainValTestSplit:
    """Tests for stratified_train_val_test_split function."""

    def test_basic_split(self, sample_dataframe):
        """Test basic split functionality."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        # Check that splits are non-empty
        assert len(df_train) > 0
        assert len(df_val) > 0
        assert len(df_test) > 0

        # Check total rows
        total_rows = len(df_train) + len(df_val) + len(df_test)
        assert total_rows == len(sample_dataframe)

    def test_split_ratios(self, sample_dataframe):
        """Test that split ratios are approximately correct."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe,
            target_col="target",
            train_size=0.70,
            val_size=0.15,
            test_size=0.15,
        )

        total = len(sample_dataframe)
        train_ratio = len(df_train) / total
        val_ratio = len(df_val) / total
        test_ratio = len(df_test) / total

        # Allow 2% tolerance
        assert abs(train_ratio - 0.70) < 0.02
        assert abs(val_ratio - 0.15) < 0.02
        assert abs(test_ratio - 0.15) < 0.02

    def test_custom_ratios(self, sample_dataframe):
        """Test custom split ratios."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe,
            target_col="target",
            train_size=0.60,
            val_size=0.20,
            test_size=0.20,
        )

        total = len(sample_dataframe)
        train_ratio = len(df_train) / total
        val_ratio = len(df_val) / total
        test_ratio = len(df_test) / total

        assert abs(train_ratio - 0.60) < 0.02
        assert abs(val_ratio - 0.20) < 0.02
        assert abs(test_ratio - 0.20) < 0.02

    def test_no_data_leakage(self, sample_dataframe):
        """Test that there's no overlap between splits."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        train_ids = set(df_train["id"].to_list())
        val_ids = set(df_val["id"].to_list())
        test_ids = set(df_test["id"].to_list())

        # Check no overlap
        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(val_ids.intersection(test_ids)) == 0

    def test_reproducibility(self, sample_dataframe):
        """Test that same random_state gives same split."""
        df_train1, df_val1, df_test1 = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target", random_state=42
        )

        df_train2, df_val2, df_test2 = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target", random_state=42
        )

        # Check same rows in each split
        assert df_train1["id"].to_list() == df_train2["id"].to_list()
        assert df_val1["id"].to_list() == df_val2["id"].to_list()
        assert df_test1["id"].to_list() == df_test2["id"].to_list()

    def test_different_random_states(self, sample_dataframe):
        """Test that different random_states give different splits."""
        df_train1, _, _ = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target", random_state=42
        )

        df_train2, _, _ = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target", random_state=123
        )

        # Check different rows
        assert df_train1["id"].to_list() != df_train2["id"].to_list()

    def test_invalid_sizes(self, sample_dataframe):
        """Test that invalid sizes raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            stratified_train_val_test_split(
                df=sample_dataframe,
                target_col="target",
                train_size=0.60,
                val_size=0.20,
                test_size=0.30,  # Sum > 1.0
            )

    def test_columns_preserved(self, sample_dataframe):
        """Test that all columns are preserved in splits."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        original_cols = set(sample_dataframe.columns)
        assert set(df_train.columns) == original_cols
        assert set(df_val.columns) == original_cols
        assert set(df_test.columns) == original_cols


class TestTemporalTrainValTestSplit:
    """Tests for temporal_train_val_test_split function."""

    def test_basic_temporal_split(self, sample_dataframe):
        """Test basic temporal split."""
        df_sorted = sample_dataframe.sort("date")

        df_train, df_val, df_test = temporal_train_val_test_split(
            df=df_sorted, date_col="date"
        )

        # Check non-empty
        assert len(df_train) > 0
        assert len(df_val) > 0
        assert len(df_test) > 0

        # Check total rows
        total = len(df_train) + len(df_val) + len(df_test)
        assert total == len(df_sorted)

    def test_temporal_order(self, sample_dataframe):
        """Test that temporal order is maintained."""
        df_sorted = sample_dataframe.sort("date")

        df_train, df_val, df_test = temporal_train_val_test_split(
            df=df_sorted, date_col="date"
        )

        # Train should have earliest dates
        # Test should have latest dates
        train_max_date = df_train["date"].max()
        val_min_date = df_val["date"].min()
        val_max_date = df_val["date"].max()
        test_min_date = df_test["date"].min()

        assert train_max_date <= val_min_date
        assert val_max_date <= test_min_date

    def test_temporal_ratios(self, sample_dataframe):
        """Test temporal split ratios."""
        df_sorted = sample_dataframe.sort("date")

        df_train, df_val, df_test = temporal_train_val_test_split(
            df=df_sorted,
            date_col="date",
            train_size=0.70,
            val_size=0.15,
            test_size=0.15,
        )

        total = len(df_sorted)
        train_ratio = len(df_train) / total
        val_ratio = len(df_val) / total
        test_ratio = len(df_test) / total

        # Temporal split is exact (no randomness)
        assert abs(train_ratio - 0.70) < 0.01
        assert abs(val_ratio - 0.15) < 0.01
        assert abs(test_ratio - 0.15) < 0.01

    def test_no_leakage_temporal(self, sample_dataframe):
        """Test no data leakage in temporal split."""
        df_sorted = sample_dataframe.sort("date")

        df_train, df_val, df_test = temporal_train_val_test_split(
            df=df_sorted, date_col="date"
        )

        train_ids = set(df_train["id"].to_list())
        val_ids = set(df_val["id"].to_list())
        test_ids = set(df_test["id"].to_list())

        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(val_ids.intersection(test_ids)) == 0


class TestValidateSplits:
    """Tests for validate_splits function."""

    def test_valid_splits(self, sample_dataframe):
        """Test validation of valid splits."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        result = validate_splits(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            target_col="target",
        )

        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_detects_size_issues(self, sample_dataframe):
        """Test detection of size issues."""
        # Create splits with wrong sizes
        df_train = sample_dataframe[:500]  # 50%
        df_val = sample_dataframe[500:650]  # 15%
        df_test = sample_dataframe[650:]  # 35% (too large)

        result = validate_splits(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            target_col="target",
            expected_train_size=0.70,
        )

        assert result["valid"] is False
        assert any("size" in issue.lower() for issue in result["issues"])

    def test_validation_statistics(self, sample_dataframe):
        """Test that validation includes statistics."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        result = validate_splits(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            target_col="target",
        )

        assert "statistics" in result
        assert "sizes" in result["statistics"]
        # Check that target statistics are present (key name may vary)
        assert len(result["statistics"]) >= 2  # At least sizes and target stats


class TestGetSplitStatistics:
    """Tests for get_split_statistics function."""

    def test_basic_statistics(self, sample_dataframe):
        """Test basic statistics calculation."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        stats = get_split_statistics(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            numeric_cols=["target", "feature1", "feature2"],
        )

        # Check structure
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats

        # Check numeric stats under 'numeric' key
        assert "numeric" in stats["train"]
        assert "target" in stats["train"]["numeric"]
        assert "mean" in stats["train"]["numeric"]["target"]
        assert "std" in stats["train"]["numeric"]["target"]

    def test_categorical_statistics(self, sample_dataframe):
        """Test categorical feature statistics."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        stats = get_split_statistics(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            categorical_cols=["category"],
        )

        # Check categorical stats under 'categorical' key
        assert "categorical" in stats["train"]
        assert "category" in stats["train"]["categorical"]
        assert "value_counts" in stats["train"] or isinstance(
            stats["train"]["categorical"]["category"], dict
        )

    def test_statistics_consistency(self, sample_dataframe):
        """Test that statistics are consistent across splits."""
        df_train, df_val, df_test = stratified_train_val_test_split(
            df=sample_dataframe, target_col="target"
        )

        stats = get_split_statistics(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            numeric_cols=["target"],
        )

        # Target mean should be similar (stratified)
        train_mean = stats["train"]["numeric"]["target"]["mean"]
        val_mean = stats["val"]["numeric"]["target"]["mean"]
        test_mean = stats["test"]["numeric"]["target"]["mean"]

        # Allow 20% difference due to randomness
        assert abs(train_mean - val_mean) / train_mean < 0.20
        assert abs(train_mean - test_mean) / train_mean < 0.20
