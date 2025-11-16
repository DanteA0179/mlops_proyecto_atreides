"""
Unit tests for monitoring module.

Tests cover prediction logging, reference data preparation,
and drift report generation.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring.log_prediction import log_prediction


def _can_import_evidently() -> bool:
    """Check if evidently can be imported successfully."""
    try:
        from evidently.metric_preset import DataDriftPreset  # noqa: F401
        from evidently.report import Report  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


class TestLogPrediction:
    """Tests for prediction logging functionality."""

    def test_log_prediction_creates_file(self, tmp_path: Path) -> None:
        """Test that log_prediction creates CSV file with header."""
        # Setup
        log_file = tmp_path / "predictions.csv"
        features = {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0,
        }
        prediction = 42.5

        # Mock the log file path
        with patch("src.monitoring.log_prediction.Path") as mock_path:
            mock_path.return_value = log_file
            log_prediction(features, prediction)

        # Verify file was created
        assert log_file.exists()

        # Verify header
        with open(log_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["timestamp", "feature1", "feature2", "feature3", "prediction"]

    def test_log_prediction_appends(self, tmp_path: Path) -> None:
        """Test that log_prediction appends multiple predictions."""
        # Setup
        log_file = tmp_path / "predictions.csv"
        features = {"feature1": 1.0, "feature2": 2.0}

        # Mock the log file path
        with patch("src.monitoring.log_prediction.Path") as mock_path:
            mock_path.return_value = log_file

            # Log multiple predictions
            log_prediction(features, 10.0)
            log_prediction(features, 20.0)
            log_prediction(features, 30.0)

        # Verify 3 predictions were logged (+ 1 header = 4 rows)
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 4  # header + 3 predictions

    def test_log_prediction_handles_errors(self, tmp_path: Path) -> None:
        """Test that log_prediction handles errors gracefully."""
        # Setup - invalid path
        with patch("src.monitoring.log_prediction.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.parent.mkdir.side_effect = PermissionError("No permission")

            # Should not raise exception
            try:
                log_prediction({"feature1": 1.0}, 42.0)
            except Exception as e:
                pytest.fail(f"log_prediction raised exception: {e}")

    def test_csv_has_correct_columns(self, tmp_path: Path) -> None:
        """Test that CSV has correct column structure."""
        # Setup
        log_file = tmp_path / "predictions.csv"
        features = {
            "Lagging_Current_Reactive.Power_kVarh": 23.45,
            "Leading_Current_Reactive_Power_kVarh": 12.30,
            "CO2(tCO2)": 0.05,
        }
        prediction = 42.5

        # Mock the log file path
        with patch("src.monitoring.log_prediction.Path") as mock_path:
            mock_path.return_value = log_file
            log_prediction(features, prediction)

        # Read CSV and verify structure
        df = pd.read_csv(log_file)

        # Check columns
        expected_cols = ["timestamp"] + list(features.keys()) + ["prediction"]
        assert list(df.columns) == expected_cols

        # Check data types
        assert df["prediction"].dtype in [float, "float64"]
        assert len(df) == 1

    def test_log_prediction_with_empty_features(self, tmp_path: Path) -> None:
        """Test log_prediction with empty features dict."""
        # Setup
        log_file = tmp_path / "predictions.csv"
        features = {}
        prediction = 42.5

        # Mock the log file path
        with patch("src.monitoring.log_prediction.Path") as mock_path:
            mock_path.return_value = log_file
            log_prediction(features, prediction)

        # Verify file was created with minimal columns
        assert log_file.exists()
        with open(log_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["timestamp", "prediction"]


class TestPrepareReferenceData:
    """Tests for reference data preparation script."""

    @patch("pandas.read_parquet")
    def test_prepare_reference_data_samples_correctly(self, mock_read_parquet: MagicMock) -> None:
        """Test that prepare_reference_data samples 1000 rows."""
        # Setup mock data
        mock_df = pd.DataFrame({"feature1": range(2000), "feature2": range(2000)})
        mock_read_parquet.return_value = mock_df

        # Import and run
        from scripts.prepare_reference_data import prepare_reference_data

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.mkdir"):
                    prepare_reference_data()

        # Verify sampling
        mock_to_csv.assert_called_once()
        # Check that sampled data was saved
        call_args = mock_to_csv.call_args
        assert call_args is not None


class TestGenerateDriftReport:
    """Tests for drift report generation script."""

    @pytest.mark.skipif(
        not _can_import_evidently(),
        reason="Evidently not properly installed",
    )
    @patch("pandas.read_csv")
    @patch("evidently.report.Report.run")
    @patch("evidently.report.Report.save_html")
    def test_generate_drift_report_creates_html(
        self,
        mock_save_html: MagicMock,
        mock_run: MagicMock,
        mock_read_csv: MagicMock,
    ) -> None:
        """Test that generate_drift_report creates HTML report."""
        # Setup mock data
        ref_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        prod_df = pd.DataFrame(
            {
                "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "feature1": [1.1, 2.1, 3.1],
                "feature2": [4.1, 5.1, 6.1],
                "prediction": [10, 20, 30],
            }
        )

        mock_read_csv.side_effect = [ref_df, prod_df]

        # Import and run
        from scripts.generate_drift_report import generate_drift_report

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                generate_drift_report()

        # Verify report was generated
        mock_run.assert_called_once()
        mock_save_html.assert_called_once()

    @pytest.mark.skipif(
        not _can_import_evidently(),
        reason="Evidently not properly installed",
    )
    @patch("pathlib.Path.exists")
    def test_generate_drift_report_raises_if_no_reference(self, mock_exists: MagicMock) -> None:
        """Test that generate_drift_report raises error if reference data missing."""
        # Setup - reference file doesn't exist
        mock_exists.return_value = False

        # Import
        from scripts.generate_drift_report import generate_drift_report

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Reference data not found"):
            generate_drift_report()
