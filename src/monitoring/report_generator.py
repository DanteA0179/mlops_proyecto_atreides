"""
Report generator for drift monitoring.

This module handles generation and storage of drift monitoring reports,
including HTML reports from Evidently and JSON/CSV metrics.

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import gzip
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

# Note: In Evidently 0.7.15, report.run() returns a Snapshot object
# We don't need to import Report here, just work with Snapshot

from src.monitoring.config import MonitoringConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Report generator for drift monitoring.

    This class handles:
    - Saving Evidently HTML reports
    - Extracting and saving metrics as JSON
    - Appending metrics to historical CSV
    - Managing report retention and cleanup
    - Compressing old reports

    Attributes
    ----------
    config : MonitoringConfig
        Monitoring configuration
    output_dir : Path
        Base output directory for reports
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize report generator.

        Parameters
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config
        self.output_dir = Path(config.reporting.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_report_path(self, timestamp: datetime, extension: str) -> Path:
        """
        Get path for report file based on timestamp.

        Reports are organized by year/month subdirectories.

        Parameters
        ----------
        timestamp : datetime
            Report timestamp
        extension : str
            File extension (e.g., 'html', 'json')

        Returns
        -------
        Path
            Full path to report file
        """
        # Create year/month subdirectories
        year_month_dir = self.output_dir / str(timestamp.year) / f"{timestamp.month:02d}"
        year_month_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        filename = f"drift_report_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.{extension}"

        return year_month_dir / filename

    def save_html_report(self, snapshot, timestamp: datetime) -> Path:
        """
        Save Evidently HTML report from Snapshot object.

        In Evidently 0.7.15, report.run() returns a Snapshot object which has
        the save_html() method.

        Parameters
        ----------
        snapshot : Snapshot
            Evidently Snapshot object from report.run()
        timestamp : datetime
            Report timestamp

        Returns
        -------
        Path
            Path where report was saved

        Raises
        ------
        IOError
            If report could not be saved
        """
        report_path = self._get_report_path(timestamp, "html")

        logger.info(f"Saving HTML report to {report_path}")

        try:
            snapshot.save_html(str(report_path))
            logger.info(f"HTML report saved successfully ({report_path.stat().st_size} bytes)")
            return report_path

        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")
            raise IOError(f"Could not save HTML report: {e}") from e

    def save_metrics_json(self, metrics: Dict[str, Any], timestamp: datetime) -> Path:
        """
        Save metrics as JSON file.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary to save
        timestamp : datetime
            Report timestamp

        Returns
        -------
        Path
            Path where metrics were saved

        Raises
        ------
        IOError
            If metrics could not be saved
        """
        if not self.config.reporting.save_json_metrics:
            logger.debug("JSON metrics saving disabled in config")
            return None

        metrics_path = self._get_report_path(timestamp, "json")

        logger.info(f"Saving metrics JSON to {metrics_path}")

        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"Metrics JSON saved successfully")
            return metrics_path

        except Exception as e:
            logger.error(f"Failed to save metrics JSON: {e}")
            raise IOError(f"Could not save metrics JSON: {e}") from e

    def append_to_history_csv(self, metrics: Dict[str, Any]) -> Path:
        """
        Append metrics to historical CSV file.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary to append

        Returns
        -------
        Path
            Path to history CSV file

        Raises
        ------
        IOError
            If metrics could not be appended
        """
        if not self.config.reporting.save_csv_history:
            logger.debug("CSV history saving disabled in config")
            return None

        history_path = self.output_dir / "metrics_history.csv"

        logger.info(f"Appending metrics to history CSV: {history_path}")

        try:
            import pandas as pd

            # Flatten metrics for CSV
            flat_metrics = {
                "timestamp": metrics.get("timestamp"),
                "drift_score": metrics.get("drift_score"),
                "share_of_drifted_features": metrics.get("share_of_drifted_features"),
                "number_of_drifted_features": metrics.get("number_of_drifted_features"),
                "dataset_drift": metrics.get("dataset_drift"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "r2_score": metrics.get("r2_score"),
                "mape": metrics.get("mape"),
                "rmse_degradation": metrics.get("rmse_degradation"),
            }

            # Convert to DataFrame
            df = pd.DataFrame([flat_metrics])

            # Append to CSV (create if doesn't exist)
            if history_path.exists():
                df.to_csv(history_path, mode="a", header=False, index=False)
            else:
                df.to_csv(history_path, mode="w", header=True, index=False)

            logger.info("Metrics appended to history CSV")
            return history_path

        except Exception as e:
            logger.error(f"Failed to append to history CSV: {e}")
            raise IOError(f"Could not append to history CSV: {e}") from e

    def save_all(
        self, snapshot, metrics: Dict[str, Any], timestamp: datetime = None
    ) -> Dict[str, Path]:
        """
        Save all report artifacts.

        This is a convenience method that saves:
        - HTML report from Snapshot
        - Metrics JSON
        - Appends to history CSV

        Parameters
        ----------
        snapshot : Snapshot
            Evidently Snapshot object from report.run()
        metrics : Dict[str, Any]
            Metrics dictionary
        timestamp : datetime, optional
            Report timestamp (defaults to now)

        Returns
        -------
        Dict[str, Path]
            Dictionary of saved file paths
        """
        if timestamp is None:
            timestamp = datetime.now()

        saved_paths = {}

        # Save HTML report from Snapshot
        html_path = self.save_html_report(snapshot, timestamp)
        saved_paths["html"] = html_path

        # Add report path to metrics
        metrics["report_path"] = str(html_path)

        # Save JSON metrics
        if self.config.reporting.save_json_metrics:
            json_path = self.save_metrics_json(metrics, timestamp)
            saved_paths["json"] = json_path

        # Append to history CSV
        if self.config.reporting.save_csv_history:
            csv_path = self.append_to_history_csv(metrics)
            saved_paths["history_csv"] = csv_path

        logger.info(f"All reports saved successfully: {list(saved_paths.keys())}")

        return saved_paths

    def cleanup_old_reports(self, retention_days: int = None) -> int:
        """
        Clean up old reports beyond retention period.

        Parameters
        ----------
        retention_days : int, optional
            Retention period in days (defaults to config)

        Returns
        -------
        int
            Number of files deleted
        """
        if retention_days is None:
            retention_days = self.config.reporting.retention_days

        cutoff_date = datetime.now() - timedelta(days=retention_days)

        logger.info(
            f"Cleaning up reports older than {cutoff_date.strftime('%Y-%m-%d')} "
            f"({retention_days} days)"
        )

        deleted_count = 0

        # Iterate through year/month subdirectories
        for year_dir in self.output_dir.glob("*"):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue

            for month_dir in year_dir.glob("*"):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue

                # Check each report file
                for report_file in month_dir.glob("drift_report_*.html"):
                    # Extract timestamp from filename
                    try:
                        # Format: drift_report_YYYY-MM-DD_HH-MM-SS.html
                        timestamp_str = report_file.stem.replace("drift_report_", "")
                        file_date = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

                        if file_date < cutoff_date:
                            # Compress before deletion if enabled
                            if self.config.reporting.compress_old_reports:
                                self._compress_file(report_file)

                            # Delete original
                            report_file.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted old report: {report_file.name}")

                            # Also delete corresponding JSON if exists
                            json_file = report_file.with_suffix(".json")
                            if json_file.exists():
                                json_file.unlink()
                                deleted_count += 1

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse timestamp from {report_file.name}: {e}")

        logger.info(f"Cleanup completed: {deleted_count} files deleted")

        return deleted_count

    def _compress_file(self, file_path: Path) -> Path:
        """
        Compress file using gzip.

        Parameters
        ----------
        file_path : Path
            File to compress

        Returns
        -------
        Path
            Path to compressed file
        """
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")

        logger.debug(f"Compressing {file_path.name} to {compressed_path.name}")

        with open(file_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return compressed_path

    def get_latest_report_path(self) -> Path:
        """
        Get path to most recent HTML report.

        Returns
        -------
        Path
            Path to latest report, or None if no reports exist
        """
        all_reports = list(self.output_dir.glob("*/*/drift_report_*.html"))

        if not all_reports:
            return None

        # Sort by modification time
        latest_report = max(all_reports, key=lambda p: p.stat().st_mtime)

        return latest_report
