"""
Metrics tracker for drift monitoring.

This module tracks historical metrics and identifies trends in drift
and model performance over time.

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.config import MonitoringConfig

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Metrics tracker for drift monitoring.

    This class tracks historical metrics and provides trend analysis
    capabilities.

    Attributes
    ----------
    config : MonitoringConfig
        Monitoring configuration
    history_path : Path
        Path to metrics history CSV file
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize metrics tracker.

        Parameters
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config
        self.history_path = Path(config.reporting.output_dir) / "metrics_history.csv"

    def load_history(self, limit: int | None = None) -> pd.DataFrame:
        """
        Load historical metrics.

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of recent records to load

        Returns
        -------
        pd.DataFrame
            Historical metrics DataFrame
        """
        if not self.history_path.exists():
            logger.warning(f"History file not found: {self.history_path}")
            return pd.DataFrame()

        logger.info(f"Loading metrics history from {self.history_path}")

        df = pd.read_csv(self.history_path)

        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp descending (most recent first)
        df = df.sort_values("timestamp", ascending=False)

        # Limit number of records if specified
        if limit:
            df = df.head(limit)

        logger.info(f"Loaded {len(df)} historical records")

        return df

    def get_latest_metrics(self) -> dict[str, Any] | None:
        """
        Get most recent metrics.

        Returns
        -------
        Optional[Dict[str, Any]]
            Latest metrics as dictionary, or None if no history
        """
        df = self.load_history(limit=1)

        if df.empty:
            return None

        return df.iloc[0].to_dict()

    def calculate_trend(self, metric_name: str, window: int = 5) -> str | None:
        """
        Calculate trend for a metric over recent history.

        Parameters
        ----------
        metric_name : str
            Name of metric to analyze
        window : int, default=5
            Number of recent records to consider

        Returns
        -------
        Optional[str]
            Trend direction: 'increasing', 'decreasing', 'stable', or None
        """
        df = self.load_history(limit=window)

        if df.empty or metric_name not in df.columns:
            return None

        values = df[metric_name].dropna()

        if len(values) < 2:
            return None

        # Reverse to get chronological order
        values = values.iloc[::-1].values

        # Calculate simple linear trend
        x = list(range(len(values)))
        slope = self._calculate_slope(x, values)

        # Determine trend based on slope
        threshold = 0.001  # Minimum slope to consider as trend

        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        else:
            return "stable"

    @staticmethod
    def _calculate_slope(x: list[float], y: list[float]) -> float:
        """
        Calculate slope of simple linear regression.

        Parameters
        ----------
        x : List[float]
            X values
        y : List[float]
            Y values

        Returns
        -------
        float
            Slope of best-fit line
        """
        n = len(x)
        if n == 0:
            return 0.0

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_summary_statistics(self, metric_name: str, window: int = 30) -> dict[str, float] | None:
        """
        Get summary statistics for a metric.

        Parameters
        ----------
        metric_name : str
            Name of metric
        window : int, default=30
            Number of recent records to consider

        Returns
        -------
        Optional[Dict[str, float]]
            Dictionary with mean, std, min, max, latest
        """
        df = self.load_history(limit=window)

        if df.empty or metric_name not in df.columns:
            return None

        values = df[metric_name].dropna()

        if values.empty:
            return None

        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "latest": float(values.iloc[0]),
            "count": len(values),
        }

    def detect_anomalies(
        self, metric_name: str, window: int = 30, std_threshold: float = 2.0
    ) -> tuple[bool, float | None]:
        """
        Detect if latest value is anomalous based on historical data.

        Uses simple z-score method: value is anomalous if more than
        std_threshold standard deviations from mean.

        Parameters
        ----------
        metric_name : str
            Name of metric
        window : int, default=30
            Number of historical records to use as baseline
        std_threshold : float, default=2.0
            Number of standard deviations for anomaly threshold

        Returns
        -------
        Tuple[bool, Optional[float]]
            (is_anomalous, z_score)
        """
        stats = self.get_summary_statistics(metric_name, window)

        if not stats or stats["count"] < 3:
            logger.warning(f"Insufficient history for anomaly detection: {metric_name}")
            return False, None

        latest = stats["latest"]
        mean = stats["mean"]
        std = stats["std"]

        if std == 0:
            return False, 0.0

        z_score = abs((latest - mean) / std)

        is_anomalous = z_score > std_threshold

        if is_anomalous:
            logger.warning(
                f"Anomaly detected in {metric_name}: "
                f"latest={latest:.4f}, mean={mean:.4f}, z_score={z_score:.2f}"
            )

        return is_anomalous, z_score

    def get_drift_trends_report(self, window: int = 10) -> dict[str, Any]:
        """
        Generate comprehensive drift trends report.

        Parameters
        ----------
        window : int, default=10
            Number of recent records to analyze

        Returns
        -------
        Dict[str, Any]
            Dictionary with trend analysis for key metrics
        """
        metrics_to_track = [
            "drift_score",
            "share_of_drifted_features",
            "number_of_drifted_features",
            "rmse",
            "mae",
            "r2_score",
        ]

        report = {
            "window_size": window,
            "metrics": {},
        }

        for metric in metrics_to_track:
            trend = self.calculate_trend(metric, window)
            stats = self.get_summary_statistics(metric, window)
            is_anomalous, z_score = self.detect_anomalies(metric, window)

            report["metrics"][metric] = {
                "trend": trend,
                "statistics": stats,
                "is_anomalous": is_anomalous,
                "z_score": z_score,
            }

        logger.info(f"Generated trends report for {len(metrics_to_track)} metrics")

        return report

    def export_history_summary(self, output_path: Path, window: int = 100) -> None:
        """
        Export summary of historical metrics to file.

        Parameters
        ----------
        output_path : Path
            Path to save summary
        window : int, default=100
            Number of recent records to include
        """
        df = self.load_history(limit=window)

        if df.empty:
            logger.warning("No history available for export")
            return

        logger.info(f"Exporting history summary to {output_path}")

        # Create summary
        summary = {
            "total_records": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
                "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None,
            },
            "metrics_summary": {},
        }

        # Add statistics for each numeric column
        numeric_columns = df.select_dtypes(include=["number"]).columns

        for col in numeric_columns:
            summary["metrics_summary"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "latest": float(df[col].iloc[0]),
            }

        # Save as JSON
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("History summary exported successfully")
