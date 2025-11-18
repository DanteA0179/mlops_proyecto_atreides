"""
Drift analyzer using Evidently AI.

This module provides drift analysis capabilities using Evidently AI to detect:
- Data drift in input features
- Target drift in ground truth
- Prediction drift in model outputs
- Model performance degradation

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from src.monitoring.config import MonitoringConfig

logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """
    Drift analyzer using Evidently AI.

    This class wraps Evidently AI functionality to provide drift analysis
    between reference (training) and production data.

    Uses Evidently 0.7+ API with Dataset and DataDefinition.

    Attributes
    ----------
    config : MonitoringConfig
        Monitoring configuration
    data_definition : DataDefinition
        Evidently data definition configuration
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize drift analyzer.

        Parameters
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config
        self.data_definition = self._create_data_definition()

    def _create_data_definition(self) -> DataDefinition:
        """
        Create Evidently data definition.

        This defines column types, roles (target, prediction), and features.

        Returns
        -------
        DataDefinition
            Evidently data definition configuration
        """
        # Define target and prediction columns
        target_column = "Usage_kWh"
        prediction_column = "predictions"

        # Identify numerical and categorical features
        # Note: This is hardcoded based on the dataset schema
        # In production, could be inferred from data or config
        numerical_columns = [
            "NSM",
            "CO2(tCO2)",
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            target_column,  # Target is also numerical
            prediction_column,  # Prediction is also numerical
        ]

        categorical_columns = [
            "WeekStatus",
            "Load_Type_Maximum_Load",
            "Load_Type_Medium_Load",
        ]

        return DataDefinition(
            regression=[Regression(target=target_column, prediction=prediction_column)],
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

    def create_drift_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """
        Create comprehensive drift report using Evidently.

        Uses Evidently 0.7+ API with Dataset objects.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Reference data (training set sample)
        current_data : pd.DataFrame
            Current production data

        Returns
        -------
        Snapshot
            Evidently Snapshot object with drift analysis results

        Raises
        ------
        ValueError
            If data is invalid or incompatible
        """
        logger.info("Creating drift report with Evidently AI (v0.7+)")

        try:
            # Convert pandas DataFrames to Evidently Dataset objects
            logger.info(
                f"Converting {len(reference_data)} reference samples "
                f"and {len(current_data)} production samples to Dataset objects"
            )

            reference_dataset = Dataset.from_pandas(
                reference_data, data_definition=self.data_definition
            )

            current_dataset = Dataset.from_pandas(
                current_data, data_definition=self.data_definition
            )

            # Build list of metrics using available presets
            metrics_list = [
                DataDriftPreset(),
                RegressionPreset(),  # Includes regression performance metrics
            ]

            # Create report with presets
            report = Report(metrics=metrics_list)

            # Run analysis on Dataset objects
            # IMPORTANT: report.run() returns a Snapshot object in Evidently 0.7.15
            logger.info("Running drift analysis with Evidently presets")

            snapshot = report.run(reference_dataset, current_dataset)

            logger.info("Drift report generated successfully")

            return snapshot

        except Exception as e:
            logger.error(f"Failed to generate drift report: {e}")
            raise ValueError(f"Drift report generation failed: {e}") from e

    def extract_drift_metrics(self, snapshot) -> dict[str, Any]:
        """
        Extract key drift metrics from Evidently Snapshot.

        In Evidently 0.7.15, report.run() returns a Snapshot object, not a Report.
        The Snapshot.dict() method returns a dict with structure:
        {
            "metrics": [
                {"id": "...", "metric_id": "...", "value": {...}},
                ...
            ],
            "tests": [...]
        }

        Parameters
        ----------
        snapshot : Snapshot
            Evidently Snapshot object from report.run()

        Returns
        -------
        Dict[str, Any]
            Dictionary with key drift metrics
        """
        logger.info("Extracting drift metrics from Snapshot")

        try:
            # Use dict() method to get metrics from Snapshot
            results = snapshot.dict()
        except Exception as e:
            logger.error(f"Failed to convert Snapshot to dict: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

        metrics: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
        }

        # Extract drift metrics from DataDriftPreset
        # The DataDriftPreset creates metrics like:
        # - DriftedColumnsCount with value: {"count": N, "share": 0.X}
        # - ValueDrift(column=X) for each column
        try:
            for metric in results.get("metrics", []):
                metric_id = metric.get("metric_id", "")
                value = metric.get("value", {})

                # DriftedColumnsCount metric contains aggregated drift info
                if "DriftedColumnsCount" in str(metric_id):
                    if isinstance(value, dict):
                        metrics["number_of_drifted_features"] = value.get("count", 0)
                        metrics["share_of_drifted_features"] = value.get("share", 0.0)
                        # Simple drift_score based on share (you can adjust this)
                        metrics["drift_score"] = value.get("share", 0.0)
                        metrics["dataset_drift"] = value.get("share", 0.0) > 0.5

                # ValueDrift metrics for individual columns
                elif "ValueDrift" in str(metric_id):
                    # Extract column name from metric_id string like "ValueDrift(column=feature1)"
                    if "column=" in str(metric_id):
                        col_name = str(metric_id).split("column=")[1].rstrip(")")
                        # Check if drift was detected
                        if isinstance(value, dict) and value.get("drift_detected", False):
                            if "drifted_features" not in metrics:
                                metrics["drifted_features"] = {}
                            # Use drift_score if available, otherwise use 1.0 to indicate drift
                            drift_score = value.get("drift_score", 1.0)
                            metrics["drifted_features"][col_name] = drift_score

        except Exception as e:
            logger.warning(f"Could not extract drift metrics: {e}")

        # Extract regression metrics from RegressionPreset
        # The RegressionPreset creates individual metrics like RMSE, MAE, R2Score, etc.
        try:
            regression_metrics = {}

            for metric in results.get("metrics", []):
                metric_id = metric.get("metric_id", "")
                value = metric.get("value", {})

                # RMSE metric
                if "RMSE" in str(metric_id) and isinstance(value, dict):
                    regression_metrics["rmse"] = value.get("mean")

                # MAE metric
                elif "MAE" in str(metric_id) and isinstance(value, dict):
                    regression_metrics["mae"] = value.get("mean")

                # R2Score metric
                elif "R2Score" in str(metric_id) and isinstance(value, dict):
                    regression_metrics["r2_score"] = value.get("mean")

                # MAPE metric
                elif "MAPE" in str(metric_id) and isinstance(value, dict):
                    regression_metrics["mape"] = value.get("mean")

            # Add regression metrics to main metrics dict
            if regression_metrics:
                metrics.update(regression_metrics)

                # Note: Calculating degradation requires storing reference metrics separately
                # For now, we just store current metrics
                # TODO: Implement degradation calculation by comparing with stored reference

        except Exception as e:
            logger.warning(f"Could not extract regression metrics: {e}")

        # Ensure drifted_features exists even if empty
        if "drifted_features" not in metrics:
            metrics["drifted_features"] = {}

        logger.info(
            f"Extracted metrics: drift_score={metrics.get('drift_score', 0.0):.3f}, "
            f"drifted_features={metrics.get('number_of_drifted_features', 0)}"
        )

        return metrics

    def analyze_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> tuple:
        """
        Perform complete drift analysis.

        This is a convenience method that creates the report and extracts metrics.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Reference data (training set sample)
        current_data : pd.DataFrame
            Current production data

        Returns
        -------
        tuple[Snapshot, Dict[str, Any]]
            (Evidently Snapshot object, extracted metrics dictionary)
        """
        # Create report (returns Snapshot in Evidently 0.7.15)
        snapshot = self.create_drift_report(reference_data, current_data)

        # Extract metrics from Snapshot
        metrics = self.extract_drift_metrics(snapshot)

        return snapshot, metrics

    def get_drifted_features_summary(
        self, metrics: dict[str, Any], threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        """
        Get summary of drifted features above threshold.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary from extract_drift_metrics
        threshold : float, default=0.5
            Drift score threshold

        Returns
        -------
        List[tuple[str, float]]
            List of (feature_name, drift_score) for drifted features
        """
        drifted_features = metrics.get("drifted_features", {})

        summary = [
            (feature, score) for feature, score in drifted_features.items() if score >= threshold
        ]

        # Sort by drift score descending
        summary.sort(key=lambda x: x[1], reverse=True)

        return summary
