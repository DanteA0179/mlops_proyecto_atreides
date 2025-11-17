"""
Main CLI script for drift monitoring with Evidently AI.

This script orchestrates the entire drift monitoring pipeline:
1. Load configuration
2. Load reference and production data
3. Analyze drift using Evidently
4. Generate and save reports
5. Evaluate alert conditions
6. Send notifications if needed
7. Clean up old reports

Usage:
    python -m src.monitoring.evidently_monitor \
        --config config/monitoring_config.yaml \
        --production-data data/production/latest.parquet \
        --send-alerts

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.monitoring.alert_manager import AlertManager
from src.monitoring.config import MonitoringConfig
from src.monitoring.data_loader import DataLoader
from src.monitoring.drift_analyzer import DriftAnalyzer
from src.monitoring.metrics_tracker import MetricsTracker
from src.monitoring.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class MonitoringPipeline:
    """
    Drift monitoring pipeline orchestrator.

    This class coordinates all components of the monitoring system.

    Attributes
    ----------
    config : MonitoringConfig
        Monitoring configuration
    data_loader : DataLoader
        Data loader instance
    drift_analyzer : DriftAnalyzer
        Drift analyzer instance
    report_generator : ReportGenerator
        Report generator instance
    alert_manager : AlertManager
        Alert manager instance
    metrics_tracker : MetricsTracker
        Metrics tracker instance
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize monitoring pipeline.

        Parameters
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config

        # Initialize components
        logger.info("Initializing monitoring pipeline components")

        self.data_loader = DataLoader(config)
        self.drift_analyzer = DriftAnalyzer(config)
        self.report_generator = ReportGenerator(config)
        self.alert_manager = AlertManager(config)
        self.metrics_tracker = MetricsTracker(config)

        logger.info("All components initialized successfully")

    def run(
        self,
        production_data_path: Path,
        send_alerts: bool = False,
        cleanup: bool = True,
    ) -> int:
        """
        Run complete monitoring pipeline.

        Parameters
        ----------
        production_data_path : Path
            Path to production data file
        send_alerts : bool, default=False
            Whether to send email alerts
        cleanup : bool, default=True
            Whether to clean up old reports

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)
        """
        logger.info("=" * 70)
        logger.info("DRIFT MONITORING PIPELINE - START")
        logger.info("=" * 70)

        try:
            # Step 1: Load and validate data
            logger.info("\n[1/7] Loading and validating data")
            reference_data, production_data = self.data_loader.load_and_validate(
                production_data_path
            )

            logger.info(
                f"  Reference data: {reference_data.shape[0]} samples, "
                f"{reference_data.shape[1]} features"
            )
            logger.info(
                f"  Production data: {production_data.shape[0]} samples, "
                f"{production_data.shape[1]} features"
            )

            # Step 2: Analyze drift
            logger.info("\n[2/7] Analyzing drift with Evidently AI")
            report, metrics = self.drift_analyzer.analyze_drift(
                reference_data, production_data
            )

            logger.info(f"  Drift score: {metrics.get('drift_score', 'N/A'):.3f}")
            logger.info(
                f"  Drifted features: {metrics.get('number_of_drifted_features', 'N/A')}"
            )

            # Step 3: Generate and save reports
            logger.info("\n[3/7] Generating and saving reports")
            timestamp = datetime.now()
            saved_paths = self.report_generator.save_all(report, metrics, timestamp)

            logger.info(f"  HTML report: {saved_paths.get('html')}")
            if saved_paths.get("json"):
                logger.info(f"  JSON metrics: {saved_paths.get('json')}")

            # Step 4: Track metrics
            logger.info("\n[4/7] Tracking historical metrics")
            trends_report = self.metrics_tracker.get_drift_trends_report(window=10)

            drift_trend = trends_report["metrics"].get("drift_score", {}).get("trend")
            if drift_trend:
                logger.info(f"  Drift score trend: {drift_trend}")

            # Step 5: Evaluate alert conditions
            logger.info("\n[5/7] Evaluating alert conditions")
            alert_context = self.alert_manager.evaluate_alert_conditions(metrics)

            logger.info(f"  Alert severity: {alert_context.severity.value}")

            # Step 6: Send alerts if enabled
            if send_alerts:
                logger.info("\n[6/7] Sending alerts")
                alert_sent = self.alert_manager.send_alert(alert_context)

                if alert_sent:
                    logger.info("  Alerts sent successfully")
                else:
                    logger.warning("  No alerts sent (may be disabled or INFO severity)")
            else:
                logger.info("\n[6/7] Skipping alerts (--send-alerts not specified)")

            # Step 7: Clean up old reports
            if cleanup:
                logger.info("\n[7/7] Cleaning up old reports")
                deleted_count = self.report_generator.cleanup_old_reports()
                logger.info(f"  Deleted {deleted_count} old report files")
            else:
                logger.info("\n[7/7] Skipping cleanup (--no-cleanup specified)")

            # Summary
            logger.info("\n" + "=" * 70)
            logger.info("DRIFT MONITORING PIPELINE - COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"  Drift Score: {metrics.get('drift_score', 0.0):.3f}")
            logger.info(
                f"  Drifted Features: {metrics.get('share_of_drifted_features', 0.0):.1%}"
            )
            logger.info(f"  Alert Severity: {alert_context.severity.value}")
            logger.info(f"  Report: {saved_paths.get('html')}")
            logger.info("=" * 70)

            return 0

        except Exception as e:
            logger.error(f"\n{'=' * 70}")
            logger.error("DRIFT MONITORING PIPELINE - FAILED")
            logger.error(f"{'=' * 70}")
            logger.error(f"Error: {e}", exc_info=True)
            logger.error(f"{'=' * 70}")

            return 1


def main():
    """Main function to parse arguments and run monitoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Drift monitoring with Evidently AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m src.monitoring.evidently_monitor \\
      --production-data data/production/latest.parquet

  # With custom config and alerts
  python -m src.monitoring.evidently_monitor \\
      --config config/monitoring_config.yaml \\
      --production-data data/production/latest.parquet \\
      --send-alerts

  # Without cleanup
  python -m src.monitoring.evidently_monitor \\
      --production-data data/production/latest.parquet \\
      --no-cleanup
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/monitoring_config.yaml"),
        help="Path to monitoring configuration YAML file",
    )

    parser.add_argument(
        "--production-data",
        type=Path,
        required=True,
        help="Path to production data parquet file",
    )

    parser.add_argument(
        "--send-alerts",
        action="store_true",
        help="Send email alerts if drift detected",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of old reports",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate arguments
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    if not args.production_data.exists():
        logger.error(f"Production data file not found: {args.production_data}")
        return 1

    # Load configuration
    try:
        logger.info(f"Loading configuration from {args.config}")
        config = MonitoringConfig.from_yaml(args.config)

        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1

        logger.info("Configuration loaded and validated successfully")

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Create and run pipeline
    pipeline = MonitoringPipeline(config)

    exit_code = pipeline.run(
        production_data_path=args.production_data,
        send_alerts=args.send_alerts,
        cleanup=not args.no_cleanup,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
