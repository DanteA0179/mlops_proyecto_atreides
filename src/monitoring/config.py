"""
Configuration management for drift monitoring system.

This module handles loading and validation of monitoring configuration from
YAML files and environment variables.

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ReferenceDataConfig:
    """Configuration for reference data."""

    path: str
    size: int = 10000
    stratify_column: str = "Load_Type"


@dataclass
class ProductionDataConfig:
    """Configuration for production data."""

    source: str = "file"
    timeframe_days: int = 7
    min_samples: int = 1000


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""

    drift_score: float = 0.7
    share_of_drifted_features: float = 0.5
    feature_drift_score: float = 0.5
    target_drift_score: float = 0.6
    prediction_drift_score: float = 0.6


@dataclass
class StatisticalTestThresholds:
    """Thresholds for statistical tests."""

    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05
    wasserstein_threshold: float = 0.1


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""

    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    statistical_tests: StatisticalTestThresholds = field(default_factory=StatisticalTestThresholds)


@dataclass
class PerformanceThresholds:
    """Thresholds for performance degradation."""

    rmse_degradation: float = 0.15
    mae_degradation: float = 0.15
    r2_degradation: float = 0.10


@dataclass
class ReportingConfig:
    """Configuration for report generation."""

    output_dir: str = "reports/monitoring"
    retention_days: int = 90
    compress_old_reports: bool = True
    save_json_metrics: bool = True
    save_csv_history: bool = True


@dataclass
class EmailConfig:
    """Configuration for email alerts."""

    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender: str = ""
    password: str = ""
    recipients: list[str] = field(default_factory=list)
    subject_template: str = "[ALERT] Data Drift Detected - {date}"
    html_template: bool = True


@dataclass
class AlertsConfig:
    """Configuration for alerts."""

    enabled: bool = True
    channels: list[str] = field(default_factory=lambda: ["email"])
    email: EmailConfig = field(default_factory=EmailConfig)


@dataclass
class MonitoringConfig:
    """Main monitoring configuration."""

    reference_data: ReferenceDataConfig
    production_data: ProductionDataConfig
    drift_detection: DriftDetectionConfig
    performance: PerformanceThresholds
    reporting: ReportingConfig
    alerts: AlertsConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> "MonitoringConfig":
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_path : Path
            Path to YAML configuration file

        Returns
        -------
        MonitoringConfig
            Loaded configuration object

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
        ValueError
            If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if not config_dict or "monitoring" not in config_dict:
            raise ValueError("Invalid configuration file: missing 'monitoring' section")

        monitoring_config = config_dict["monitoring"]

        # Load environment variables
        load_dotenv()

        # Parse reference data config
        ref_data_config = ReferenceDataConfig(
            path=monitoring_config["reference_data"]["path"],
            size=monitoring_config["reference_data"].get("size", 10000),
            stratify_column=monitoring_config["reference_data"].get("stratify_column", "Load_Type"),
        )

        # Parse production data config
        prod_data_config = ProductionDataConfig(
            source=monitoring_config["production_data"].get("source", "file"),
            timeframe_days=monitoring_config["production_data"].get("timeframe_days", 7),
            min_samples=monitoring_config["production_data"].get("min_samples", 1000),
        )

        # Parse drift detection config
        drift_thresholds = DriftThresholds(
            drift_score=monitoring_config["drift_detection"]["thresholds"].get("drift_score", 0.7),
            share_of_drifted_features=monitoring_config["drift_detection"]["thresholds"].get(
                "share_of_drifted_features", 0.5
            ),
            feature_drift_score=monitoring_config["drift_detection"]["thresholds"].get(
                "feature_drift_score", 0.5
            ),
            target_drift_score=monitoring_config["drift_detection"]["thresholds"].get(
                "target_drift_score", 0.6
            ),
            prediction_drift_score=monitoring_config["drift_detection"]["thresholds"].get(
                "prediction_drift_score", 0.6
            ),
        )

        stat_test_thresholds = StatisticalTestThresholds(
            psi_threshold=monitoring_config["drift_detection"]["statistical_tests"].get(
                "psi_threshold", 0.2
            ),
            ks_pvalue_threshold=monitoring_config["drift_detection"]["statistical_tests"].get(
                "ks_pvalue_threshold", 0.05
            ),
            wasserstein_threshold=monitoring_config["drift_detection"]["statistical_tests"].get(
                "wasserstein_threshold", 0.1
            ),
        )

        drift_detection = DriftDetectionConfig(
            thresholds=drift_thresholds, statistical_tests=stat_test_thresholds
        )

        # Parse performance thresholds
        perf_thresholds = PerformanceThresholds(
            rmse_degradation=monitoring_config["performance"]["thresholds"].get(
                "rmse_degradation", 0.15
            ),
            mae_degradation=monitoring_config["performance"]["thresholds"].get(
                "mae_degradation", 0.15
            ),
            r2_degradation=monitoring_config["performance"]["thresholds"].get(
                "r2_degradation", 0.10
            ),
        )

        # Parse reporting config
        reporting = ReportingConfig(
            output_dir=monitoring_config["reporting"].get("output_dir", "reports/monitoring"),
            retention_days=monitoring_config["reporting"].get("retention_days", 90),
            compress_old_reports=monitoring_config["reporting"].get("compress_old_reports", True),
            save_json_metrics=monitoring_config["reporting"].get("save_json_metrics", True),
            save_csv_history=monitoring_config["reporting"].get("save_csv_history", True),
        )

        # Parse alerts config
        email_config_dict = monitoring_config["alerts"]["email"]

        # Resolve environment variables in email config
        email_sender = cls._resolve_env_var(email_config_dict.get("sender", ""))
        email_password = cls._resolve_env_var(email_config_dict.get("password", ""))

        email_config = EmailConfig(
            smtp_server=email_config_dict.get("smtp_server", "smtp.gmail.com"),
            smtp_port=email_config_dict.get("smtp_port", 587),
            use_tls=email_config_dict.get("use_tls", True),
            sender=email_sender,
            password=email_password,
            recipients=email_config_dict.get("recipients", []),
            subject_template=email_config_dict.get(
                "subject_template", "[ALERT] Data Drift Detected - {date}"
            ),
            html_template=email_config_dict.get("html_template", True),
        )

        alerts = AlertsConfig(
            enabled=monitoring_config["alerts"].get("enabled", True),
            channels=monitoring_config["alerts"].get("channels", ["email"]),
            email=email_config,
        )

        return cls(
            reference_data=ref_data_config,
            production_data=prod_data_config,
            drift_detection=drift_detection,
            performance=perf_thresholds,
            reporting=reporting,
            alerts=alerts,
        )

    @staticmethod
    def _resolve_env_var(value: str) -> str:
        """
        Resolve environment variable references in config values.

        Supports ${VAR_NAME} syntax.

        Parameters
        ----------
        value : str
            Value potentially containing env var reference

        Returns
        -------
        str
            Resolved value with env vars substituted
        """
        if not value:
            return ""

        if value.startswith("${") and value.endswith("}"):
            env_var_name = value[2:-1]
            return os.getenv(env_var_name, "")

        return value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary
        """
        return {
            "reference_data": {
                "path": self.reference_data.path,
                "size": self.reference_data.size,
                "stratify_column": self.reference_data.stratify_column,
            },
            "production_data": {
                "source": self.production_data.source,
                "timeframe_days": self.production_data.timeframe_days,
                "min_samples": self.production_data.min_samples,
            },
            "drift_detection": {
                "thresholds": {
                    "drift_score": self.drift_detection.thresholds.drift_score,
                    "share_of_drifted_features": self.drift_detection.thresholds.share_of_drifted_features,
                    "feature_drift_score": self.drift_detection.thresholds.feature_drift_score,
                    "target_drift_score": self.drift_detection.thresholds.target_drift_score,
                    "prediction_drift_score": self.drift_detection.thresholds.prediction_drift_score,
                },
                "statistical_tests": {
                    "psi_threshold": self.drift_detection.statistical_tests.psi_threshold,
                    "ks_pvalue_threshold": self.drift_detection.statistical_tests.ks_pvalue_threshold,
                    "wasserstein_threshold": self.drift_detection.statistical_tests.wasserstein_threshold,
                },
            },
            "performance": {
                "rmse_degradation": self.performance.rmse_degradation,
                "mae_degradation": self.performance.mae_degradation,
                "r2_degradation": self.performance.r2_degradation,
            },
            "reporting": {
                "output_dir": self.reporting.output_dir,
                "retention_days": self.reporting.retention_days,
                "compress_old_reports": self.reporting.compress_old_reports,
                "save_json_metrics": self.reporting.save_json_metrics,
                "save_csv_history": self.reporting.save_csv_history,
            },
            "alerts": {
                "enabled": self.alerts.enabled,
                "channels": self.alerts.channels,
                "email": {
                    "smtp_server": self.alerts.email.smtp_server,
                    "smtp_port": self.alerts.email.smtp_port,
                    "use_tls": self.alerts.email.use_tls,
                    "sender": self.alerts.email.sender,
                    "recipients": self.alerts.email.recipients,
                    "subject_template": self.alerts.email.subject_template,
                    "html_template": self.alerts.email.html_template,
                },
            },
        }

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns
        -------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate reference data
        ref_path = Path(self.reference_data.path)
        if not ref_path.exists():
            errors.append(f"Reference data file not found: {ref_path}")

        # Validate thresholds are in valid ranges
        if not 0 <= self.drift_detection.thresholds.drift_score <= 1:
            errors.append("drift_score threshold must be between 0 and 1")

        if not 0 <= self.drift_detection.thresholds.share_of_drifted_features <= 1:
            errors.append("share_of_drifted_features threshold must be between 0 and 1")

        # Validate email config if alerts enabled
        if self.alerts.enabled and "email" in self.alerts.channels:
            if not self.alerts.email.sender:
                errors.append("Email sender not configured (check SMTP_SENDER_EMAIL)")

            if not self.alerts.email.password:
                errors.append("Email password not configured (check SMTP_SENDER_PASSWORD)")

            if not self.alerts.email.recipients:
                errors.append("No email recipients configured")

        return errors
