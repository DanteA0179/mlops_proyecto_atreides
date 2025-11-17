"""
Alert manager for drift monitoring.

This module handles alert evaluation and notification sending when drift
is detected above configured thresholds.

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import logging
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

from src.monitoring.config import MonitoringConfig

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertContext:
    """
    Context information for an alert.

    Attributes
    ----------
    severity : AlertSeverity
        Alert severity level
    drift_score : float
        Overall drift score
    share_of_drifted_features : float
        Proportion of features with drift
    drifted_features : List[str]
        List of drifted features with scores
    performance_degradation : float
        Performance degradation percentage
    report_path : str
        Path to full HTML report
    timestamp : str
        Timestamp of analysis
    """

    severity: AlertSeverity
    drift_score: float
    share_of_drifted_features: float
    drifted_features: List[str]
    performance_degradation: float
    report_path: str
    timestamp: str


class AlertManager:
    """
    Alert manager for drift monitoring.

    This class evaluates alert conditions and sends notifications via email
    when drift is detected above configured thresholds.

    Attributes
    ----------
    config : MonitoringConfig
        Monitoring configuration
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize alert manager.

        Parameters
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config
        self.smtp_config = config.alerts.email

    def evaluate_alert_conditions(
        self, metrics: Dict[str, Any], reference_rmse: Optional[float] = None
    ) -> AlertContext:
        """
        Evaluate if alert should be triggered.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics from drift analysis
        reference_rmse : Optional[float]
            Reference RMSE for degradation calculation

        Returns
        -------
        AlertContext
            Alert context with severity and details
        """
        drift_score = metrics.get("drift_score", 0.0)
        share_drifted = metrics.get("share_of_drifted_features", 0.0)

        # Calculate performance degradation
        current_rmse = metrics.get("rmse", 0.0)
        degradation = 0.0

        if reference_rmse and reference_rmse > 0 and current_rmse > 0:
            degradation = (current_rmse - reference_rmse) / reference_rmse
        elif metrics.get("rmse_degradation") is not None:
            degradation = metrics.get("rmse_degradation", 0.0)

        # Determine severity
        severity = self._determine_severity(drift_score, share_drifted, degradation)

        # Get drifted features
        drifted_features_dict = metrics.get("drifted_features", {})
        threshold = self.config.drift_detection.thresholds.feature_drift_score

        drifted_features = [
            f"{feat}: {score:.3f}"
            for feat, score in drifted_features_dict.items()
            if score >= threshold
        ]

        return AlertContext(
            severity=severity,
            drift_score=drift_score,
            share_of_drifted_features=share_drifted,
            drifted_features=drifted_features,
            performance_degradation=degradation,
            report_path=metrics.get("report_path", ""),
            timestamp=metrics.get("timestamp", ""),
        )

    def _determine_severity(
        self, drift_score: float, share_drifted: float, degradation: float
    ) -> AlertSeverity:
        """
        Determine alert severity based on metrics.

        Parameters
        ----------
        drift_score : float
            Overall drift score
        share_drifted : float
            Share of drifted features
        degradation : float
            Performance degradation

        Returns
        -------
        AlertSeverity
            Determined severity level
        """
        thresholds = self.config.drift_detection.thresholds
        perf_threshold = self.config.performance.rmse_degradation

        # Critical conditions
        if (
            drift_score >= thresholds.drift_score
            or share_drifted >= thresholds.share_of_drifted_features
            or degradation >= perf_threshold
        ):
            return AlertSeverity.CRITICAL

        # Warning conditions (70% of critical thresholds)
        warning_multiplier = 0.7
        if (
            drift_score >= thresholds.drift_score * warning_multiplier
            or share_drifted >= thresholds.share_of_drifted_features * warning_multiplier
        ):
            return AlertSeverity.WARNING

        return AlertSeverity.INFO

    def send_alert(self, context: AlertContext) -> bool:
        """
        Send alert via email.

        Parameters
        ----------
        context : AlertContext
            Alert context with details

        Returns
        -------
        bool
            True if alert sent successfully, False otherwise
        """
        if not self.config.alerts.enabled:
            logger.info("Alerts disabled in configuration")
            return False

        if context.severity == AlertSeverity.INFO:
            logger.info("No alert needed (severity=INFO)")
            return True

        if "email" not in self.config.alerts.channels:
            logger.info("Email alerts not enabled")
            return False

        logger.info(f"Sending {context.severity.value} alert via email")

        try:
            msg = self._create_email_message(context)
            self._send_email(msg)
            logger.info(f"Alert sent successfully: {context.severity.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    def _create_email_message(self, context: AlertContext) -> MIMEMultipart:
        """
        Create email message with HTML template.

        Parameters
        ----------
        context : AlertContext
            Alert context

        Returns
        -------
        MIMEMultipart
            Email message
        """
        msg = MIMEMultipart("alternative")

        # Format subject
        subject = self.smtp_config.subject_template.format(date=context.timestamp)
        msg["Subject"] = subject
        msg["From"] = self.smtp_config.sender
        msg["To"] = ", ".join(self.smtp_config.recipients)

        # Create HTML body
        html_body = self._generate_html_body(context)

        # Create plain text body (fallback)
        text_body = self._generate_text_body(context)

        # Attach bodies
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        return msg

    def _generate_html_body(self, context: AlertContext) -> str:
        """
        Generate HTML email body.

        Parameters
        ----------
        context : AlertContext
            Alert context

        Returns
        -------
        str
            HTML email body
        """
        severity_color = {
            AlertSeverity.CRITICAL: "red",
            AlertSeverity.WARNING: "orange",
            AlertSeverity.INFO: "blue",
        }

        color = severity_color.get(context.severity, "gray")

        # Format drifted features list
        drifted_features_html = "".join(
            f"<li>{feat}</li>" for feat in context.drifted_features[:10]
        )

        if len(context.drifted_features) > 10:
            remaining = len(context.drifted_features) - 10
            drifted_features_html += f"<li><em>... and {remaining} more</em></li>"

        # Action required text
        if context.severity == AlertSeverity.CRITICAL:
            action_text = (
                "<strong style='color: red;'>IMMEDIATE ACTION REQUIRED:</strong> "
                "Review the drift report and consider retraining the model"
            )
        elif context.severity == AlertSeverity.WARNING:
            action_text = (
                "<strong style='color: orange;'>ACTION RECOMMENDED:</strong> "
                "Monitor the situation and review the detailed report"
            )
        else:
            action_text = "No immediate action required"

        html = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
              .header {{ background-color: {color}; color: white; padding: 20px; }}
              .content {{ padding: 20px; }}
              .metric {{ margin: 10px 0; }}
              .metric-label {{ font-weight: bold; }}
              .action-box {{
                background-color: #f0f0f0;
                border-left: 4px solid {color};
                padding: 15px;
                margin: 20px 0;
              }}
              ul {{ margin: 10px 0; padding-left: 20px; }}
            </style>
          </head>
          <body>
            <div class="header">
              <h2>Data Drift Alert - {context.severity.value}</h2>
            </div>

            <div class="content">
              <h3>Summary</h3>
              <div class="metric">
                <span class="metric-label">Timestamp:</span> {context.timestamp}
              </div>
              <div class="metric">
                <span class="metric-label">Drift Score:</span> {context.drift_score:.3f}
              </div>
              <div class="metric">
                <span class="metric-label">Drifted Features:</span>
                {context.share_of_drifted_features:.1%}
                ({len(context.drifted_features)} features)
              </div>
              <div class="metric">
                <span class="metric-label">Performance Degradation:</span>
                {context.performance_degradation:+.2%}
              </div>

              <h3>Drifted Features ({len(context.drifted_features)})</h3>
              <ul>
                {drifted_features_html}
              </ul>

              <div class="action-box">
                <h3>Action Required</h3>
                <p>{action_text}</p>
              </div>

              <p>
                <a href="file:///{context.report_path}"
                   style="display: inline-block; padding: 10px 20px;
                          background-color: {color}; color: white;
                          text-decoration: none; border-radius: 5px;">
                  View Full Report
                </a>
              </p>

              <hr style="margin-top: 40px; border: none; border-top: 1px solid #ccc;">
              <p style="color: #666; font-size: 12px;">
                This is an automated alert from the Energy Optimization Copilot monitoring system.
                <br>
                Report generated on {context.timestamp}
              </p>
            </div>
          </body>
        </html>
        """

        return html

    def _generate_text_body(self, context: AlertContext) -> str:
        """
        Generate plain text email body.

        Parameters
        ----------
        context : AlertContext
            Alert context

        Returns
        -------
        str
            Plain text email body
        """
        drifted_features_text = "\n".join(
            f"  - {feat}" for feat in context.drifted_features[:10]
        )

        if len(context.drifted_features) > 10:
            remaining = len(context.drifted_features) - 10
            drifted_features_text += f"\n  ... and {remaining} more"

        text = f"""
DATA DRIFT ALERT - {context.severity.value}

Summary:
--------
Timestamp: {context.timestamp}
Drift Score: {context.drift_score:.3f}
Drifted Features: {context.share_of_drifted_features:.1%} ({len(context.drifted_features)} features)
Performance Degradation: {context.performance_degradation:+.2%}

Drifted Features ({len(context.drifted_features)}):
{drifted_features_text}

Action Required:
----------------
{"IMMEDIATE: Review and potentially retrain model" if context.severity == AlertSeverity.CRITICAL else "Review report and monitor"}

Full Report: {context.report_path}

---
This is an automated alert from the Energy Optimization Copilot monitoring system.
Report generated on {context.timestamp}
        """

        return text.strip()

    def _send_email(self, msg: MIMEMultipart) -> None:
        """
        Send email via SMTP.

        Parameters
        ----------
        msg : MIMEMultipart
            Email message to send

        Raises
        ------
        Exception
            If email sending fails
        """
        if not self.smtp_config.sender:
            raise ValueError("SMTP sender email not configured")

        if not self.smtp_config.password:
            raise ValueError("SMTP password not configured")

        if not self.smtp_config.recipients:
            raise ValueError("No email recipients configured")

        logger.info(
            f"Sending email via {self.smtp_config.smtp_server}:{self.smtp_config.smtp_port}"
        )

        with smtplib.SMTP(
            self.smtp_config.smtp_server, self.smtp_config.smtp_port
        ) as server:
            if self.smtp_config.use_tls:
                server.starttls()

            server.login(self.smtp_config.sender, self.smtp_config.password)
            server.send_message(msg)

        logger.info(f"Email sent to {len(self.smtp_config.recipients)} recipients")
