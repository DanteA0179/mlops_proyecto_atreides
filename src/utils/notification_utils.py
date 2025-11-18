"""
Notification System for Training Pipeline.

This module provides a notification manager that supports multiple channels:
- Console: Print to logger
- File: Write to log file
- Slack: Send webhook (future implementation)
- Email: Send email notification (future implementation)
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Manage notifications across multiple channels.

    Supports console, file, Slack, and email notifications.
    """

    def __init__(self, channels: list[str] | None = None):
        """
        Initialize notification manager.

        Parameters
        ----------
        channels : list[str] | None
            List of channels: ['console', 'file', 'slack', 'email']
            Defaults to ['console', 'file']
        """
        self.channels = channels or ["console", "file"]

    def send(self, status: str, metrics: dict, run_id: str | None, config: dict) -> None:
        """
        Send notification to all configured channels.

        Parameters
        ----------
        status : str
            Pipeline status: 'success' or 'failed_threshold'
        metrics : dict
            Model metrics dictionary
        run_id : str | None
            MLflow run ID (None if threshold failed)
        config : dict
            Configuration dictionary
        """
        message = self._format_message(status, metrics, run_id, config)

        if "console" in self.channels:
            self._send_console(message)

        if "file" in self.channels:
            self._send_file(message, config)

        if "slack" in self.channels:
            slack_config = config.get("notifications", {}).get("slack", {})
            if slack_config.get("enabled", False):
                self._send_slack(message, config)

        if "email" in self.channels:
            email_config = config.get("notifications", {}).get("email", {})
            if email_config.get("enabled", False):
                self._send_email(message, config)

    def _format_message(self, status: str, metrics: dict, run_id: str | None, config: dict) -> str:
        """
        Format notification message.

        Parameters
        ----------
        status : str
            Pipeline status
        metrics : dict
            Model metrics
        run_id : str | None
            MLflow run ID
        config : dict
            Configuration

        Returns
        -------
        str
            Formatted message
        """
        model_type = config.get("model", {}).get("type", "unknown")
        timestamp = datetime.now().isoformat()

        if status == "success":
            test_rmse = metrics.get("test", {}).get("rmse", 0.0)
            test_r2 = metrics.get("test", {}).get("r2", 0.0)
            test_mae = metrics.get("test", {}).get("mae", 0.0)

            return f"""
{'='*70}
✅ TRAINING COMPLETED SUCCESSFULLY
{'='*70}

Model Type:     {model_type}
Model Version:  {config.get('model', {}).get('version', 'unknown')}

Test Metrics:
  RMSE:  {test_rmse:.4f} kWh
  R²:    {test_r2:.4f}
  MAE:   {test_mae:.4f} kWh

MLflow Run ID:  {run_id}
Timestamp:      {timestamp}

{'='*70}
"""
        else:
            val_rmse = metrics.get("val", {}).get("rmse", 0.0)
            val_r2 = metrics.get("val", {}).get("r2", 0.0)
            threshold_rmse = config.get("thresholds", {}).get("rmse", 0.0)
            threshold_r2 = config.get("thresholds", {}).get("r2", 0.0)

            return f"""
{'='*70}
⚠️  TRAINING COMPLETED - MODEL FAILED THRESHOLD
{'='*70}

Model Type:     {model_type}
Model Version:  {config.get('model', {}).get('version', 'unknown')}

Validation Metrics:
  RMSE:  {val_rmse:.4f} kWh (threshold: {threshold_rmse:.4f})
  R²:    {val_r2:.4f} (threshold: {threshold_r2:.4f})

Status:         Model not registered in MLflow
Reason:         Performance below threshold
Timestamp:      {timestamp}

{'='*70}
"""

    def _send_console(self, message: str) -> None:
        """
        Send notification to console via logger.

        Parameters
        ----------
        message : str
            Formatted message
        """
        logger.info(message)

    def _send_file(self, message: str, config: dict) -> None:
        """
        Send notification to log file.

        Parameters
        ----------
        message : str
            Formatted message
        config : dict
            Configuration
        """
        try:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Append to log file with UTF-8 encoding to support emojis
            log_file = log_dir / "training_notifications.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{message}\n")

            logger.debug(f"Notification written to file: {log_file}")

        except Exception as e:
            logger.warning(f"Failed to write notification to file: {e}")

    def _send_slack(self, message: str, config: dict) -> None:
        """
        Send notification to Slack via webhook.

        Parameters
        ----------
        message : str
            Formatted message
        config : dict
            Configuration with Slack webhook URL

        Notes
        -----
        This is a placeholder for future implementation.
        Requires 'requests' library and Slack webhook URL.
        """
        # TODO: Implement Slack webhook integration
        logger.info("Slack notification requested but not yet implemented")
        logger.info("To implement: Use requests.post() with webhook_url")

        # Example implementation (commented out):
        # import requests
        # webhook_url = config['notifications']['slack']['webhook_url']
        # payload = {'text': message}
        # requests.post(webhook_url, json=payload)

    def _send_email(self, message: str, config: dict) -> None:
        """
        Send notification via email.

        Parameters
        ----------
        message : str
            Formatted message
        config : dict
            Configuration with email settings

        Notes
        -----
        This is a placeholder for future implementation.
        Requires 'smtplib' configuration and email credentials.
        """
        # TODO: Implement email notification
        logger.info("Email notification requested but not yet implemented")
        logger.info("To implement: Use smtplib or SendGrid/AWS SES")

        # Example implementation (commented out):
        # import smtplib
        # from email.mime.text import MIMEText
        # recipients = config['notifications']['email']['recipients']
        # msg = MIMEText(message)
        # msg['Subject'] = 'Training Pipeline Notification'
        # msg['From'] = 'pipeline@example.com'
        # msg['To'] = ', '.join(recipients)
        # s = smtplib.SMTP('localhost')
        # s.send_message(msg)
        # s.quit()
