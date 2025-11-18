"""
Model evaluation utilities for regression models.

This module provides functions for comprehensive model evaluation including
metric calculation, visualization, and reporting.

Functions:
    calculate_regression_metrics: Calculate comprehensive regression metrics
    plot_predictions_vs_actual: Create scatter plot of predictions vs actual
    plot_residuals: Create residual plots (histogram + scatter)
    plot_feature_importance: Create horizontal bar plot of feature importance
    create_evaluation_report: Generate markdown report with evaluation results
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary with metrics:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R-squared score
        - mape: Mean Absolute Percentage Error
        - max_error: Maximum absolute error

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    >>> metrics = calculate_regression_metrics(y_true, y_pred)
    >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    try:
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate MAPE (avoid division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "max_error": float(max_error),
        }

        logger.info(f"Calculated metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Path | None = None,
) -> Figure:
    """
    Create scatter plot of predictions vs actual values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    title : str, default="Predictions vs Actual"
        Plot title
    save_path : Path, optional
        If provided, save plot to this path

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> fig = plot_predictions_vs_actual(
    ...     y_true=y_test,
    ...     y_pred=predictions,
    ...     save_path=Path("reports/figures/pred_vs_actual.png")
    ... )
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")

        # Calculate R2 for annotation
        r2 = r2_score(y_true, y_pred)

        # Labels and title
        ax.set_xlabel("Actual Values (Usage_kWh)", fontsize=12)
        ax.set_ylabel("Predicted Values (Usage_kWh)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add R2 annotation
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()

        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Failed to create predictions vs actual plot: {e}")
        raise


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> Figure:
    """
    Create residual plots (histogram + scatter).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    save_path : Path, optional
        If provided, save plot to this path

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> fig = plot_residuals(
    ...     y_true=y_test,
    ...     y_pred=predictions,
    ...     save_path=Path("reports/figures/residuals.png")
    ... )
    """
    try:
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residual scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors="k", linewidths=0.5)
        axes[0].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0].set_xlabel("Predicted Values (Usage_kWh)", fontsize=12)
        axes[0].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
        axes[0].set_title("Residual Plot", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        # Residual histogram
        axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1].set_xlabel("Residuals (Actual - Predicted)", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Residual Distribution", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Add statistics annotation
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        axes[1].text(
            0.05,
            0.95,
            f"Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}",
            transform=axes[1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()

        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Failed to create residual plots: {e}")
        raise


def plot_feature_importance(
    importance_dict: dict[str, float],
    top_n: int = 10,
    importance_type: str = "gain",
    save_path: Path | None = None,
) -> Figure:
    """
    Create horizontal bar plot of feature importance.

    Parameters
    ----------
    importance_dict : dict
        Dictionary mapping feature names to importance values
    top_n : int, default=10
        Number of top features to display
    importance_type : str, default="gain"
        Type of importance (for labeling): "gain", "weight", or "cover"
    save_path : Path, optional
        If provided, save plot to this path

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> importance = {'CO2': 0.45, 'NSM': 0.30, 'hour': 0.15}
    >>> fig = plot_feature_importance(
    ...     importance,
    ...     top_n=3,
    ...     save_path=Path("reports/figures/feature_importance.png")
    ... )
    """
    try:
        # Sort by importance and take top N
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_items, strict=False)

        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align="center", alpha=0.8, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel(f"Importance ({importance_type.capitalize()})", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(
            f"Top {top_n} Features by {importance_type.capitalize()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for i, v in enumerate(importances):
            ax.text(v + 0.01 * max(importances), i, f"{v:.4f}", va="center")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Failed to create feature importance plot: {e}")
        raise


def create_evaluation_report(
    metrics: dict[str, float],
    cv_scores: dict[str, dict[str, float]],
    feature_importance: dict[str, float],
    output_path: Path,
    model_name: str = "XGBoost Baseline",
) -> None:
    """
    Generate markdown report with evaluation results.

    Parameters
    ----------
    metrics : dict
        Test set metrics dictionary
    cv_scores : dict
        Cross-validation scores dictionary
    feature_importance : dict
        Feature importance dictionary
    output_path : Path
        Path where to save the markdown report
    model_name : str, default="XGBoost Baseline"
        Name of the model for the report

    Examples
    --------
    >>> create_evaluation_report(
    ...     metrics={'rmse': 0.195, 'mae': 0.044, 'r2': 0.91},
    ...     cv_scores={'rmse': {'mean': 0.196, 'std': 0.008}},
    ...     feature_importance={'CO2': 0.45, 'NSM': 0.30},
    ...     output_path=Path("reports/xgboost_evaluation.md")
    ... )
    """
    try:
        # Create report content
        report = f"""# {model_name} - Evaluation Report

## Test Set Performance

| Metric | Value |
|--------|-------|
| RMSE | {metrics.get('rmse', 0):.4f} |
| MAE | {metrics.get('mae', 0):.4f} |
| R² | {metrics.get('r2', 0):.4f} |
| MAPE | {metrics.get('mape', 0):.2f}% |
| Max Error | {metrics.get('max_error', 0):.4f} |

## Cross-Validation Results

| Metric | Mean | Std Dev |
|--------|------|---------|
"""

        for metric_name, stats in cv_scores.items():
            report += f"| {metric_name.upper()} | {stats['mean']:.4f} | {stats['std']:.4f} |\n"

        report += """
## Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
"""

        for i, (feature, importance) in enumerate(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10], 1
        ):
            report += f"| {i} | {feature} | {importance:.4f} |\n"

        report += """
## Model Performance Assessment

"""

        # Add assessment based on metrics
        rmse = metrics.get("rmse", 0)
        target_rmse = 0.205

        if rmse < target_rmse:
            status = "Exceeds Target"
            assessment = f"Model RMSE ({rmse:.4f}) is below the target threshold ({target_rmse:.4f}), successfully meeting project objectives."
        else:
            status = "Below Target"
            gap = ((rmse / target_rmse) - 1) * 100
            assessment = f"Model RMSE ({rmse:.4f}) is {gap:.2f}% above target ({target_rmse:.4f}). Further optimization recommended."

        report += f"**Status**: {status}\n\n{assessment}\n"

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Generated evaluation report at {output_path}")

    except Exception as e:
        logger.error(f"Failed to create evaluation report: {e}")
        raise
