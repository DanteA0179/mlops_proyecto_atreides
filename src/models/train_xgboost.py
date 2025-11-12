"""
XGBoost Baseline Model Training Script.

This script trains an XGBoost baseline model with hyperparameter optimization,
cross-validation, and comprehensive evaluation. All results are logged to MLflow.

Usage:
    poetry run python src/models/train_xgboost.py [--n-trials 100] [--cv-folds 5]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Disable MLflow emoji output on Windows to avoid encoding issues
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

import numpy as np
import polars as pl

import mlflow
from src.models.xgboost_trainer import (
    evaluate_model,
    get_feature_names_from_pipeline,
    optimize_xgboost_with_optuna,
    train_xgboost_with_cv,
)
from src.utils.mlflow_utils import (
    log_cv_results,
    log_feature_importance,
    log_model_metrics,
    log_model_params,
    log_system_metrics,
    save_and_log_model,
    setup_mlflow_experiment,
)
from src.utils.model_evaluation import (
    create_evaluation_report,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "baselines"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


def setup_directories() -> None:
    """Create necessary directories for outputs."""
    directories = [MODELS_DIR, FIGURES_DIR, METRICS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directories")


def load_preprocessed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load preprocessed data from US-012.

    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    logger.info("Loading preprocessed data from US-012")

    # Load parquet files
    train_df = pl.read_parquet(DATA_DIR / "steel_preprocessed_train.parquet")
    val_df = pl.read_parquet(DATA_DIR / "steel_preprocessed_val.parquet")
    test_df = pl.read_parquet(DATA_DIR / "steel_preprocessed_test.parquet")

    logger.info(f"Train set: {train_df.shape}")
    logger.info(f"Validation set: {val_df.shape}")
    logger.info(f"Test set: {test_df.shape}")

    # Separate features and target
    feature_cols = [col for col in train_df.columns if col != "Usage_kWh"]
    target_col = "Usage_kWh"

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select(target_col).to_numpy().ravel()

    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select(target_col).to_numpy().ravel()

    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.select(target_col).to_numpy().ravel()

    logger.info(f"Features: {feature_cols}")
    logger.info("Data loaded successfully")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def main(n_trials: int = 100, cv_folds: int = 5, model_version: str = None) -> None:
    """
    Main training pipeline.

    Parameters
    ----------
    n_trials : int, default=100
        Number of Optuna trials for hyperparameter optimization
    cv_folds : int, default=5
        Number of cross-validation folds
    model_version : str, optional
        Model version name (e.g., 'v1', 'v2', 'experiment_1')
        If None, generates timestamp-based version
    """
    try:
        logger.info("Starting XGBoost Baseline Model Training - US-013")

        # Import GPU info from trainer module
        from src.models.xgboost_trainer import DEVICE_INFO, GPU_AVAILABLE

        logger.info(f"Compute device: {DEVICE_INFO}")
        if GPU_AVAILABLE:
            logger.info("GPU acceleration enabled for XGBoost")
        else:
            logger.info("Using CPU for XGBoost training")

        # Generate model version if not provided
        if model_version is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version = f"v_{timestamp}"

        logger.info(f"Model version: {model_version}")

        # Setup directories and MLflow
        setup_directories()
        experiment_id = setup_mlflow_experiment(
            experiment_name="steel_energy_xgboost_baseline",
            tracking_uri="http://localhost:5000",
        )

        # Load preprocessed data
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_preprocessed_data()

        # Start MLflow run with versioned name
        run_name = f"xgboost_baseline_{model_version}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            logger.info(f"MLflow Run URL: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")

            # Add tags
            mlflow.set_tags(
                {
                    "experiment_type": "baseline",
                    "model_type": "xgboost",
                    "model_version": model_version,
                    "dataset_version": "steel_preprocessed_v1",
                    "optimization": "optuna",
                    "user_story": "US-013",
                    "n_trials": n_trials,
                    "cv_folds": cv_folds,
                }
            )

            # Log system metrics
            logger.info("Logging system information")
            log_system_metrics()

            # Hyperparameter optimization
            logger.info(f"Starting hyperparameter optimization with Optuna ({n_trials} trials)")
            start_time = time.time()

            optimization_results = optimize_xgboost_with_optuna(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=n_trials,
                cv_folds=cv_folds,
                random_state=42,
                mlflow_tracking=True,
            )

            optimization_time = time.time() - start_time
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")

            # Log optimization results
            mlflow.log_metric("n_trials", n_trials)
            mlflow.log_metric("optimization_time_seconds", optimization_time)
            mlflow.log_metric("best_trial_number", optimization_results["study"].best_trial.number)

            # Save trials history with version
            trials_path = METRICS_DIR / f"optuna_trials_{model_version}.csv"
            optimization_results["trials_df"].write_csv(trials_path)
            logger.info(f"Saved trials history to {trials_path}")

            # Log trials as artifact
            try:
                mlflow.log_artifact(str(trials_path), artifact_path="optuna")
                logger.info("Logged trials history to MLflow")
            except Exception as e:
                logger.warning(f"Could not log trials to MLflow: {e}")

            # Cross-validation with best params
            logger.info(f"Starting cross-validation ({cv_folds}-fold) with best parameters")
            best_params = optimization_results["best_params"]
            log_model_params(best_params)

            cv_results = train_xgboost_with_cv(
                X_train=X_train,
                y_train=y_train,
                model_params=best_params,
                cv_folds=cv_folds,
                random_state=42,
            )

            # Log CV results
            log_cv_results(cv_results["cv_scores"], cv_results["fold_scores"])
            mlflow.log_metric("training_time_seconds", cv_results["training_time"])

            # Model evaluation on all datasets
            logger.info("Evaluating model on train, validation and test sets")

            # Validation set
            val_metrics = evaluate_model(
                model=cv_results["model"],
                X_test=X_val,
                y_test=y_val,
                dataset_name="validation",
            )
            log_model_metrics(val_metrics, prefix="val_")

            # Test set
            test_metrics = evaluate_model(
                model=cv_results["model"],
                X_test=X_test,
                y_test=y_test,
                dataset_name="test",
            )
            log_model_metrics(test_metrics, prefix="test_")

            # Training set (for comparison)
            train_metrics = evaluate_model(
                model=cv_results["model"],
                X_test=X_train,
                y_test=y_train,
                dataset_name="train",
            )
            log_model_metrics(train_metrics, prefix="train_")

            # Save test metrics to JSON with version
            test_metrics_path = METRICS_DIR / f"xgboost_test_metrics_{model_version}.json"
            import json

            with open(test_metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=2)

            # Log metrics as artifact
            try:
                mlflow.log_artifact(str(test_metrics_path), artifact_path="metrics")
                logger.info("Logged test metrics to MLflow")
            except Exception as e:
                logger.warning(f"Could not log metrics to MLflow: {e}")

            # Extract feature importance (only gain - most informative)
            logger.info("Extracting feature importance")

            # Get transformed feature names
            transformed_features = get_feature_names_from_pipeline(
                cv_results["model"], feature_names
            )

            # Extract importance (gain is most informative, weight/cover are redundant)
            importance_dict = log_feature_importance(
                model=cv_results["model"],
                feature_names=transformed_features,
                importance_type="gain",
            )

            # Create and save plot with version
            fig_path = FIGURES_DIR / f"xgboost_feature_importance_{model_version}.png"
            fig = plot_feature_importance(
                importance_dict=importance_dict,
                top_n=10,
                importance_type="gain",
                save_path=fig_path,
            )

            # Log plot as artifact
            try:
                mlflow.log_artifact(str(fig_path), artifact_path="plots")
                logger.info("Logged feature importance plot to MLflow")
            except Exception as e:
                logger.warning(f"Could not log plot to MLflow: {e}")

            import matplotlib.pyplot as plt
            plt.close(fig)

            # Create evaluation visualizations
            logger.info("Creating evaluation visualizations")

            # Predictions vs Actual with version
            y_test_pred = cv_results["model"].predict(X_test)
            pred_path = FIGURES_DIR / f"xgboost_predictions_{model_version}.png"
            fig_pred = plot_predictions_vs_actual(
                y_true=y_test,
                y_pred=y_test_pred,
                title=f"XGBoost {model_version}: Predictions vs Actual (Test Set)",
                save_path=pred_path,
            )

            # Log plot as artifact
            try:
                mlflow.log_artifact(str(pred_path), artifact_path="plots")
                logger.info("Logged predictions plot to MLflow")
            except Exception as e:
                logger.warning(f"Could not log predictions plot to MLflow: {e}")

            plt.close(fig_pred)

            # Residuals plot with version
            resid_path = FIGURES_DIR / f"xgboost_residuals_{model_version}.png"
            fig_resid = plot_residuals(
                y_true=y_test,
                y_pred=y_test_pred,
                save_path=resid_path,
            )

            # Log plot as artifact
            try:
                mlflow.log_artifact(str(resid_path), artifact_path="plots")
                logger.info("Logged residuals plot to MLflow")
            except Exception as e:
                logger.warning(f"Could not log residuals plot to MLflow: {e}")

            plt.close(fig_resid)

            # Save model to disk with version
            logger.info("Saving model")
            model_path = MODELS_DIR / f"xgboost_{model_version}.pkl"
            model_info = save_and_log_model(
                model=cv_results["model"],
                model_path=model_path,
                artifact_name=f"xgboost_{model_version}",
                log_to_mlflow=True,
            )

            logger.info(f"Model saved to: {model_info['model_path']}")
            logger.info(f"Model checksum: {model_info['md5_checksum']}")

            # Generate evaluation report
            logger.info("Generating evaluation report")

            # Get feature importance for report (use gain)
            importance_dict = log_feature_importance(
                model=cv_results["model"],
                feature_names=transformed_features,
                importance_type="gain",
            )

            report_path = REPORTS_DIR / f"xgboost_evaluation_{model_version}.md"
            create_evaluation_report(
                metrics=test_metrics,
                cv_scores=cv_results["cv_scores"],
                feature_importance=importance_dict,
                output_path=report_path,
                model_name=f"XGBoost Baseline {model_version}",
            )
            logger.info(f"Evaluation report saved to: {report_path}")

            # Log report as artifact
            try:
                mlflow.log_artifact(str(report_path), artifact_path="reports")
                logger.info("Logged evaluation report to MLflow")
            except Exception as e:
                logger.warning(f"Could not log report to MLflow: {e}")

            # Final summary
            logger.info("Training completed successfully")
            logger.info(f"Test Set Results: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}, MAPE={test_metrics['mape']:.2f}%")

            # Check if target met
            target_rmse = 0.205
            if test_metrics["rmse"] < target_rmse:
                logger.info(f"Target met: RMSE ({test_metrics['rmse']:.4f}) < {target_rmse}")
            else:
                gap = ((test_metrics["rmse"] / target_rmse) - 1) * 100
                logger.info(f"Target not met: RMSE ({test_metrics['rmse']:.4f}) is {gap:.2f}% above target ({target_rmse})")

            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Report saved to: {report_path}")
            logger.info(f"MLflow Run: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost baseline model with hyperparameter optimization"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for hyperparameter optimization (default: 100)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version name (e.g., 'v1', 'v2', 'experiment_1'). If not provided, uses timestamp.",
    )

    args = parser.parse_args()

    main(n_trials=args.n_trials, cv_folds=args.cv_folds, model_version=args.model_version)
