"""
Training script for Stacking Ensemble.

This script combines multiple trained models (XGBoost, LightGBM, CatBoost)
using a stacking ensemble approach with a meta-model.

Steps:
    1. Load preprocessed data
    2. Load trained base models
    3. Create stacking ensemble with meta-model (Ridge or LightGBM)
    4. Train ensemble using out-of-fold predictions
    5. Evaluate on validation and test sets
    6. Compare with base models
    7. Analyze base model contributions
    8. Save ensemble and results
    9. Log everything to MLflow

Usage:
    poetry run python src/models/train_ensemble.py --meta-model ridge --cv-folds 5 --model-version v1
    poetry run python src/models/train_ensemble.py --meta-model lightgbm --model-version v2
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

import mlflow

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.stacking_ensemble import StackingEnsemble
from src.utils.mlflow_utils import setup_mlflow_experiment
from src.utils.model_evaluation import (
    calculate_regression_metrics,
    plot_predictions_vs_actual,
    plot_residuals,
)

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train stacking ensemble of gradient boosting models"
    )

    parser.add_argument(
        "--meta-model",
        type=str,
        default="ridge",
        choices=["ridge", "lightgbm"],
        help="Meta-model type: ridge or lightgbm (default: ridge)",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for out-of-fold predictions (default: 5)",
    )

    parser.add_argument(
        "--model-version", type=str, required=True, help="Model version for tracking and saving"
    )

    parser.add_argument(
        "--base-models-dir",
        type=Path,
        default=Path("models/gradient_boosting"),
        help="Directory containing trained base models",
    )

    return parser.parse_args()


def load_preprocessed_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load preprocessed train, validation, and test datasets.

    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    logger.info("Loading preprocessed data")

    data_dir = Path("data/processed")

    train_df = pl.read_parquet(data_dir / "steel_preprocessed_train.parquet")
    val_df = pl.read_parquet(data_dir / "steel_preprocessed_val.parquet")
    test_df = pl.read_parquet(data_dir / "steel_preprocessed_test.parquet")

    logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    return train_df, val_df, test_df


def load_base_models(base_models_dir: Path) -> dict:
    """
    Load trained base models from disk.

    Parameters
    ----------
    base_models_dir : Path
        Directory containing base model files

    Returns
    -------
    dict
        Dictionary of loaded models {name: model}
    """
    logger.info("Loading base models")

    base_models = {}

    # Define model files to load
    model_files = {
        "xgboost": "xgboost_model.pkl",
        "lightgbm": "lightgbm_model.pkl",
        "catboost": "catboost_model.pkl",
    }

    for name, filename in model_files.items():
        filepath = base_models_dir / filename

        if filepath.exists():
            logger.info(f"Loading {name} from {filepath}")
            model = joblib.load(filepath)

            # Fix XGBoost GPU parameters for XGBoost 3.x compatibility
            if name == "xgboost":
                # Handle both Pipeline and direct XGBRegressor
                xgb_model = model
                if hasattr(model, 'named_steps'):  # It's a Pipeline
                    # Get the final estimator from the pipeline
                    xgb_model = model.steps[-1][1]

                if hasattr(xgb_model, 'get_params'):
                    params = xgb_model.get_params()
                    if params.get('tree_method') == 'gpu_hist':
                        logger.warning("Updating deprecated 'gpu_hist' to 'hist' with device='cuda'")
                        xgb_model.set_params(tree_method='hist', device='cuda')

            base_models[name] = model
        else:
            logger.warning(f"Model file not found: {filepath}")
            logger.warning(f"Skipping {name}")

    if not base_models:
        raise ValueError("No base models found. Train base models first.")

    logger.info(f"Loaded {len(base_models)} base models: {list(base_models.keys())}")

    return base_models


def create_meta_model(meta_model_type: str):
    """
    Create meta-model for stacking.

    Parameters
    ----------
    meta_model_type : str
        Type of meta-model ('ridge' or 'lightgbm')

    Returns
    -------
    estimator
        Meta-model instance
    """
    logger.info(f"Creating meta-model: {meta_model_type}")

    if meta_model_type == "ridge":
        meta_model = Ridge(alpha=1.0, random_state=42)
        logger.info("Ridge regression with alpha=1.0")

    elif meta_model_type == "lightgbm":
        meta_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        logger.info("LightGBM with shallow trees (max_depth=3)")

    else:
        raise ValueError(f"Unknown meta-model type: {meta_model_type}")

    return meta_model


def prepare_data_for_training(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple:
    """
    Prepare data for ensemble training.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data
    val_df : pl.DataFrame
        Validation data
    test_df : pl.DataFrame
        Test data

    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Preparing data for training")

    # Separate features and target
    target_col = "Usage_kWh"

    X_train = train_df.drop(target_col).to_numpy()
    y_train = train_df[target_col].to_numpy()

    X_val = val_df.drop(target_col).to_numpy()
    y_val = val_df[target_col].to_numpy()

    X_test = test_df.drop(target_col).to_numpy()
    y_test = test_df[target_col].to_numpy()

    logger.info(f"Features: {train_df.drop(target_col).columns}")
    logger.info(f"X_train shape: {X_train.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_ensemble_model(
    base_models: dict,
    meta_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int,
) -> StackingEnsemble:
    """
    Train stacking ensemble.

    Parameters
    ----------
    base_models : dict
        Dictionary of base models
    meta_model : estimator
        Meta-model for stacking
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    cv_folds : int
        Number of CV folds

    Returns
    -------
    StackingEnsemble
        Trained ensemble
    """
    logger.info("=" * 70)
    logger.info("TRAINING STACKING ENSEMBLE")
    logger.info("=" * 70)

    start_time = time.time()

    # Create ensemble
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        cv_folds=cv_folds,
    )

    # Train ensemble
    ensemble.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds")

    return ensemble


def evaluate_all_models(
    base_models: dict,
    ensemble: StackingEnsemble,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate all models on validation and test sets.

    Parameters
    ----------
    base_models : dict
        Base models
    ensemble : StackingEnsemble
        Trained ensemble
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target

    Returns
    -------
    dict
        Results for all models on both sets
    """
    logger.info("=" * 70)
    logger.info("EVALUATING ALL MODELS")
    logger.info("=" * 70)

    results = {
        "validation": {},
        "test": {},
    }

    # Evaluate on validation set
    logger.info("\nValidation Set:")
    for name, model in base_models.items():
        y_pred = model.predict(X_val)
        metrics = calculate_regression_metrics(y_val, y_pred)
        results["validation"][name] = metrics
        logger.info(f"{name:15s} - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

    # Ensemble on validation
    y_pred_ensemble = ensemble.predict(X_val)
    metrics_ensemble = calculate_regression_metrics(y_val, y_pred_ensemble)
    results["validation"]["ensemble"] = metrics_ensemble
    logger.info(
        f"{'ensemble':15s} - RMSE: {metrics_ensemble['rmse']:.4f}, R2: {metrics_ensemble['r2']:.4f}"
    )

    # Evaluate on test set
    logger.info("\nTest Set:")
    for name, model in base_models.items():
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)
        results["test"][name] = metrics
        logger.info(f"{name:15s} - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

    # Ensemble on test
    y_pred_ensemble = ensemble.predict(X_test)
    metrics_ensemble = calculate_regression_metrics(y_test, y_pred_ensemble)
    results["test"]["ensemble"] = metrics_ensemble
    logger.info(
        f"{'ensemble':15s} - RMSE: {metrics_ensemble['rmse']:.4f}, R2: {metrics_ensemble['r2']:.4f}"
    )

    logger.info("=" * 70)

    return results


def save_ensemble_artifacts(
    ensemble: StackingEnsemble,
    results: dict,
    model_version: str,
    meta_model_type: str,
) -> dict:
    """
    Save ensemble model and results.

    Parameters
    ----------
    ensemble : StackingEnsemble
        Trained ensemble
    results : dict
        Evaluation results
    model_version : str
        Model version
    meta_model_type : str
        Type of meta-model

    Returns
    -------
    dict
        Paths to saved artifacts
    """
    logger.info("Saving ensemble artifacts")

    # Create directories
    models_dir = Path("models/ensembles")
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Generate filenames with meta-model type
    version_str = f"{meta_model_type}_{model_version}"

    # Save ensemble model
    model_path = models_dir / f"ensemble_{version_str}.pkl"
    ensemble.save(model_path)

    # Save metrics
    metrics_path = metrics_dir / f"ensemble_metrics_{version_str}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved ensemble to {model_path}")
    logger.info(f"Saved metrics to {metrics_path}")

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
    }


def create_comparison_visualizations(
    base_models: dict,
    ensemble: StackingEnsemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_version: str,
    meta_model_type: str,
) -> list[Path]:
    """
    Create comparison visualizations.

    Parameters
    ----------
    base_models : dict
        Base models
    ensemble : StackingEnsemble
        Trained ensemble
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    model_version : str
        Model version
    meta_model_type : str
        Type of meta-model

    Returns
    -------
    list
        Paths to visualization files
    """
    logger.info("Creating comparison visualizations")

    plots_dir = Path("reports/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)

    version_str = f"{meta_model_type}_{model_version}"
    plot_paths = []

    # Actual vs Predicted for ensemble
    y_pred_ensemble = ensemble.predict(X_test)

    plot_path = plots_dir / f"ensemble_actual_vs_predicted_{version_str}.png"
    plot_predictions_vs_actual(
        y_true=y_test,
        y_pred=y_pred_ensemble,
        title=f"Ensemble Predictions - {meta_model_type.upper()}",
        save_path=plot_path,
    )
    plot_paths.append(plot_path)
    logger.info(f"Saved plot: {plot_path}")

    # Residuals plot
    plot_path = plots_dir / f"ensemble_residuals_{version_str}.png"
    plot_residuals(
        y_true=y_test,
        y_pred=y_pred_ensemble,
        save_path=plot_path,
    )
    plot_paths.append(plot_path)
    logger.info(f"Saved plot: {plot_path}")

    # Base model contributions
    contributions = ensemble.get_base_model_contributions()

    if contributions:
        import matplotlib.pyplot as plt

        plot_path = plots_dir / f"ensemble_contributions_{version_str}.png"

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(contributions.keys())
        values = list(contributions.values())

        ax.barh(models, values, color="steelblue")
        ax.set_xlabel("Contribution")
        ax.set_title(f"Base Model Contributions - {meta_model_type.upper()}")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        plot_paths.append(plot_path)
        logger.info(f"Saved plot: {plot_path}")

    return plot_paths


def log_to_mlflow(
    ensemble: StackingEnsemble,
    results: dict,
    contributions: dict,
    cv_folds: int,
    meta_model_type: str,
    model_version: str,
    artifact_paths: dict,
    plot_paths: list,
) -> None:
    """
    Log ensemble training to MLflow.

    Parameters
    ----------
    ensemble : StackingEnsemble
        Trained ensemble
    results : dict
        Evaluation results
    contributions : dict
        Base model contributions
    cv_folds : int
        Number of CV folds
    meta_model_type : str
        Type of meta-model
    model_version : str
        Model version
    artifact_paths : dict
        Paths to saved artifacts
    plot_paths : list
        Paths to plots
    """
    logger.info("Logging to MLflow")

    with mlflow.start_run(run_name=f"{meta_model_type}_{model_version}"):
        # Log parameters
        mlflow.log_param("meta_model_type", meta_model_type)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("n_base_models", len(ensemble.base_models_))
        mlflow.log_param("base_models", list(ensemble.base_models_.keys()))

        # Log meta-model parameters
        if hasattr(ensemble.meta_model_, "get_params"):
            for key, value in ensemble.meta_model_.get_params().items():
                mlflow.log_param(f"meta_{key}", value)

        # Log metrics (validation)
        for model_name, metrics in results["validation"].items():
            mlflow.log_metric(f"val_{model_name}_rmse", metrics["rmse"])
            mlflow.log_metric(f"val_{model_name}_mae", metrics["mae"])
            mlflow.log_metric(f"val_{model_name}_r2", metrics["r2"])
            mlflow.log_metric(f"val_{model_name}_mape", metrics["mape"])

        # Log metrics (test)
        for model_name, metrics in results["test"].items():
            mlflow.log_metric(f"test_{model_name}_rmse", metrics["rmse"])
            mlflow.log_metric(f"test_{model_name}_mae", metrics["mae"])
            mlflow.log_metric(f"test_{model_name}_r2", metrics["r2"])
            mlflow.log_metric(f"test_{model_name}_mape", metrics["mape"])

        # Log base model contributions
        for model_name, contribution in contributions.items():
            mlflow.log_metric(f"contribution_{model_name}", contribution)

        # Log ensemble model
        mlflow.sklearn.log_model(ensemble, "ensemble_model")

        # Log artifacts
        for _artifact_name, path in artifact_paths.items():
            mlflow.log_artifact(path)

        # Log plots
        for plot_path in plot_paths:
            mlflow.log_artifact(str(plot_path))

        logger.info("MLflow logging completed")


def main():
    """Main training pipeline."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("STACKING ENSEMBLE TRAINING")
    logger.info("=" * 70)
    logger.info(f"Meta-model: {args.meta_model}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info(f"Model version: {args.model_version}")
    logger.info("=" * 70)

    # Setup MLflow experiment
    _ = setup_mlflow_experiment(
        experiment_name="steel_energy_stacking_ensemble", tracking_uri="http://localhost:5000"
    )

    # Step 1: Load data
    train_df, val_df, test_df = load_preprocessed_data()

    # Step 2: Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_training(
        train_df, val_df, test_df
    )

    # Step 3: Load base models
    base_models = load_base_models(args.base_models_dir)

    # Step 4: Create meta-model
    meta_model = create_meta_model(args.meta_model)

    # Step 5: Train ensemble
    ensemble = train_ensemble_model(
        base_models=base_models,
        meta_model=meta_model,
        X_train=X_train,
        y_train=y_train,
        cv_folds=args.cv_folds,
    )

    # Step 6: Evaluate all models
    results = evaluate_all_models(
        base_models=base_models,
        ensemble=ensemble,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # Step 7: Analyze contributions
    contributions = ensemble.get_base_model_contributions()

    # Step 8: Save artifacts
    artifact_paths = save_ensemble_artifacts(
        ensemble=ensemble,
        results=results,
        model_version=args.model_version,
        meta_model_type=args.meta_model,
    )

    # Step 9: Create visualizations
    plot_paths = create_comparison_visualizations(
        base_models=base_models,
        ensemble=ensemble,
        X_test=X_test,
        y_test=y_test,
        model_version=args.model_version,
        meta_model_type=args.meta_model,
    )

    # Step 10: Log to MLflow
    log_to_mlflow(
        ensemble=ensemble,
        results=results,
        contributions=contributions,
        cv_folds=args.cv_folds,
        meta_model_type=args.meta_model,
        model_version=args.model_version,
        artifact_paths=artifact_paths,
        plot_paths=plot_paths,
    )

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
