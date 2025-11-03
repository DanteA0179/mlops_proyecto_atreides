"""
LightGBM model training and optimization utilities.

This module provides functions for training LightGBM models with cross-validation
and hyperparameter optimization using Optuna.

Functions:
    check_gpu_availability: Check if GPU is available for LightGBM
    create_lightgbm_pipeline: Create sklearn pipeline with LightGBM model
    train_lightgbm_with_cv: Train LightGBM with cross-validation
    optimize_lightgbm_with_optuna: Optimize hyperparameters with Optuna
    evaluate_model: Evaluate model on test set
    get_feature_names_from_pipeline: Extract feature names from pipeline
"""

import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.utils.model_evaluation import calculate_regression_metrics

logger = logging.getLogger(__name__)


def check_gpu_availability() -> tuple[bool, str]:
    """
    Check if GPU is available for LightGBM.

    Returns
    -------
    tuple[bool, str]
        (gpu_available, device_info)

    Examples
    --------
    >>> gpu_available, device_info = check_gpu_availability()
    >>> print(f"GPU: {gpu_available}, Device: {device_info}")
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
            logger.info("GPU detected: %s", gpu_name)
            logger.info("LightGBM will attempt to use GPU with device='gpu'")
            return True, gpu_name
        else:
            logger.info("No GPU detected, using CPU")
            return False, "CPU"

    except Exception as e:
        logger.info("GPU check failed: %s. Using CPU", e)
        return False, "CPU"


# Check GPU availability at module load
GPU_AVAILABLE, DEVICE_INFO = check_gpu_availability()

# Default LightGBM parameters (GPU-aware)
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu" if GPU_AVAILABLE else "cpu",
    "random_state": 42,
    "n_jobs": 1 if GPU_AVAILABLE else -1,
    "verbose": -1,
    "force_col_wise": True,
}

logger.info("LightGBM configured for: %s", DEVICE_INFO)
logger.info("Using device: %s", DEFAULT_PARAMS["device"])

# Optuna search space (optimized for efficiency)
SEARCH_SPACE = {
    "num_leaves": (20, 150),
    "max_depth": (-1, 15),  # -1 = no limit
    "learning_rate": (0.01, 0.3),
    "n_estimators": (50, 300),
    "min_child_samples": (5, 100),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
}


def create_lightgbm_pipeline(
    model_params: dict[str, Any] | None = None,
    use_preprocessing: bool = False,
) -> Pipeline:
    """
    Create sklearn pipeline with LightGBM model.

    Note: preprocessing is already done in US-012, so by default
    we only include the model. Set use_preprocessing=True only
    for demo/testing purposes.

    Parameters
    ----------
    model_params : dict, optional
        LightGBM hyperparameters (if None, uses DEFAULT_PARAMS)
    use_preprocessing : bool, default=False
        Whether to include preprocessing steps (usually False)

    Returns
    -------
    Pipeline
        sklearn pipeline with LightGBM regressor

    Examples
    --------
    >>> params = {'num_leaves': 31, 'learning_rate': 0.1}
    >>> pipeline = create_lightgbm_pipeline(model_params=params)
    >>> pipeline.fit(X_train, y_train)
    """
    try:
        # Merge with default params
        params = DEFAULT_PARAMS.copy()
        if model_params:
            params.update(model_params)

        # Ensure GPU settings are preserved if not explicitly overridden
        if "device" not in model_params and GPU_AVAILABLE:
            params["device"] = "gpu"
            params["n_jobs"] = 1

        # Create LightGBM regressor
        lgb_model = lgb.LGBMRegressor(**params)

        # Create pipeline
        if use_preprocessing:
            from sklearn.preprocessing import StandardScaler

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", lgb_model),
                ]
            )
            logger.info("Created pipeline with preprocessing")
        else:
            # Simple pipeline with just the model
            pipeline = Pipeline([("model", lgb_model)])
            device_type = "GPU" if GPU_AVAILABLE else "CPU"
            logger.info(f"Created pipeline without preprocessing (using {device_type})")

        return pipeline

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise


def train_lightgbm_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict[str, Any],
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train LightGBM with cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    model_params : dict
        LightGBM hyperparameters
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random state for reproducibility

    Returns
    -------
    dict
        Dictionary with:
        - model: trained model (fitted on full training set)
        - cv_scores: dict with mean/std for each metric
        - fold_scores: list of scores per fold
        - training_time: total training time in seconds

    Examples
    --------
    >>> params = {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100}
    >>> results = train_lightgbm_with_cv(X_train, y_train, params, cv_folds=5)
    >>> print(f"CV RMSE: {results['cv_scores']['rmse']['mean']:.4f}")
    """
    try:
        logger.info(f"Starting {cv_folds}-fold cross-validation training")
        start_time = time.time()

        # Create pipeline
        pipeline = create_lightgbm_pipeline(model_params=model_params)

        # Define scoring metrics
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

        # Perform cross-validation
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        # Process CV results
        cv_scores = {}
        fold_scores = []

        for metric in ["rmse", "mae", "r2"]:
            key = f"test_{metric}"
            scores = cv_results[key]

            # For negative metrics (rmse, mae), convert to positive
            if metric in ["rmse", "mae"]:
                scores = -scores

            cv_scores[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

            # Store individual fold scores
            for i, score in enumerate(scores):
                if i >= len(fold_scores):
                    fold_scores.append({})
                fold_scores[i][metric] = float(score)

        logger.info(
            f"CV Results - RMSE: {cv_scores['rmse']['mean']:.4f} ± {cv_scores['rmse']['std']:.4f}"
        )
        logger.info(
            f"CV Results - MAE: {cv_scores['mae']['mean']:.4f} ± {cv_scores['mae']['std']:.4f}"
        )
        logger.info(
            f"CV Results - R2: {cv_scores['r2']['mean']:.4f} ± {cv_scores['r2']['std']:.4f}"
        )

        # Train final model on full training set
        logger.info("Training final model on full training set")
        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return {
            "model": pipeline,
            "cv_scores": cv_scores,
            "fold_scores": fold_scores,
            "training_time": training_time,
        }

    except Exception as e:
        logger.error(f"Failed to train with cross-validation: {e}")
        raise


def optimize_lightgbm_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    mlflow_tracking: bool = True,
) -> dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation target
    n_trials : int, default=100
        Number of Optuna trials
    cv_folds : int, default=5
        Number of cross-validation folds for each trial
    random_state : int, default=42
        Random state for reproducibility
    mlflow_tracking : bool, default=True
        Whether to log trials to MLflow

    Returns
    -------
    dict
        Dictionary with:
        - study: optuna study object
        - best_params: best hyperparameters found
        - best_model: model trained with best params
        - trials_df: polars dataframe with all trials

    Examples
    --------
    >>> results = optimize_lightgbm_with_optuna(
    ...     X_train, y_train, X_val, y_val,
    ...     n_trials=100, cv_folds=5
    ... )
    >>> print(f"Best RMSE: {results['study'].best_value:.4f}")
    >>> print(f"Best params: {results['best_params']}")
    """
    try:
        logger.info(f"Starting Optuna optimization with {n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Sample hyperparameters
            params = {
                "num_leaves": trial.suggest_int("num_leaves", *SEARCH_SPACE["num_leaves"]),
                "max_depth": trial.suggest_int("max_depth", *SEARCH_SPACE["max_depth"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", *SEARCH_SPACE["n_estimators"]),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", *SEARCH_SPACE["min_child_samples"]
                ),
                "subsample": trial.suggest_float("subsample", *SEARCH_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *SEARCH_SPACE["colsample_bytree"]
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", *SEARCH_SPACE["reg_alpha"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *SEARCH_SPACE["reg_lambda"]),
            }

            # Create and train model
            pipeline = create_lightgbm_pipeline(model_params=params)
            pipeline.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = pipeline.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

            return rmse

        # Create Optuna study
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info("Optimization completed!")
        logger.info("Best trial: %d", study.best_trial.number)
        logger.info("Best RMSE: %.4f", study.best_value)
        logger.info("Best params: %s", study.best_params)

        # Train final model with best params
        logger.info("Training final model with best parameters")
        best_params = study.best_params
        best_pipeline = create_lightgbm_pipeline(model_params=best_params)
        best_pipeline.fit(X_train, y_train)

        # Create trials dataframe
        trials_data = []
        for trial in study.trials:
            trial_dict = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            trial_dict.update(trial.params)
            trials_data.append(trial_dict)

        trials_df = pl.DataFrame(trials_data)

        return {
            "study": study,
            "best_params": best_params,
            "best_model": best_pipeline,
            "trials_df": trials_df,
        }

    except Exception as e:
        logger.error(f"Failed to optimize with Optuna: {e}")
        raise


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str = "test",
) -> dict[str, float]:
    """
    Evaluate model on test set.

    Parameters
    ----------
    model : Pipeline
        Trained model pipeline
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    dataset_name : str, default="test"
        Name of the dataset (for logging)

    Returns
    -------
    dict
        Dictionary with RMSE, MAE, R2, MAPE

    Examples
    --------
    >>> metrics = evaluate_model(model, X_test, y_test, dataset_name="test")
    >>> print(f"Test RMSE: {metrics['rmse']:.4f}")
    """
    try:
        logger.info(f"Evaluating model on {dataset_name} set")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)

        logger.info(f"{dataset_name.capitalize()} Metrics:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R2: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")

        return metrics

    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise


def get_feature_names_from_pipeline(pipeline: Pipeline, original_features: list[str]) -> list[str]:
    """
    Get feature names from pipeline, handling preprocessing steps.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    original_features : list of str
        Original feature names before preprocessing

    Returns
    -------
    list of str
        Feature names after preprocessing

    Examples
    --------
    >>> feature_names = get_feature_names_from_pipeline(
    ...     pipeline,
    ...     original_features=['CO2', 'NSM', 'hour']
    ... )
    """
    try:
        # If pipeline has preprocessing steps, get transformed feature names
        if len(pipeline.named_steps) > 1:
            # Try to get feature names from preprocessing steps
            for step_name, step in pipeline.named_steps.items():
                if step_name == "model":
                    break
                if hasattr(step, "get_feature_names_out"):
                    feature_names = step.get_feature_names_out(original_features)
                    return list(feature_names)

        # If no preprocessing or can't get names, use original
        return original_features

    except Exception as e:
        logger.warning(f"Could not get transformed feature names: {e}. Using original names.")
        return original_features
