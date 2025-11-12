"""
Stacking Ensemble implementation for combining multiple models.

This module implements a stacking ensemble approach that combines predictions
from multiple base models (Level 0) using a meta-model (Level 1).

Classes:
    StackingEnsemble: Main class for stacking ensemble

Functions:
    generate_oof_predictions: Generate out-of-fold predictions for training
    train_meta_model: Train the meta-model on base model predictions
    evaluate_ensemble: Evaluate the complete ensemble
    analyze_base_model_contributions: Analyze importance of each base model
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble of multiple regression models.

    This implementation uses out-of-fold predictions from base models
    to train a meta-model, preventing overfitting.

    Parameters
    ----------
    base_models : dict
        Dictionary of base models {name: model}
    meta_model : estimator, default=None
        Meta-model for combining predictions.
        If None, uses Ridge regression with alpha=1.0
    cv_folds : int, default=5
        Number of cross-validation folds for out-of-fold predictions

    Attributes
    ----------
    base_models_ : dict
        Fitted base models
    meta_model_ : estimator
        Fitted meta-model
    oof_predictions_ : np.ndarray
        Out-of-fold predictions used for training meta-model

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> import lightgbm as lgb
    >>> base_models = {
    ...     'xgboost': xgb_model,
    ...     'lightgbm': lgb_model,
    ...     'catboost': cb_model
    ... }
    >>> ensemble = StackingEnsemble(
    ...     base_models=base_models,
    ...     meta_model=Ridge(alpha=1.0),
    ...     cv_folds=5
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: dict[str, Any],
        meta_model: Any = None,
        cv_folds: int = 5,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Fit the stacking ensemble.

        1. Generate out-of-fold predictions from base models
        2. Train meta-model on these predictions

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training target

        Returns
        -------
        self
            Fitted StackingEnsemble instance
        """
        logger.info("Training stacking ensemble")
        logger.info(f"Base models: {list(self.base_models.keys())}")
        logger.info(f"CV folds: {self.cv_folds}")

        # Initialize meta-model if not provided
        if self.meta_model is None:
            self.meta_model = Ridge(alpha=1.0)
            logger.info("Using Ridge regression as meta-model (default)")
        else:
            logger.info(f"Using {type(self.meta_model).__name__} as meta-model")

        # Generate out-of-fold predictions
        logger.info("Generating out-of-fold predictions")
        oof_predictions = self._generate_oof_predictions(X, y)

        # Store for analysis
        self.oof_predictions_ = oof_predictions

        # Train meta-model on out-of-fold predictions
        logger.info("Training meta-model")
        self.meta_model_ = self.meta_model.fit(oof_predictions, y)

        # Train base models on full training set
        logger.info("Retraining base models on full training set")
        self.base_models_ = {}
        for name, model in self.base_models.items():
            logger.info(f"  Training {name}")
            self.base_models_[name] = model.fit(X, y)

        logger.info("Stacking ensemble training completed")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction

        Returns
        -------
        np.ndarray
            Predictions from meta-model
        """
        # Get predictions from each base model
        base_predictions = np.column_stack(
            [model.predict(X) for model in self.base_models_.values()]
        )

        # Use meta-model to combine predictions
        final_predictions = self.meta_model_.predict(base_predictions)

        return final_predictions

    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate out-of-fold predictions for training meta-model.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training target

        Returns
        -------
        np.ndarray
            Out-of-fold predictions (n_samples, n_base_models)
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Initialize array for out-of-fold predictions
        oof_predictions = np.zeros((n_samples, n_models))

        # K-Fold cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"  Generating OOF predictions for {name}")

            for _fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Split data
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]

                # Clone and train model on fold
                model_fold = clone(model)
                model_fold.fit(X_train_fold, y_train_fold)

                # Predict on out-of-fold data
                oof_predictions[val_idx, model_idx] = model_fold.predict(X_val_fold)

            logger.info(f"    Completed {self.cv_folds} folds")

        return oof_predictions

    def get_base_model_contributions(self) -> dict[str, float]:
        """
        Analyze contribution of each base model to final predictions.

        For Ridge meta-model, returns coefficients.
        For LightGBM meta-model, returns feature importance.

        Returns
        -------
        dict
            Contribution scores for each base model
        """
        model_names = list(self.base_models_.keys())

        if hasattr(self.meta_model_, "coef_"):
            # Ridge regression coefficients
            contributions = {
                name: float(coef)
                for name, coef in zip(model_names, self.meta_model_.coef_, strict=True)
            }
            logger.info("Base model contributions (Ridge coefficients):")
        elif hasattr(self.meta_model_, "feature_importances_"):
            # LightGBM or similar - feature importance
            contributions = {
                name: float(imp)
                for name, imp in zip(
                    model_names, self.meta_model_.feature_importances_, strict=True
                )
            }
            logger.info("Base model contributions (feature importance):")
        else:
            logger.warning("Meta-model does not have coefficients or feature importance")
            contributions = {name: 1.0 / len(model_names) for name in model_names}
            logger.info("Base model contributions (equal weights):")

        for name, value in contributions.items():
            logger.info(f"  {name}: {value:.4f}")

        return contributions

    def save(self, filepath: Path) -> None:
        """
        Save the ensemble to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the ensemble
        """
        logger.info(f"Saving ensemble to {filepath}")
        joblib.dump(self, filepath)
        logger.info("Ensemble saved successfully")

    @staticmethod
    def load(filepath: Path) -> "StackingEnsemble":
        """
        Load ensemble from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the ensemble from

        Returns
        -------
        StackingEnsemble
            Loaded ensemble instance
        """
        logger.info(f"Loading ensemble from {filepath}")
        ensemble = joblib.load(filepath)
        logger.info("Ensemble loaded successfully")
        return ensemble


def evaluate_ensemble(
    ensemble: StackingEnsemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate ensemble on test set.

    Parameters
    ----------
    ensemble : StackingEnsemble
        Fitted ensemble
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target

    Returns
    -------
    dict
        Evaluation metrics (RMSE, MAE, R2, MAPE)
    """
    from src.utils.model_evaluation import calculate_regression_metrics

    logger.info("Evaluating ensemble")

    # Make predictions
    y_pred = ensemble.predict(X_test)

    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred)

    logger.info("Ensemble Metrics:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  R2: {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics


def compare_base_models_vs_ensemble(
    base_models: dict[str, Any],
    ensemble: StackingEnsemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compare performance of base models vs ensemble.

    Parameters
    ----------
    base_models : dict
        Dictionary of fitted base models
    ensemble : StackingEnsemble
        Fitted ensemble
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target

    Returns
    -------
    dict
        Metrics for each model and ensemble
    """
    from src.utils.model_evaluation import calculate_regression_metrics

    logger.info("Comparing base models vs ensemble")

    results = {}

    # Evaluate each base model
    for name, model in base_models.items():
        logger.info(f"Evaluating {name}")
        y_pred = model.predict(X_test)
        results[name] = calculate_regression_metrics(y_test, y_pred)

    # Evaluate ensemble
    logger.info("Evaluating ensemble")
    results["ensemble"] = evaluate_ensemble(ensemble, X_test, y_test)

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison:")
    logger.info("=" * 60)
    for model_name, metrics in results.items():
        logger.info(f"{model_name:15s} - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
    logger.info("=" * 60)

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]["rmse"])
    logger.info(f"Best model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")

    return results
