"""
Model Training with MLflow Tracking

This module provides classes and functions for training models with
automatic MLflow experiment tracking.
"""

from pathlib import Path

import joblib
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import mlflow


class ModelTrainer:
    """
    Trains and evaluates models with MLflow tracking.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Model to train
    model_name : str
        Name for MLflow run and logging
    experiment_name : str, default='energy-optimization'
        MLflow experiment name
    tracking_uri : str, optional
        MLflow tracking URI (defaults to ./mlruns)

    Examples
    --------
    >>> from xgboost import XGBRegressor
    >>> from src.models.train_model import ModelTrainer
    >>>
    >>> model = XGBRegressor(n_estimators=100)
    >>> trainer = ModelTrainer(model, 'xgboost_baseline')
    >>> trainer.train(X_train, y_train)
    >>> metrics = trainer.evaluate(X_test, y_test)
    """

    def __init__(self, model, model_name, experiment_name="energy-optimization", tracking_uri=None):
        self.model = model
        self.model_name = model_name
        self.experiment_name = experiment_name

        # Setup MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")

        mlflow.set_experiment(experiment_name)

        self.run_id = None
        self.metrics = {}

    def train(self, X_train, y_train, cv=5, log_model=True):
        """
        Trains the model with MLflow tracking.

        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv : int, default=5
            Number of cross-validation folds
        log_model : bool, default=True
            Whether to log the model to MLflow

        Returns
        -------
        self
        """
        with mlflow.start_run(run_name=self.model_name) as run:
            self.run_id = run.info.run_id

            # Log parameters
            if hasattr(self.model, "get_params"):
                params = self.model.get_params()
                mlflow.log_params(params)

            # Cross-validation
            cv_scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse = -cv_scores

            mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
            mlflow.log_metric("cv_rmse_std", cv_rmse.std())

            # Train on full training set
            self.model.fit(X_train, y_train)

            # Log model
            if log_model:
                mlflow.sklearn.log_model(self.model, "model")

            print(f"Model trained: {self.model_name}")
            print(f"   CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
            print(f"   MLflow Run ID: {self.run_id}")

        return self

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on test set and logs metrics to MLflow.

        Parameters
        ----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target

        Returns
        -------
        dict
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_percent = (rmse / y_test.mean()) * 100

        # Benchmark comparison (CUBIST: RMSE 0.2410)
        improvement = ((0.2410 - rmse) / 0.2410) * 100

        self.metrics = {
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_cv_percent": cv_percent,
            "improvement_over_cubist": improvement,
        }

        # Log to MLflow
        with mlflow.start_run(run_id=self.run_id):
            for metric_name, metric_value in self.metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        print("\nEvaluation Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   CV%: {cv_percent:.2f}%")
        print(f"   Improvement over CUBIST: {improvement:+.2f}%")

        return self.metrics

    def save_model(self, filepath):
        """
        Saves the trained model to disk.

        Parameters
        ----------
        filepath : str or Path
            Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, filepath)
        print(f"Model saved to: {filepath}")

        # Log artifact to MLflow
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(str(filepath))


def train_multiple_models(models_config, X_train, y_train, X_test, y_test):
    """
    Trains multiple models and compares their performance.

    Parameters
    ----------
    models_config : dict
        Dictionary mapping model names to model instances
        Example: {'xgboost': XGBRegressor(), 'lightgbm': LGBMRegressor()}
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data

    Returns
    -------
    dict
        Dictionary mapping model names to their metrics

    Examples
    --------
    >>> from xgboost import XGBRegressor
    >>> from lightgbm import LGBMRegressor
    >>>
    >>> models = {
    ...     'xgboost': XGBRegressor(n_estimators=100),
    ...     'lightgbm': LGBMRegressor(n_estimators=100)
    ... }
    >>> results = train_multiple_models(models, X_train, y_train, X_test, y_test)
    """
    results = {}

    for model_name, model in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        trainer = ModelTrainer(model, model_name)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        results[model_name] = {"trainer": trainer, "metrics": metrics}

    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"{'-'*60}")

    for model_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{model_name:<20} {metrics['test_rmse']:<10.4f} "
            f"{metrics['test_mae']:<10.4f} {metrics['test_r2']:<10.4f}"
        )

    return results
