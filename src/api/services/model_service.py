"""
Model service orchestrator for Energy Optimization API.

Coordinates model loading, validation, and prediction using composition.
Follows Single Responsibility Principle with separate classes for each concern.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.api.services.feature_validator import FeatureValidator
from src.api.services.model_loader import ModelLoader
from src.api.services.predictor import Predictor

logger = logging.getLogger(__name__)


class ModelService:
    """
    Orchestrate model operations using composition.

    Single Responsibility: Coordinate between loader, validator, and predictor.
    Delegates specific responsibilities to specialized classes.

    Supports:
    - Traditional ML: XGBoost, LightGBM, CatBoost
    - Ensembles: Stacking, Voting
    - Foundation Models: Chronos-2 (3 variants)

    Parameters
    ----------
    model_type : str, default="stacking_ensemble"
        Type of model to load from supported models
    mlflow_tracking_uri : str, default="http://localhost:5000"
        MLflow tracking server URI
    expected_features : list[str], optional
        List of expected feature names for validation

    Examples
    --------
    >>> from src.api.services.model_service import ModelService
    >>> service = ModelService(model_type="stacking_ensemble")
    >>> service.load_model()
    >>> prediction = service.predict(features)
    """

    SUPPORTED_MODELS = {
        # Traditional ML (from US-013, US-015)
        "xgboost": "models/baselines/xgboost_model.pkl",
        "lightgbm": "models/gradient_boosting/lightgbm_model.pkl",
        "catboost": "models/gradient_boosting/catboost_model.pkl",
        # Ensembles (from US-015)
        "stacking_ensemble": "models/ensembles/ensemble_lightgbm_v1.pkl",
        "voting_ensemble": "models/ensembles/voting_ensemble_v1.pkl",
        # Foundation Models (from US-019)
        "chronos2_zeroshot": "models/chronos/chronos2_zeroshot.pkl",
        "chronos2_finetuned": "models/chronos/chronos2_finetuned.pkl",
        "chronos2_covariates": "models/chronos/chronos2_covariates.pkl",
    }

    def __init__(
        self,
        model_type: str = "stacking_ensemble",
        mlflow_tracking_uri: str = "http://localhost:5000",
        expected_features: list[str] | None = None,
    ):
        """Initialize model service with dependency injection."""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_path = Path(self.SUPPORTED_MODELS[model_type])

        # Composition: delegate responsibilities to specialized classes
        self.loader = ModelLoader(mlflow_tracking_uri)
        self.predictor: Predictor | None = None
        self.validator: FeatureValidator | None = None
        self.scaler = None
        self.metadata = None

        # Setup validator if features provided
        if expected_features:
            self.validator = FeatureValidator(expected_features)

        logger.info(f"ModelService initialized with model_type={model_type}")

    def load_model(self) -> None:
        """
        Load model, metadata, and scaler from disk.

        For Chronos-2 models, loads from Dagster pipeline artifacts.
        For traditional models, loads from MLflow/local storage.

        Raises
        ------
        FileNotFoundError
            If model file not found
        RuntimeError
            If model loading fails
        """
        # Delegate loading to ModelLoader
        model = self.loader.load_from_disk(self.model_path)

        # Initialize predictor with loaded model
        self.predictor = Predictor(model)

        # Load metadata (JSON)
        metadata_path = self.model_path.with_suffix(".json")
        self.metadata = self.loader.load_metadata(metadata_path)

        # For traditional models, load scaler
        if self.model_type in ["xgboost", "lightgbm", "catboost"]:
            scaler_path = Path("models/preprocessing/scaler.pkl")
            self.scaler = self.loader.load_scaler(scaler_path)

        logger.info(f"Model loaded successfully from {self.model_path}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make prediction using loaded model.

        Parameters
        ----------
        features : np.ndarray
            Input features array, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions array

        Raises
        ------
        RuntimeError
            If model not loaded or prediction fails
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply scaler if available
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Delegate prediction to Predictor
        return self.predictor.predict_batch(features)

    def predict_interval(
        self, features: np.ndarray, alpha: float = 0.05
    ) -> tuple[float | None, float | None]:
        """
        Calculate confidence intervals for predictions.

        Parameters
        ----------
        features : np.ndarray
            Input features array
        alpha : float, default=0.05
            Significance level (default 95% CI)

        Returns
        -------
        tuple[float | None, float | None]
            Lower and upper confidence bounds (None if not supported)

        Notes
        -----
        Currently returns None for models without interval support.
        Future implementation can add quantile regression or bootstrapping.
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply scaler if available
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Delegate to predictor
        _, intervals = self.predictor.predict_with_confidence(features, alpha)
        return intervals

    def get_mlflow_info(self) -> dict | None:
        """
        Get MLflow experiment info for current model.

        Returns
        -------
        dict | None
            MLflow experiment information or None if not found

        Examples
        --------
        >>> model_service.get_mlflow_info()
        {'experiment_id': '1', 'experiment_name': '...', 'artifact_location': '...'}
        """
        experiments = {
            "xgboost": "steel_energy_xgboost_baseline",
            "lightgbm": "steel_energy_lightgbm_baseline",
            "catboost": "steel_energy_catboost_baseline",
            "stacking_ensemble": "steel_energy_stacking_ensemble",
            "chronos2_zeroshot": "chronos2_zeroshot",
            "chronos2_finetuned": "chronos2_finetuned",
            "chronos2_covariates": "chronos2_covariates",
        }

        exp_name = experiments.get(self.model_type)
        if exp_name:
            return self.loader.get_mlflow_experiment_info(exp_name)
        return None

    @property
    def model_version(self) -> str:
        """
        Get model version string.

        Returns
        -------
        str
            Model version identifier
        """
        if self.metadata and "version" in self.metadata:
            return self.metadata["version"]
        return f"{self.model_type}_v1"

    @property
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns
        -------
        bool
            True if model is loaded, False otherwise
        """
        return self.predictor is not None

    def validate_features(self, features: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Validate input features against schema.

        Parameters
        ----------
        features : dict[str, any]
            Feature dictionary to validate

        Returns
        -------
        tuple[bool, str | None]
            (is_valid, error_message)
        """
        if self.validator is None:
            logger.debug("No validator configured, skipping validation")
            return True, None

        return self.validator.validate(features)
