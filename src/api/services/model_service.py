"""
Model Service for Energy Optimization API.

This module provides model loading and prediction functionality,
supporting multiple model types from Dagster pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

try:
    from mlflow.tracking import MlflowClient
except ImportError:
    MlflowClient = None

# Import model classes for pickle deserialization
try:
    from src.models.stacking_ensemble import StackingEnsemble
except ImportError:
    StackingEnsemble = None
    logger.warning("StackingEnsemble not available")

logger = logging.getLogger(__name__)


class ModelService:
    """
    Model service with support for 8 models from Dagster pipeline.

    Supports:
    - Traditional ML: XGBoost, LightGBM, CatBoost
    - Ensembles: Stacking, Voting
    - Foundation Models: Chronos-2 (3 variants)

    Attributes
    ----------
    SUPPORTED_MODELS : Dict[str, str]
        Mapping of model types to file paths
    model_type : str
        Current model type
    model_path : Path
        Path to model file
    model : object
        Loaded model object
    scaler : object
        Loaded scaler (if applicable)
    metadata : dict
        Model metadata
    mlflow_client : mlflow.tracking.MlflowClient
        MLflow client for tracking

    Examples
    --------
    >>> model_service = ModelService(model_type="stacking_ensemble")
    >>> model_service.load_model()
    >>> prediction = model_service.predict(features)
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
    ):
        """
        Initialize model service.

        Parameters
        ----------
        model_type : str, default="stacking_ensemble"
            Type of model to load
        mlflow_tracking_uri : str, default="http://localhost:5000"
            MLflow tracking server URI

        Raises
        ------
        ValueError
            If model_type is not supported
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_path = Path(self.SUPPORTED_MODELS.get(model_type))
        self.model = None
        self.scaler = None
        self.metadata = None

        # Initialize MLflow client if available
        if MlflowClient is not None:
            try:
                self.mlflow_client = MlflowClient(mlflow_tracking_uri)
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow client: {e}")
                self.mlflow_client = None
        else:
            logger.warning("MLflow not available, client disabled")
            self.mlflow_client = None

        logger.info(f"ModelService initialized with model_type={model_type}")

    def load_model(self) -> None:
        """
        Load model from pickle with metadata.

        For Chronos-2 models, loads from Dagster pipeline artifacts.
        For traditional models, loads from MLflow/local storage.

        Raises
        ------
        FileNotFoundError
            If model file not found
        RuntimeError
            If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                f"Run Dagster pipeline to train model first."
            )

        try:
            # Load main model using joblib (compatible with sklearn pipelines and ensembles)
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")

            # Load metadata (JSON)
            metadata_path = self.model_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                logger.info(f"Model metadata loaded from {metadata_path}")

            # For traditional models, load scaler
            if self.model_type in ["xgboost", "lightgbm", "catboost"]:
                scaler_path = Path("models/preprocessing/scaler.pkl")
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Scaler loaded from {scaler_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}")

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
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            predictions = self.model.predict(features)
            logger.debug(f"Prediction made: shape={predictions.shape}")
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_interval(
        self, features: np.ndarray, alpha: float = 0.05
    ) -> Tuple[Optional[float], Optional[float]]:
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
        Tuple[Optional[float], Optional[float]]
            Lower and upper confidence bounds (None if not supported)

        Notes
        -----
        Currently returns None for models without interval support.
        Future implementation can add quantile regression or bootstrapping.
        """
        # For now, return None (not all models support intervals)
        # Future: implement quantile regression or bootstrapping
        logger.debug("Confidence intervals not implemented yet")
        return None, None

    def get_mlflow_info(self) -> Optional[Dict]:
        """
        Get MLflow experiment info for current model.

        Returns
        -------
        Optional[Dict]
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
        if exp_name and self.mlflow_client is not None:
            try:
                experiment = self.mlflow_client.get_experiment_by_name(exp_name)
                if experiment:
                    return {
                        "experiment_id": experiment.experiment_id,
                        "experiment_name": exp_name,
                        "artifact_location": experiment.artifact_location,
                    }
            except Exception as e:
                logger.warning(f"Failed to get MLflow info: {str(e)}")
                return None
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
        return self.model is not None
