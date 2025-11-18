"""
Model loader for Energy Optimization API.

Single Responsibility: Loading models from disk or MLflow.
"""

import json
import logging
from pathlib import Path

import joblib

try:
    from mlflow.tracking import MlflowClient
except ImportError:
    MlflowClient = None

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Load ML models from disk or MLflow registry.

    Single Responsibility: Model Loading only.
    Supports multiple model formats (joblib, pickle) and sources.

    Parameters
    ----------
    mlflow_tracking_uri : str, default="http://localhost:5000"
        MLflow tracking server URI

    Examples
    --------
    >>> from src.api.services.model_loader import ModelLoader
    >>> loader = ModelLoader()
    >>> model = loader.load_from_disk("models/ensemble.pkl")
    >>> metadata = loader.load_metadata("models/ensemble.json")
    """

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        """Initialize model loader with MLflow client."""
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_client = None

        if MlflowClient is not None:
            try:
                self.mlflow_client = MlflowClient(mlflow_tracking_uri)
                logger.info(f"MLflow client initialized: {mlflow_tracking_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow client: {e}")
        else:
            logger.warning("MLflow not available, client disabled")

    def load_from_disk(self, model_path: Path | str):
        """
        Load model from local disk using joblib.

        Parameters
        ----------
        model_path : Path | str
            Path to model file

        Returns
        -------
        any
            Loaded model object

        Raises
        ------
        FileNotFoundError
            If model file does not exist
        RuntimeError
            If model loading fails
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from disk: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def load_metadata(self, metadata_path: Path | str) -> dict | None:
        """
        Load model metadata from JSON file.

        Parameters
        ----------
        metadata_path : Path | str
            Path to metadata JSON file

        Returns
        -------
        dict | None
            Metadata dictionary or None if file not found
        """
        metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            logger.debug(f"Metadata file not found: {metadata_path}")
            return None

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from: {metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def load_from_mlflow(self, run_id: str, artifact_path: str = "model"):
        """
        Load model from MLflow registry.

        Parameters
        ----------
        run_id : str
            MLflow run ID
        artifact_path : str, default="model"
            Artifact path within the run

        Returns
        -------
        any
            Loaded model object

        Raises
        ------
        RuntimeError
            If MLflow client not available or loading fails
        """
        if self.mlflow_client is None:
            raise RuntimeError("MLflow client not available")

        try:
            import mlflow

            model_uri = f"runs:/{run_id}/{artifact_path}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model loaded from MLflow: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise RuntimeError(f"MLflow model loading failed: {e}") from e

    def load_scaler(self, scaler_path: Path | str):
        """
        Load preprocessing scaler from disk.

        Parameters
        ----------
        scaler_path : Path | str
            Path to scaler file

        Returns
        -------
        any
            Loaded scaler object or None if not found
        """
        scaler_path = Path(scaler_path)

        if not scaler_path.exists():
            logger.debug(f"Scaler file not found: {scaler_path}")
            return None

        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from: {scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler from {scaler_path}: {e}")
            return None

    def get_mlflow_experiment_info(self, experiment_name: str) -> dict | None:
        """
        Get MLflow experiment information.

        Parameters
        ----------
        experiment_name : str
            Name of MLflow experiment

        Returns
        -------
        dict | None
            Experiment information or None if not found
        """
        if self.mlflow_client is None:
            return None

        try:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            if experiment:
                return {
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment_name,
                    "artifact_location": experiment.artifact_location,
                }
        except Exception as e:
            logger.warning(f"Failed to get MLflow experiment info: {e}")
            return None

        return None
