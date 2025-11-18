"""
ONNX Model Service for Energy Optimization Copilot API.

This module provides ONNX model inference service with support for
multiple model types and optimized performance.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXModelService:
    """
    ONNX model inference service with support for 8 models.

    Provides optimized inference using ONNX Runtime with GPU support.

    Attributes
    ----------
    model_type : str
        Type of model to use
    model_path : str
        Path to ONNX model
    use_gpu : bool
        Whether to use GPU acceleration
    session : ort.InferenceSession
        ONNX Runtime session
    """

    AVAILABLE_MODELS = {
        "xgboost": "models/onnx/xgboost.onnx",
        "lightgbm": "models/onnx/lightgbm.onnx",
        "catboost": "models/onnx/catboost.onnx",
        "ridge_ensemble": "models/onnx/ridge_ensemble/",
        "lightgbm_ensemble": "models/onnx/lightgbm_ensemble/",
        "chronos2_zeroshot": "models/onnx/chronos2_zeroshot.onnx",
        "chronos2_finetuned": "models/onnx/chronos2_finetuned.onnx",
        "chronos2_covariates": "models/onnx/chronos2_covariates.onnx",
    }

    def __init__(self, model_type: str = "lightgbm_ensemble", use_gpu: bool = True):
        """
        Initialize ONNX model service.

        Parameters
        ----------
        model_type : str, default='lightgbm'
            Type of model to use
        use_gpu : bool, default=True
            Whether to use GPU acceleration

        Raises
        ------
        ValueError
            If model_type is not available
        FileNotFoundError
            If model file does not exist
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_type = model_type
        self.base_model_path = self.AVAILABLE_MODELS[model_type]
        self.model_path = self._resolve_model_path()
        self.use_gpu = use_gpu
        self.session = None
        self.is_ensemble = "ensemble" in model_type
        self.sessions = {}

        logger.info(f"ONNXModelService initialized with model_type={model_type}, use_gpu={use_gpu}")
        logger.info(f"Using models from: {self.model_path}")

    def _resolve_model_path(self) -> Path:
        """
        Resolve model path with fallback strategy.

        Priority:
        1. External volume: /app/models/external
        2. Environment variable: MODEL_PATH
        3. Embedded models: models/onnx (default)

        Returns
        -------
        Path
            Resolved model path

        Notes
        -----
        This allows updating models without rebuilding the Docker image
        by mounting a volume at /app/models/external.
        """
        # Check external volume first (for updates without rebuild)
        external_path = Path("/app/models/external")
        if external_path.exists():
            model_file = external_path / Path(self.base_model_path).name
            if model_file.exists() or (external_path / Path(self.base_model_path).name).is_dir():
                logger.info(f"Using external models from volume: {external_path}")
                return external_path / Path(self.base_model_path).name

        # Check environment variable
        env_path = os.getenv("MODEL_PATH")
        if env_path:
            path = Path(env_path) / Path(self.base_model_path).name
            if path.exists():
                logger.info(f"Using models from MODEL_PATH env var: {path}")
                return path

        # Fallback to embedded models
        embedded_path = Path(self.base_model_path)
        logger.info(f"Using embedded models: {embedded_path}")
        return embedded_path

    def load_model(self) -> None:
        """
        Load ONNX model with GPU support.

        Raises
        ------
        FileNotFoundError
            If model file does not exist
        RuntimeError
            If model loading fails
        """
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )

        if self.is_ensemble:
            self._load_ensemble(providers)
        else:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            try:
                self.session = ort.InferenceSession(str(model_path), providers=providers)
                logger.info(f"Loaded ONNX model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def _load_ensemble(self, providers: list[str]) -> None:
        """
        Load ensemble models (base models + meta-model).

        Parameters
        ----------
        providers : list[str]
            ONNX Runtime execution providers
        """
        ensemble_dir = Path(self.model_path)
        if not ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

        ensemble_type = "ridge" if "ridge" in self.model_type else "lightgbm"

        base_model_names = ["xgboost", "lightgbm", "catboost"]
        for base_name in base_model_names:
            base_path = ensemble_dir / f"{ensemble_type}_base_{base_name}.onnx"
            if base_path.exists():
                self.sessions[f"base_{base_name}"] = ort.InferenceSession(
                    str(base_path), providers=providers
                )
                logger.info(f"Loaded base model: {base_name}")

        meta_path = ensemble_dir / f"{ensemble_type}_meta.onnx"
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta-model not found: {meta_path}")

        self.sessions["meta"] = ort.InferenceSession(str(meta_path), providers=providers)
        logger.info("Loaded meta-model")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on ONNX model.

        Parameters
        ----------
        features : np.ndarray
            Input features of shape (batch_size, num_features)

        Returns
        -------
        np.ndarray
            Predictions

        Raises
        ------
        RuntimeError
            If model is not loaded or prediction fails
        """
        if self.session is None and not self.sessions:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            if self.is_ensemble:
                return self._predict_ensemble(features)
            else:
                input_name = self.session.get_inputs()[0].name
                predictions = self.session.run(None, {input_name: features.astype(np.float32)})[0]

                if predictions.ndim > 1:
                    predictions = predictions.flatten()

                return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def _predict_ensemble(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on ensemble model.

        Parameters
        ----------
        features : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predictions from ensemble
        """
        base_predictions = []

        for base_name in ["xgboost", "lightgbm", "catboost"]:
            session_key = f"base_{base_name}"
            if session_key in self.sessions:
                session = self.sessions[session_key]
                input_name = session.get_inputs()[0].name
                pred = session.run(None, {input_name: features.astype(np.float32)})[0]

                if pred.ndim > 1:
                    pred = pred.flatten()

                base_predictions.append(pred)

        base_preds_array = np.column_stack(base_predictions)

        meta_session = self.sessions["meta"]
        input_name = meta_session.get_inputs()[0].name
        final_predictions = meta_session.run(
            None, {input_name: base_preds_array.astype(np.float32)}
        )[0]

        if final_predictions.ndim > 1:
            final_predictions = final_predictions.flatten()

        return final_predictions

    @classmethod
    def list_available_models(cls) -> list[str]:
        """
        List all available ONNX models.

        Returns
        -------
        list[str]
            List of model names
        """
        return list(cls.AVAILABLE_MODELS.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> dict[str, Any]:
        """
        Get information about a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        dict[str, Any]
            Model information

        Raises
        ------
        ValueError
            If model_name is not available
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = Path(cls.AVAILABLE_MODELS[model_name])

        model_type = "ensemble" if "ensemble" in model_name else "gradient_boosting"
        if "chronos" in model_name:
            model_type = "foundation"

        return {
            "name": model_name,
            "path": str(model_path),
            "exists": model_path.exists(),
            "type": model_type,
            "is_ensemble": "ensemble" in model_name,
        }


onnx_service_cache: dict[str, ONNXModelService] = {}


def get_onnx_service(
    model_type: str = "lightgbm_ensemble", use_gpu: bool = True
) -> ONNXModelService:
    """
    Get or create ONNX model service (singleton pattern).

    Parameters
    ----------
    model_type : str, default='lightgbm'
        Type of model to use
    use_gpu : bool, default=True
        Whether to use GPU acceleration

    Returns
    -------
    ONNXModelService
        ONNX model service instance
    """
    cache_key = f"{model_type}_{use_gpu}"

    if cache_key not in onnx_service_cache:
        service = ONNXModelService(model_type=model_type, use_gpu=use_gpu)
        service.load_model()
        onnx_service_cache[cache_key] = service
        logger.info(f"Created new ONNX service: {cache_key}")
    else:
        logger.debug(f"Using cached ONNX service: {cache_key}")

    return onnx_service_cache[cache_key]
