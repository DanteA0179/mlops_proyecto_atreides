"""
Model predictor for Energy Optimization API.

Single Responsibility: Making predictions with loaded models.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Predictor:
    """
    Make predictions using loaded ML models.

    Single Responsibility: Inference only.
    Supports single and batch predictions.

    Parameters
    ----------
    model : any
        Loaded ML model with predict() method

    Examples
    --------
    >>> from src.api.services.predictor import Predictor
    >>> predictor = Predictor(model)
    >>> prediction = predictor.predict_single(features)
    >>> predictions = predictor.predict_batch(features_list)
    """

    def __init__(self, model):
        """Initialize predictor with model."""
        if model is None:
            raise ValueError("Model cannot be None")

        if not hasattr(model, "predict"):
            raise ValueError("Model must have predict() method")

        self.model = model
        logger.debug("Predictor initialized")

    def predict_single(self, features: np.ndarray) -> float:
        """
        Predict single instance.

        Parameters
        ----------
        features : np.ndarray
            Input features array, shape (n_features,) or (1, n_features)

        Returns
        -------
        float
            Prediction value

        Raises
        ------
        RuntimeError
            If prediction fails
        """
        try:
            # Ensure 2D array for sklearn compatibility
            if features.ndim == 1:
                features = features.reshape(1, -1)

            predictions = self.model.predict(features)
            result = float(predictions[0]) if len(predictions) > 0 else 0.0

            logger.debug(f"Single prediction made: {result:.4f}")
            return result

        except Exception as e:
            logger.error(f"Single prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}") from e

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Predict batch of instances.

        Parameters
        ----------
        features : np.ndarray
            Input features array, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions array, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If prediction fails
        """
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            predictions = self.model.predict(features)
            logger.debug(f"Batch prediction made: {len(predictions)} samples")
            return predictions

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch prediction failed: {e}") from e

    def predict_with_confidence(
        self, features: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, tuple[np.ndarray | None, np.ndarray | None]]:
        """
        Predict with confidence intervals (if model supports it).

        Parameters
        ----------
        features : np.ndarray
            Input features array
        alpha : float, default=0.05
            Significance level for confidence intervals (95% CI)

        Returns
        -------
        tuple
            (predictions, (lower_bounds, upper_bounds))
            Confidence bounds are None if model doesn't support intervals

        Notes
        -----
        Currently returns None for confidence bounds.
        Future: implement quantile regression or bootstrapping.
        """
        predictions = self.predict_batch(features)

        # Check if model supports prediction intervals
        if hasattr(self.model, "predict_interval"):
            try:
                intervals = self.model.predict_interval(features, alpha=alpha)
                return predictions, intervals
            except Exception as e:
                logger.warning(f"Interval prediction not available: {e}")

        return predictions, (None, None)

    @property
    def supports_intervals(self) -> bool:
        """
        Check if model supports confidence intervals.

        Returns
        -------
        bool
            True if model supports intervals, False otherwise
        """
        return hasattr(self.model, "predict_interval") or hasattr(self.model, "predict_proba")
