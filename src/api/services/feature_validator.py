"""
Feature validator for Energy Optimization API.

Single Responsibility: Validating input features.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Validate input features against expected schema.

    Single Responsibility: Input validation only.
    Checks feature names, types, ranges, and missing values.

    Parameters
    ----------
    expected_features : list[str]
        List of expected feature names in order
    feature_ranges : dict[str, tuple[float, float]], optional
        Valid ranges for numeric features {feature: (min, max)}

    Examples
    --------
    >>> from src.api.services.feature_validator import FeatureValidator
    >>> validator = FeatureValidator(
    ...     expected_features=['feature1', 'feature2'],
    ...     feature_ranges={'feature1': (0, 100)}
    ... )
    >>> is_valid, error = validator.validate(features_dict)
    """

    def __init__(
        self,
        expected_features: list[str],
        feature_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        """Initialize validator with expected schema."""
        self.expected_features = expected_features
        self.feature_ranges = feature_ranges or {}
        self.n_features = len(expected_features)
        logger.debug(f"FeatureValidator initialized with {self.n_features} features")

    def validate(self, features: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Validate feature dictionary.

        Parameters
        ----------
        features : dict[str, any]
            Dictionary of feature names to values

        Returns
        -------
        tuple[bool, str | None]
            (is_valid, error_message)
            is_valid is True if validation passes, False otherwise
            error_message is None if valid, error string if invalid

        Examples
        --------
        >>> validator = FeatureValidator(['a', 'b'])
        >>> is_valid, error = validator.validate({'a': 1, 'b': 2})
        >>> assert is_valid is True
        >>> assert error is None
        """
        # Check missing features
        missing_features = set(self.expected_features) - set(features.keys())
        if missing_features:
            error = f"Missing features: {sorted(missing_features)}"
            logger.warning(error)
            return False, error

        # Check extra features
        extra_features = set(features.keys()) - set(self.expected_features)
        if extra_features:
            logger.debug(f"Extra features will be ignored: {sorted(extra_features)}")

        # Check feature values
        for feature in self.expected_features:
            value = features[feature]

            # Check for None/NaN
            if value is None or (isinstance(value, float) and np.isnan(value)):
                error = f"Feature '{feature}' has null value"
                logger.warning(error)
                return False, error

            # Check numeric types
            if not isinstance(value, int | float | np.number):
                error = f"Feature '{feature}' must be numeric, got {type(value)}"
                logger.warning(error)
                return False, error

            # Check ranges if specified
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                if not (min_val <= value <= max_val):
                    error = (
                        f"Feature '{feature}' out of range: {value} "
                        f"not in [{min_val}, {max_val}]"
                    )
                    logger.warning(error)
                    return False, error

        return True, None

    def validate_array(self, features: np.ndarray) -> tuple[bool, str | None]:
        """
        Validate numpy array of features.

        Parameters
        ----------
        features : np.ndarray
            Array of features, shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        tuple[bool, str | None]
            (is_valid, error_message)
        """
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Check shape
        if features.shape[1] != self.n_features:
            error = f"Expected {self.n_features} features, got {features.shape[1]}"
            logger.warning(error)
            return False, error

        # Check for NaN/Inf
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            error = "Features contain NaN or Inf values"
            logger.warning(error)
            return False, error

        # Check ranges for each feature
        for i, feature_name in enumerate(self.expected_features):
            if feature_name in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature_name]
                feature_values = features[:, i]

                if np.any(feature_values < min_val) or np.any(feature_values > max_val):
                    error = (
                        f"Feature '{feature_name}' has values out of range "
                        f"[{min_val}, {max_val}]"
                    )
                    logger.warning(error)
                    return False, error

        return True, None

    def validate_batch(self, features_list: list[dict[str, Any]]) -> tuple[bool, str | None]:
        """
        Validate batch of feature dictionaries.

        Parameters
        ----------
        features_list : list[dict[str, any]]
            List of feature dictionaries

        Returns
        -------
        tuple[bool, str | None]
            (is_valid, error_message)
            Returns error for first invalid sample
        """
        if not features_list:
            return False, "Empty features list"

        for i, features in enumerate(features_list):
            is_valid, error = self.validate(features)
            if not is_valid:
                return False, f"Sample {i}: {error}"

        return True, None

    def to_array(self, features: dict[str, Any]) -> np.ndarray:
        """
        Convert validated feature dictionary to numpy array.

        Parameters
        ----------
        features : dict[str, any]
            Feature dictionary (must be validated first)

        Returns
        -------
        np.ndarray
            Features as array in expected order, shape (n_features,)

        Examples
        --------
        >>> validator = FeatureValidator(['a', 'b'])
        >>> features = {'a': 1.0, 'b': 2.0}
        >>> arr = validator.to_array(features)
        >>> assert arr.shape == (2,)
        """
        return np.array([features[f] for f in self.expected_features], dtype=np.float64)

    def batch_to_array(self, features_list: list[dict[str, Any]]) -> np.ndarray:
        """
        Convert batch of feature dictionaries to numpy array.

        Parameters
        ----------
        features_list : list[dict[str, any]]
            List of feature dictionaries

        Returns
        -------
        np.ndarray
            Features as array, shape (n_samples, n_features)
        """
        return np.array(
            [[f[feat] for feat in self.expected_features] for f in features_list],
            dtype=np.float64,
        )
