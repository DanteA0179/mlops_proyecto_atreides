"""
Temporal Feature Engineering

This module provides transformers for creating temporal features from time-based columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates temporal features from NSM (Number of Seconds from Midnight).

    Features created:
    - hour (0-23)
    - minute (0-59)
    - is_night (boolean)
    - is_peak_hour (boolean)
    - time_of_day (morning/afternoon/evening/night)

    Parameters
    ----------
    nsm_column : str, default='NSM'
        Name of the column containing seconds from midnight
    drop_original : bool, default=False
        Whether to drop the original NSM column

    Examples
    --------
    >>> from src.features.temporal_features import TemporalFeatureEngineer
    >>> transformer = TemporalFeatureEngineer()
    >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(self, nsm_column="NSM", drop_original=False):
        self.nsm_column = nsm_column
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X):
        """Transform X by creating temporal features."""
        X = X.copy()

        # Convert NSM to hours and minutes
        X["hour"] = (X[self.nsm_column] // 3600).astype(int)
        X["minute"] = ((X[self.nsm_column] % 3600) // 60).astype(int)

        # Create derived features
        X["is_night"] = ((X["hour"] >= 22) | (X["hour"] <= 6)).astype(int)
        X["is_peak_hour"] = ((X["hour"] >= 9) & (X["hour"] <= 18)).astype(int)

        # Time of day categories
        X["time_of_day"] = pd.cut(
            X["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True,
        )

        if self.drop_original:
            X = X.drop(columns=[self.nsm_column])

        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes cyclical features using sine and cosine transformations.

    This is useful for temporal features like hour, day of week, month, etc.
    that have a cyclical nature (e.g., hour 23 is close to hour 0).

    Parameters
    ----------
    columns : list of str
        List of column names to encode
    periods : dict
        Dictionary mapping column names to their periods
        Example: {'hour': 24, 'day_of_week': 7, 'month': 12}
    drop_original : bool, default=True
        Whether to drop the original columns after encoding

    Examples
    --------
    >>> encoder = CyclicalEncoder(
    ...     columns=['hour', 'day_of_week'],
    ...     periods={'hour': 24, 'day_of_week': 7}
    ... )
    >>> X_transformed = encoder.fit_transform(X)
    """

    def __init__(self, columns, periods, drop_original=True):
        self.columns = columns
        self.periods = periods
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X):
        """Transform X by creating cyclical encodings."""
        X = X.copy()

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            if col not in self.periods:
                raise ValueError(f"Period not specified for column '{col}'")

            period = self.periods[col]

            # Create sine and cosine features
            X[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / period)
            X[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / period)

        if self.drop_original:
            X = X.drop(columns=self.columns)

        return X


class DayOfWeekEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes day of week with both cyclical and categorical features.

    Creates:
    - Cyclical encoding (sin/cos)
    - is_weekend boolean
    - day_type (weekday/weekend)

    Parameters
    ----------
    day_column : str, default='Day_of_week'
        Name of the column containing day of week
    weekend_days : list, default=['Saturday', 'Sunday']
        List of days considered as weekend
    drop_original : bool, default=False
        Whether to drop the original column
    """

    def __init__(self, day_column="Day_of_week", weekend_days=None, drop_original=False):
        self.day_column = day_column
        self.weekend_days = weekend_days or ["Saturday", "Sunday"]
        self.drop_original = drop_original
        self.day_mapping = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X):
        """Transform X by encoding day of week."""
        X = X.copy()

        # Convert day names to numbers
        X["day_numeric"] = X[self.day_column].map(self.day_mapping)

        # Cyclical encoding
        X["day_sin"] = np.sin(2 * np.pi * X["day_numeric"] / 7)
        X["day_cos"] = np.cos(2 * np.pi * X["day_numeric"] / 7)

        # Weekend indicator
        X["is_weekend"] = X[self.day_column].isin(self.weekend_days).astype(int)

        # Drop temporary column
        X = X.drop(columns=["day_numeric"])

        if self.drop_original:
            X = X.drop(columns=[self.day_column])

        return X
