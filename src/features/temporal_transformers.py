"""
Temporal Feature Engineering Transformers.

This module provides sklearn-compatible transformers for temporal feature engineering,
designed for pipeline integration and maximum reusability.

Classes
-------
HourExtractor : Extract hour from NSM (seconds from midnight)
DayOfWeekEncoder : Convert day names to numeric values
WeekendIndicator : Create binary weekend indicator
CyclicalEncoder : Generic cyclical encoding using sin/cos (highly reusable)
TemporalFeatureEngineer : Complete pipeline combining all transformers
"""

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class HourExtractor(BaseEstimator, TransformerMixin):
    """
    Extract hour of day (0-23) from NSM column.

    Parameters
    ----------
    nsm_col : str, default='NSM'
        Name of column containing seconds from midnight
    output_col : str, default='hour'
        Name for output hour column
    drop_nsm : bool, default=False
        Whether to drop NSM column after extraction

    Examples
    --------
    >>> import polars as pl
    >>> from src.features.temporal_transformers import HourExtractor
    >>>
    >>> df = pl.DataFrame({'NSM': [0, 3600, 7200]})
    >>> transformer = HourExtractor()
    >>> df_transformed = transformer.fit_transform(df)
    >>> df_transformed['hour'].to_list()
    [0, 1, 2]
    """

    def __init__(self, nsm_col: str = "NSM", output_col: str = "hour", drop_nsm: bool = False):
        self.nsm_col = nsm_col
        self.output_col = output_col
        self.drop_nsm = drop_nsm

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        """
        Fit transformer (no-op for this transformer).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame
        y : pl.Series, optional
            Target variable (ignored)

        Returns
        -------
        self : HourExtractor
            Fitted transformer
        """
        # Validate NSM column exists
        if self.nsm_col not in X.columns:
            raise ValueError(f"Column '{self.nsm_col}' not found in DataFrame")
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Extract hour from NSM column.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with NSM column

        Returns
        -------
        pl.DataFrame
            DataFrame with hour column added

        Notes
        -----
        Special handling for NSM=86400 (end of day) which maps to hour 23.
        """
        X = X.clone()

        # Extract hour with special case for NSM=86400
        X = X.with_columns(
            pl.when(pl.col(self.nsm_col) >= 86400)
            .then(23)
            .otherwise(pl.col(self.nsm_col) // 3600)
            .cast(pl.Int32)
            .alias(self.output_col)
        )

        if self.drop_nsm:
            X = X.drop(self.nsm_col)

        return X


class DayOfWeekEncoder(BaseEstimator, TransformerMixin):
    """
    Convert day name to numeric value (0-6).

    Parameters
    ----------
    day_col : str, default='Day_of_week'
        Name of column containing day names
    output_col : str, default='day_of_week'
        Name for output numeric column
    drop_original : bool, default=False
        Whether to drop original day column

    Examples
    --------
    >>> df = pl.DataFrame({'Day_of_week': ['Monday', 'Tuesday', 'Wednesday']})
    >>> transformer = DayOfWeekEncoder()
    >>> df_transformed = transformer.fit_transform(df)
    >>> df_transformed['day_of_week'].to_list()
    [0, 1, 2]
    """

    DAY_MAPPING = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    def __init__(
        self,
        day_col: str = "Day_of_week",
        output_col: str = "day_of_week",
        drop_original: bool = False,
    ):
        self.day_col = day_col
        self.output_col = output_col
        self.drop_original = drop_original

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        """
        Fit transformer (validates column exists).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame
        y : pl.Series, optional
            Target variable (ignored)

        Returns
        -------
        self : DayOfWeekEncoder
            Fitted transformer
        """
        if self.day_col not in X.columns:
            raise ValueError(f"Column '{self.day_col}' not found in DataFrame")
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Convert day names to numeric values.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with day name column

        Returns
        -------
        pl.DataFrame
            DataFrame with numeric day column
        """
        X = X.clone()

        # Build mapping expression
        mapping_expr = pl.col(self.day_col)
        for day_name, day_num in self.DAY_MAPPING.items():
            if day_name == "Monday":
                mapping_expr = pl.when(pl.col(self.day_col) == day_name).then(day_num)
            else:
                mapping_expr = mapping_expr.when(pl.col(self.day_col) == day_name).then(day_num)

        X = X.with_columns(mapping_expr.cast(pl.Int32).alias(self.output_col))

        if self.drop_original:
            X = X.drop(self.day_col)

        return X


class WeekendIndicator(BaseEstimator, TransformerMixin):
    """
    Create binary weekend indicator from numeric day column.

    Parameters
    ----------
    day_col : str, default='day_of_week'
        Name of column containing numeric day (0-6)
    output_col : str, default='is_weekend'
        Name for output boolean column
    weekend_days : list of int, default=[5, 6]
        Numeric values representing weekend days (5=Saturday, 6=Sunday)

    Examples
    --------
    >>> df = pl.DataFrame({'day_of_week': [0, 1, 5, 6]})  # Mon, Tue, Sat, Sun
    >>> transformer = WeekendIndicator()
    >>> df_transformed = transformer.fit_transform(df)
    >>> df_transformed['is_weekend'].to_list()
    [False, False, True, True]
    """

    def __init__(
        self,
        day_col: str = "day_of_week",
        output_col: str = "is_weekend",
        weekend_days: list[int] = None,
    ):
        self.day_col = day_col
        self.output_col = output_col
        self.weekend_days = weekend_days if weekend_days is not None else [5, 6]

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        """
        Fit transformer (validates column exists).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame
        y : pl.Series, optional
            Target variable (ignored)

        Returns
        -------
        self : WeekendIndicator
            Fitted transformer
        """
        if self.day_col not in X.columns:
            raise ValueError(f"Column '{self.day_col}' not found in DataFrame")
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Create weekend indicator column.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric day column

        Returns
        -------
        pl.DataFrame
            DataFrame with boolean weekend column
        """
        X = X.clone()

        # Create weekend indicator
        X = X.with_columns(pl.col(self.day_col).is_in(self.weekend_days).alias(self.output_col))

        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Generic cyclical encoding transformer using sin/cos.

    This transformer can encode ANY cyclical feature (hour, day, month, angle, etc.)
    by mapping it to the unit circle using trigonometric functions.

    Parameters
    ----------
    column : str
        Name of column to encode
    period : int
        Period of the cyclical feature (e.g., 24 for hours, 7 for days, 12 for months)
    sin_col : str, optional
        Name for sine output column (default: 'cyclical_{column}_sin')
    cos_col : str, optional
        Name for cosine output column (default: 'cyclical_{column}_cos')
    drop_original : bool, default=False
        Whether to drop original column after encoding

    Examples
    --------
    >>> # Encode hour of day
    >>> df = pl.DataFrame({'hour': [0, 6, 12, 18, 23]})
    >>> encoder = CyclicalEncoder(column='hour', period=24)
    >>> df_encoded = encoder.fit_transform(df)
    >>> df_encoded.columns
    ['hour', 'cyclical_hour_sin', 'cyclical_hour_cos']

    >>> # Encode month (generic usage!)
    >>> df = pl.DataFrame({'month': [1, 3, 6, 9, 12]})
    >>> encoder = CyclicalEncoder(column='month', period=12)
    >>> df_encoded = encoder.fit_transform(df)
    >>> df_encoded.columns
    ['month', 'cyclical_month_sin', 'cyclical_month_cos']

    >>> # Encode compass direction
    >>> df = pl.DataFrame({'wind_direction': [0, 90, 180, 270, 359]})
    >>> encoder = CyclicalEncoder(column='wind_direction', period=360)
    >>> df_encoded = encoder.fit_transform(df)
    >>> df_encoded.columns
    ['wind_direction', 'cyclical_wind_direction_sin', 'cyclical_wind_direction_cos']
    """

    def __init__(
        self,
        column: str,
        period: int,
        sin_col: str | None = None,
        cos_col: str | None = None,
        drop_original: bool = False,
    ):
        self.column = column
        self.period = period
        self.sin_col = sin_col
        self.cos_col = cos_col
        self.drop_original = drop_original

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        """
        Fit transformer (validates column exists and period is positive).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame
        y : pl.Series, optional
            Target variable (ignored)

        Returns
        -------
        self : CyclicalEncoder
            Fitted transformer
        """
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        if self.period <= 0:
            raise ValueError(f"Period must be positive, got {self.period}")

        # Set default column names if not provided
        if self.sin_col is None:
            self.sin_col = f"cyclical_{self.column}_sin"
        if self.cos_col is None:
            self.cos_col = f"cyclical_{self.column}_cos"

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Apply cyclical encoding to column.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with column to encode

        Returns
        -------
        pl.DataFrame
            DataFrame with sin and cos columns added

        Notes
        -----
        The encoding preserves cyclical proximity:
        - sin(2π × value / period)
        - cos(2π × value / period)

        This ensures that values at the boundary (e.g., hour 23 and hour 0)
        have similar encoded values, correctly representing their proximity.
        """
        X = X.clone()

        # Apply cyclical encoding
        X = X.with_columns(
            [
                (pl.col(self.column) * 2 * np.pi / self.period).sin().alias(self.sin_col),
                (pl.col(self.column) * 2 * np.pi / self.period).cos().alias(self.cos_col),
            ]
        )

        if self.drop_original:
            X = X.drop(self.column)

        return X


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Complete temporal feature engineering pipeline.

    This transformer combines all temporal transformers into a single pipeline
    for convenient usage. It creates 7 temporal features:

    1. hour (0-23) from NSM
    2. day_of_week (0-6) from day name
    3. is_weekend (boolean)
    4-5. cyclical_hour_sin, cyclical_hour_cos
    6-7. cyclical_day_sin, cyclical_day_cos

    Parameters
    ----------
    nsm_col : str, default='NSM'
        Column containing seconds from midnight
    day_name_col : str, default='Day_of_week'
        Column containing day names
    create_hour : bool, default=True
        Whether to create hour feature
    create_day : bool, default=True
        Whether to create numeric day feature
    create_weekend : bool, default=True
        Whether to create weekend indicator
    create_cyclical : bool, default=True
        Whether to create cyclical encodings

    Examples
    --------
    >>> import polars as pl
    >>> from src.features.temporal_transformers import TemporalFeatureEngineer
    >>>
    >>> df = pl.DataFrame({
    ...     'NSM': [0, 43200, 86399],
    ...     'Day_of_week': ['Monday', 'Wednesday', 'Sunday'],
    ...     'Usage_kWh': [10, 20, 15]
    ... })
    >>>
    >>> engineer = TemporalFeatureEngineer()
    >>> df_featured = engineer.fit_transform(df)
    >>> df_featured.shape
    (3, 10)  # Original 3 columns + 7 new features

    >>> # Use in sklearn Pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> pipeline = Pipeline([
    ...     ('temporal', TemporalFeatureEngineer()),
    ...     ('scaler', StandardScaler()),
    ...     ('model', LinearRegression())
    ... ])
    >>> pipeline.fit(X_train, y_train)
    """

    def __init__(
        self,
        nsm_col: str = "NSM",
        day_name_col: str = "Day_of_week",
        create_hour: bool = True,
        create_day: bool = True,
        create_weekend: bool = True,
        create_cyclical: bool = True,
    ):
        self.nsm_col = nsm_col
        self.day_name_col = day_name_col
        self.create_hour = create_hour
        self.create_day = create_day
        self.create_weekend = create_weekend
        self.create_cyclical = create_cyclical

        # Initialize sub-transformers
        self._hour_extractor = None
        self._day_encoder = None
        self._weekend_indicator = None
        self._hour_cyclical = None
        self._day_cyclical = None

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        """
        Fit all sub-transformers.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame
        y : pl.Series, optional
            Target variable (ignored)

        Returns
        -------
        self : TemporalFeatureEngineer
            Fitted transformer
        """
        # Initialize and fit sub-transformers
        if self.create_hour:
            self._hour_extractor = HourExtractor(nsm_col=self.nsm_col)
            self._hour_extractor.fit(X)

        if self.create_day:
            self._day_encoder = DayOfWeekEncoder(day_col=self.day_name_col)
            self._day_encoder.fit(X)

        if self.create_weekend and self.create_day:
            self._weekend_indicator = WeekendIndicator()

        if self.create_cyclical:
            if self.create_hour:
                self._hour_cyclical = CyclicalEncoder(
                    column="hour",
                    period=24,
                    sin_col="cyclical_hour_sin",
                    cos_col="cyclical_hour_cos",
                )
            if self.create_day:
                self._day_cyclical = CyclicalEncoder(
                    column="day_of_week",
                    period=7,
                    sin_col="cyclical_day_sin",
                    cos_col="cyclical_day_cos",
                )

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Apply all temporal transformations.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame

        Returns
        -------
        pl.DataFrame
            DataFrame with all temporal features added
        """
        X = X.clone()

        # Apply transformations in order
        if self.create_hour and self._hour_extractor:
            X = self._hour_extractor.transform(X)

        if self.create_day and self._day_encoder:
            X = self._day_encoder.transform(X)

        if self.create_weekend and self._weekend_indicator:
            X = self._weekend_indicator.transform(X)

        if self.create_cyclical:
            if self.create_hour and self._hour_cyclical:
                # Fit cyclical encoder on transformed data
                self._hour_cyclical.fit(X)
                X = self._hour_cyclical.transform(X)

            if self.create_day and self._day_cyclical:
                # Fit cyclical encoder on transformed data
                self._day_cyclical.fit(X)
                X = self._day_cyclical.transform(X)

        return X

    def get_feature_names_out(self) -> list[str]:
        """
        Get names of output features.

        Returns
        -------
        list of str
            Names of created features
        """
        features = []

        if self.create_hour:
            features.append("hour")

        if self.create_day:
            features.append("day_of_week")

        if self.create_weekend:
            features.append("is_weekend")

        if self.create_cyclical:
            if self.create_hour:
                features.extend(["cyclical_hour_sin", "cyclical_hour_cos"])
            if self.create_day:
                features.extend(["cyclical_day_sin", "cyclical_day_cos"])

        return features
