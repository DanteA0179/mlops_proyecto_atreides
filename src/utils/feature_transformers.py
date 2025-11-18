"""
Feature transformers following Open/Closed Principle.

Abstract base classes and concrete implementations for feature engineering.
Easy to extend with new transformers without modifying existing code.
"""

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from src.config.constants import HOURS_PER_DAY


class FeatureTransformer(ABC):
    """
    Abstract base for feature transformers.

    Open for extension, closed for modification (OCP from SOLID).
    All feature transformers should inherit from this class.

    Examples
    --------
    >>> from src.utils.feature_transformers import HourFeatureTransformer
    >>> import polars as pl
    >>>
    >>> df = pl.DataFrame({"NSM": [0, 3600, 7200]})
    >>> transformer = HourFeatureTransformer()
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
    """

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform dataframe by adding new features.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe

        Returns
        -------
        pl.DataFrame
            Transformed dataframe with new features
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Get names of features created by this transformer.

        Returns
        -------
        list[str]
            List of feature names created
        """
        pass


class HourFeatureTransformer(FeatureTransformer):
    """
    Creates hour feature from NSM (seconds from midnight).

    Parameters
    ----------
    nsm_column : str, default="NSM"
        Name of column containing seconds from midnight

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_transformers import HourFeatureTransformer
    >>>
    >>> df = pl.DataFrame({"NSM": [0, 3600, 43200, 86399]})
    >>> transformer = HourFeatureTransformer()
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
    shape: (4, 2)
    ┌───────┬──────┐
    │ NSM   │ hour │
    │ ---   │ ---  │
    │ i64   │ i32  │
    ╞═══════╪══════╡
    │ 0     │ 0    │
    │ 3600  │ 1    │
    │ 43200 │ 12   │
    │ 86399 │ 23   │
    └───────┴──────┘
    """

    def __init__(self, nsm_column: str = "NSM"):
        """Initialize transformer with NSM column name."""
        self.nsm_column = nsm_column

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform dataframe by adding hour feature."""
        return df.with_columns((pl.col(self.nsm_column) // 3600).cast(pl.Int32).alias("hour"))

    def get_feature_names(self) -> list[str]:
        """Get feature names created by transformer."""
        return ["hour"]


class CyclicTimeTransformer(FeatureTransformer):
    """
    Creates cyclic encoding (sine/cosine) of time features.

    Cyclic encoding preserves temporal continuity:
    - Hour 23 is close to hour 0 (end/start of day)
    - Uses sine/cosine to map circular time to 2D coordinates

    Parameters
    ----------
    time_column : str, default="hour"
        Name of time column to encode
    period : int, default=24
        Period of cyclic feature (e.g., 24 for hours, 7 for days)

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_transformers import CyclicTimeTransformer
    >>>
    >>> df = pl.DataFrame({"hour": [0, 6, 12, 18, 23]})
    >>> transformer = CyclicTimeTransformer(time_column="hour", period=24)
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
    shape: (5, 3)
    ┌──────┬───────────┬───────────┐
    │ hour │ hour_sin  │ hour_cos  │
    │ ---  │ ---       │ ---       │
    │ i64  │ f64       │ f64       │
    ╞══════╪═══════════╪═══════════╡
    │ 0    │ 0.0       │ 1.0       │
    │ 6    │ 1.0       │ 0.0       │
    │ 12   │ 0.0       │ -1.0      │
    │ 18   │ -1.0      │ 0.0       │
    │ 23   │ 0.2588    │ 0.9659    │
    └──────┴───────────┴───────────┘

    Notes
    -----
    Cyclic encoding is recommended for periodic patterns as it:
    - Preserves continuity at period boundaries
    - Provides better distance metrics for ML models
    - Captures circular nature of time

    References
    ----------
    .. [1] "Encoding Cyclical Features for Deep Learning"
           https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
    """

    def __init__(self, time_column: str = "hour", period: int = HOURS_PER_DAY):
        """Initialize transformer with time column and period."""
        self.time_column = time_column
        self.period = period

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform dataframe by adding cyclic features."""
        return df.with_columns(
            [
                (2 * np.pi * pl.col(self.time_column) / self.period)
                .sin()
                .alias(f"{self.time_column}_sin"),
                (2 * np.pi * pl.col(self.time_column) / self.period)
                .cos()
                .alias(f"{self.time_column}_cos"),
            ]
        )

    def get_feature_names(self) -> list[str]:
        """Get feature names created by transformer."""
        return [f"{self.time_column}_sin", f"{self.time_column}_cos"]


class WeekdayTransformer(FeatureTransformer):
    """
    Creates weekday feature from Day_of_week column.

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_transformers import WeekdayTransformer
    >>>
    >>> df = pl.DataFrame({"Day_of_week": ["Monday", "Saturday", "Sunday"]})
    >>> transformer = WeekdayTransformer()
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
    shape: (3, 2)
    ┌─────────────┬───────────┐
    │ Day_of_week │ is_weekend│
    │ ---         │ ---       │
    │ str         │ i32       │
    ╞═════════════╪═══════════╡
    │ Monday      │ 0         │
    │ Saturday    │ 1         │
    │ Sunday      │ 1         │
    └─────────────┴───────────┘
    """

    def __init__(self, day_column: str = "Day_of_week"):
        """Initialize transformer with day column name."""
        self.day_column = day_column

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform dataframe by adding weekend indicator."""
        return df.with_columns(
            pl.when(pl.col(self.day_column).is_in(["Saturday", "Sunday"]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("is_weekend")
        )

    def get_feature_names(self) -> list[str]:
        """Get feature names created by transformer."""
        return ["is_weekend"]


class FeaturePipeline:
    """
    Composes multiple feature transformers into a pipeline.

    Easy to extend with new transformers without modifying existing code.
    Follows the Open/Closed Principle (OCP) from SOLID.

    Parameters
    ----------
    transformers : list[FeatureTransformer]
        List of transformers to apply sequentially

    Examples
    --------
    >>> import polars as pl
    >>> from src.utils.feature_transformers import (
    ...     FeaturePipeline,
    ...     HourFeatureTransformer,
    ...     CyclicTimeTransformer,
    ...     WeekdayTransformer
    ... )
    >>>
    >>> df = pl.DataFrame({
    ...     "NSM": [0, 43200],
    ...     "Day_of_week": ["Monday", "Saturday"]
    ... })
    >>>
    >>> pipeline = FeaturePipeline([
    ...     HourFeatureTransformer(),
    ...     CyclicTimeTransformer(),
    ...     WeekdayTransformer()
    ... ])
    >>>
    >>> df_transformed = pipeline.transform(df)
    >>> print(pipeline.get_feature_names())
    ['hour', 'hour_sin', 'hour_cos', 'is_weekend']
    """

    def __init__(self, transformers: list[FeatureTransformer]):
        """Initialize pipeline with list of transformers."""
        self.transformers = transformers

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply all transformers sequentially.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe

        Returns
        -------
        pl.DataFrame
            Transformed dataframe with all features added
        """
        for transformer in self.transformers:
            df = transformer.transform(df)
        return df

    def get_feature_names(self) -> list[str]:
        """
        Get all feature names created by pipeline.

        Returns
        -------
        list[str]
            Combined list of all feature names
        """
        names = []
        for transformer in self.transformers:
            names.extend(transformer.get_feature_names())
        return names
