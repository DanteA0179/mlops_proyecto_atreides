"""Feature engineering package."""

# Import utility functions for convenient access
from src.utils.temporal_features import (
    create_all_temporal_features,
    create_cyclical_encoding,
    create_is_weekend,
    extract_day_of_week_numeric,
    extract_hour_from_nsm,
    validate_temporal_features,
)

from .temporal_transformers import (
    CyclicalEncoder,
    DayOfWeekEncoder,
    HourExtractor,
    TemporalFeatureEngineer,
    WeekendIndicator,
)

__all__ = [
    # POO Transformers (sklearn-compatible)
    "HourExtractor",
    "DayOfWeekEncoder",
    "WeekendIndicator",
    "CyclicalEncoder",
    "TemporalFeatureEngineer",
    # Utility functions
    "extract_hour_from_nsm",
    "extract_day_of_week_numeric",
    "create_is_weekend",
    "create_cyclical_encoding",
    "create_all_temporal_features",
    "validate_temporal_features",
]
