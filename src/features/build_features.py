"""
Feature Building Pipeline

This module provides the main feature engineering pipeline that combines
all feature transformers.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .temporal_features import (
    TemporalFeatureEngineer,
    CyclicalEncoder,
    DayOfWeekEncoder
)


def create_feature_engineering_pipeline(config=None):
    """
    Creates a complete feature engineering pipeline.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with feature engineering parameters
        
    Returns
    -------
    Pipeline
        Sklearn pipeline with all feature engineering steps
        
    Examples
    --------
    >>> from src.features.build_features import create_feature_engineering_pipeline
    >>> pipeline = create_feature_engineering_pipeline()
    >>> X_transformed = pipeline.fit_transform(X)
    """
    if config is None:
        config = get_default_config()
    
    # Step 1: Temporal features from NSM
    temporal_step = TemporalFeatureEngineer(
        nsm_column='NSM',
        drop_original=False
    )
    
    # Step 2: Day of week encoding
    day_step = DayOfWeekEncoder(
        day_column='Day_of_week',
        drop_original=False
    )
    
    # Step 3: Cyclical encoding for hour
    cyclical_step = CyclicalEncoder(
        columns=['hour'],
        periods={'hour': 24},
        drop_original=False
    )
    
    # Combine all steps
    feature_pipeline = Pipeline([
        ('temporal', temporal_step),
        ('day_encoding', day_step),
        ('cyclical', cyclical_step)
    ])
    
    return feature_pipeline


def get_default_config():
    """Returns default configuration for feature engineering."""
    return {
        'temporal': {
            'nsm_column': 'NSM',
            'drop_original': False
        },
        'cyclical': {
            'columns': ['hour'],
            'periods': {'hour': 24}
        },
        'day_encoding': {
            'weekend_days': ['Saturday', 'Sunday']
        }
    }
