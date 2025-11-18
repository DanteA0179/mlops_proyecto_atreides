"""
Feature Engineering Service for Energy Optimization API.

This module provides feature transformation for API requests,
integrating temporal features (US-011) and preprocessing pipeline (US-012).
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.api.models.requests import PredictionRequest
from src.features.preprocessing import PreprocessingPipeline
from src.utils.temporal_features import create_all_temporal_features

logger = logging.getLogger(__name__)


class FeatureService:
    """
    Feature transformation service integrating US-011 and US-012.

    Transforms raw API request → model-ready features (18 total):
    - 9 original features
    - 7 temporal features (US-011: hour, cyclical sin/cos, etc.)
    - 2 one-hot encoded (US-012: Load_Type preprocessing)

    References:
    - US-011: Temporal feature engineering
    - US-012: Scaling + encoding pipeline

    Attributes
    ----------
    preprocessing : PreprocessingPipeline
        Fitted preprocessing pipeline
    preprocessing_path : Path
        Path to preprocessing pipeline file

    Examples
    --------
    >>> feature_service = FeatureService()
    >>> features = feature_service.transform_request(request)
    >>> features.shape
    (1, 18)
    """

    def __init__(self, preprocessing_pipeline_path: str = None):
        """
        Initialize feature service.

        Parameters
        ----------
        preprocessing_pipeline_path : str, optional
            Path to preprocessing pipeline pickle file

        Raises
        ------
        FileNotFoundError
            If preprocessing pipeline not found
        """
        if not preprocessing_pipeline_path:
            preprocessing_pipeline_path = "models/preprocessing/preprocessing_pipeline.pkl"

        pipeline_path = Path(preprocessing_pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found: {pipeline_path}. "
                "Run US-012 pipeline first: python src/features/build_preprocessed_dataset.py"
            )

        # Load fitted pipeline (StandardScaler + OneHotEncoder)
        self.preprocessing = PreprocessingPipeline.load(pipeline_path)
        self.preprocessing_path = pipeline_path
        logger.info(f"FeatureService initialized with pipeline: {pipeline_path}")

    def transform_request(self, request: PredictionRequest) -> np.ndarray:
        """
        Transform API request to model-ready features.

        Pipeline:
        1. Convert request → Polars DataFrame (9 original features)
        2. Create 7 temporal features (US-011)
        3. Apply preprocessing: scaling + one-hot encoding (US-012)
        4. Return numpy array (18 features final)

        Parameters
        ----------
        request : PredictionRequest
            Validated API request with 8 input fields

        Returns
        -------
        np.ndarray
            Shape (1, 18) ready for model inference

        Examples
        --------
        >>> feature_service = FeatureService()
        >>> request = PredictionRequest(...)
        >>> features = feature_service.transform_request(request)
        >>> model.predict(features)
        """
        # Step 1: Convert request to DataFrame
        df = pl.DataFrame(
            [
                {
                    "Lagging_Current_Reactive.Power_kVarh": request.lagging_reactive_power,
                    "Leading_Current_Reactive_Power_kVarh": request.leading_reactive_power,
                    "CO2(tCO2)": request.co2,
                    "Lagging_Current_Power_Factor": request.lagging_power_factor,
                    "Leading_Current_Power_Factor": request.leading_power_factor,
                    "NSM": request.nsm,
                    "Day_of_week": self._map_day_to_string(request.day_of_week),
                    "Load_Type": request.load_type,
                    "WeekStatus": "Weekday" if request.day_of_week < 5 else "Weekend",
                }
            ]
        )

        logger.debug(f"Created DataFrame with shape: {df.shape}")

        # Step 2: Create temporal features (US-011)
        # Adds: hour, day_of_week (int), is_weekend, 4 cyclical features
        df = create_all_temporal_features(df, nsm_col="NSM", day_name_col="Day_of_week")

        logger.debug(f"After temporal features: {df.shape}")

        # Step 3: Apply preprocessing pipeline (US-012)
        # - StandardScaler for numeric features (excluding cyclical)
        # - OneHotEncoder for Load_Type (drop='first')
        # - Binary mapping for WeekStatus
        df_processed = self.preprocessing.transform(df)

        logger.debug(f"After preprocessing: {df_processed.shape}")

        # Step 4: Convert to numpy array
        features = df_processed.to_numpy()

        logger.debug(f"Final features shape: {features.shape}")

        return features

    def transform_batch(self, requests: list[PredictionRequest]) -> np.ndarray:
        """
        Transform batch of API requests to model-ready features.

        Parameters
        ----------
        requests : List[PredictionRequest]
            List of validated API requests

        Returns
        -------
        np.ndarray
            Shape (n_samples, 18) ready for model inference

        Examples
        --------
        >>> feature_service = FeatureService()
        >>> requests = [PredictionRequest(...), PredictionRequest(...)]
        >>> features = feature_service.transform_batch(requests)
        >>> model.predict(features)
        """
        # Convert all requests to list of dicts
        data = []
        for req in requests:
            data.append(
                {
                    "Lagging_Current_Reactive.Power_kVarh": req.lagging_reactive_power,
                    "Leading_Current_Reactive_Power_kVarh": req.leading_reactive_power,
                    "CO2(tCO2)": req.co2,
                    "Lagging_Current_Power_Factor": req.lagging_power_factor,
                    "Leading_Current_Power_Factor": req.leading_power_factor,
                    "NSM": req.nsm,
                    "Day_of_week": self._map_day_to_string(req.day_of_week),
                    "Load_Type": req.load_type,
                    "WeekStatus": "Weekday" if req.day_of_week < 5 else "Weekend",
                }
            )

        # Create DataFrame
        df = pl.DataFrame(data)

        logger.debug(f"Created batch DataFrame with shape: {df.shape}")

        # Create temporal features
        df = create_all_temporal_features(df, nsm_col="NSM", day_name_col="Day_of_week")

        # Apply preprocessing
        df_processed = self.preprocessing.transform(df)

        # Convert to numpy
        features = df_processed.to_numpy()

        logger.info(f"Transformed batch: {len(requests)} requests → {features.shape}")

        return features

    @staticmethod
    def _map_day_to_string(day_of_week: int) -> str:
        """
        Map day number (0-6) to string name.

        Parameters
        ----------
        day_of_week : int
            Day number (0=Monday, 6=Sunday)

        Returns
        -------
        str
            Day name

        Examples
        --------
        >>> FeatureService._map_day_to_string(0)
        'Monday'
        >>> FeatureService._map_day_to_string(6)
        'Sunday'
        """
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return days[day_of_week]

    def get_feature_names(self) -> list[str]:
        """
        Get names of all 18 features after transformation.

        Returns
        -------
        List[str]
            List of feature names

        Examples
        --------
        >>> feature_service.get_feature_names()
        ['feature1', 'feature2', ..., 'feature18']
        """
        return self.preprocessing.get_feature_names_out()

    def get_feature_count(self) -> int:
        """
        Get total number of features (should be 18).

        Returns
        -------
        int
            Number of features

        Examples
        --------
        >>> feature_service.get_feature_count()
        18
        """
        return len(self.get_feature_names())
