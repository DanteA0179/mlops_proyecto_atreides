"""
Preprocessing Transformers for ML Pipeline.

Sklearn-compatible transformers for feature selection, scaling, and encoding.
These transformers work with Polars DataFrames and can be used in sklearn pipelines.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select subset of features from dataframe.

    Parameters
    ----------
    features : list of str
        Features to select
    drop_target : bool, default=True
        Whether to drop target column if present
    target_col : str, default='Usage_kWh'
        Name of target column

    Attributes
    ----------
    features_ : list of str
        Features to select (set after fit)
    selected_features_ : list of str
        Features actually selected after validation

    Examples
    --------
    >>> selector = FeatureSelector(features=['NSM', 'CO2(tCO2)', 'Load_Type'])
    >>> X_selected = selector.fit_transform(X_train)
    """

    def __init__(
        self, features: list[str], drop_target: bool = True, target_col: str = "Usage_kWh"
    ):
        self.features = features
        self.drop_target = drop_target
        self.target_col = target_col

    def fit(self, X, y=None):
        """
        Fit the selector (validates features exist).

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data
        y : ignored

        Returns
        -------
        self
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        # Validate features exist
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Features not found in dataframe: {missing}")

        # Store validated features
        self.features_ = self.features.copy()

        # Filter out target if requested
        if self.drop_target and self.target_col in self.features_:
            self.selected_features_ = [
                f for f in self.features_ if f != self.target_col
            ]
        else:
            self.selected_features_ = self.features_

        return self

    def transform(self, X):
        """
        Select features.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Selected features
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        return X.select(self.selected_features_)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        np.ndarray
            Selected feature names
        """
        return np.array(self.selected_features_)


class NumericScaler(BaseEstimator, TransformerMixin):
    """
    Scale numeric features using StandardScaler.

    Parameters
    ----------
    features : list of str
        Numeric features to scale
    exclude_features : list of str, optional
        Features to exclude from scaling (e.g., cyclical features)

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted scaler
    features_to_scale_ : list of str
        Features that will be scaled
    scaling_statistics_ : dict
        Mean and std for each feature

    Examples
    --------
    >>> scaler = NumericScaler(
    ...     features=['NSM', 'CO2(tCO2)'],
    ...     exclude_features=['hour_sin', 'hour_cos']
    ... )
    >>> X_scaled = scaler.fit_transform(X_train)
    """

    def __init__(self, features: list[str], exclude_features: list[str] = None):
        self.features = features
        self.exclude_features = exclude_features if exclude_features else []

    def fit(self, X, y=None):
        """
        Fit the scaler.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data
        y : ignored

        Returns
        -------
        self
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        # Determine features to scale
        self.features_to_scale_ = [
            f for f in self.features if f not in self.exclude_features
        ]

        # Validate features exist
        missing = set(self.features_to_scale_) - set(X.columns)
        if missing:
            raise ValueError(f"Features not found in dataframe: {missing}")

        # Fit StandardScaler
        self.scaler_ = StandardScaler()
        X_to_scale = X.select(self.features_to_scale_).to_pandas()
        self.scaler_.fit(X_to_scale)

        # Store statistics
        self.scaling_statistics_ = {
            feat: {"mean": float(mean), "std": float(std)}
            for feat, mean, std in zip(
                self.features_to_scale_, self.scaler_.mean_, self.scaler_.scale_, strict=True
            )
        }

        return self

    def transform(self, X):
        """
        Scale features.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Data with scaled features
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        X = X.clone()

        # Scale features
        if self.features_to_scale_:
            X_to_scale = X.select(self.features_to_scale_).to_pandas()
            X_scaled = self.scaler_.transform(X_to_scale)

            # Replace scaled features
            for i, feat in enumerate(self.features_to_scale_):
                X = X.with_columns(pl.Series(feat, X_scaled[:, i]))

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names (same as input for scaling).

        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names

        Returns
        -------
        np.ndarray
            Feature names
        """
        if input_features is not None:
            return np.array(input_features)
        return np.array(self.features)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features using OneHotEncoder.

    Parameters
    ----------
    categorical_features : list of str
        Categorical features to encode
    binary_features : dict, optional
        Binary features to map {feature: {category: value}}
    drop : str, default='first'
        Strategy for OneHotEncoder drop parameter
    handle_unknown : str, default='ignore'
        Strategy for unknown categories

    Attributes
    ----------
    encoder_ : OneHotEncoder
        Fitted encoder
    feature_names_out_ : list of str
        Output feature names after encoding
    categories_ : dict
        Categories per feature

    Examples
    --------
    >>> encoder = CategoricalEncoder(
    ...     categorical_features=['Load_Type'],
    ...     binary_features={'WeekStatus': {'Weekday': 0, 'Weekend': 1}},
    ...     drop='first'
    ... )
    >>> X_encoded = encoder.fit_transform(X_train)
    """

    def __init__(
        self,
        categorical_features: list[str],
        binary_features: dict[str, dict[str, int]] = None,
        drop: str = "first",
        handle_unknown: str = "ignore",
    ):
        self.categorical_features = categorical_features
        self.binary_features = binary_features if binary_features else {}
        self.drop = drop
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """
        Fit the encoder.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data
        y : ignored

        Returns
        -------
        self
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        # Validate features exist
        all_features = self.categorical_features + list(self.binary_features.keys())
        missing = set(all_features) - set(X.columns)
        if missing:
            raise ValueError(f"Features not found in dataframe: {missing}")

        # Fit OneHotEncoder for categorical features
        if self.categorical_features:
            self.encoder_ = OneHotEncoder(
                drop=self.drop, sparse_output=False, handle_unknown=self.handle_unknown
            )
            X_cat = X.select(self.categorical_features).to_pandas()
            self.encoder_.fit(X_cat)

            # Store categories
            self.categories_ = {
                feat: list(cats)
                for feat, cats in zip(
                    self.categorical_features, self.encoder_.categories_, strict=True
                )
            }

            # Get feature names from encoder
            self.feature_names_out_ = list(
                self.encoder_.get_feature_names_out(self.categorical_features)
            )
        else:
            self.encoder_ = None
            self.categories_ = {}
            self.feature_names_out_ = []

        return self

    def transform(self, X):
        """
        Encode categorical features.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Data with encoded features
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        X = X.clone()

        # Apply binary mappings
        for feat, mapping in self.binary_features.items():
            if feat in X.columns:
                # Create mapping expression
                mapping_expr = pl.col(feat)
                for i, (category, value) in enumerate(mapping.items()):
                    if i == 0:
                        mapping_expr = pl.when(pl.col(feat) == category).then(value)
                    else:
                        mapping_expr = mapping_expr.when(pl.col(feat) == category).then(
                            value
                        )
                mapping_expr = mapping_expr.otherwise(None)

                X = X.with_columns(mapping_expr.cast(pl.Int32).alias(feat))

        # Apply OneHotEncoding
        if self.encoder_ is not None and self.categorical_features:
            X_cat = X.select(self.categorical_features).to_pandas()
            X_encoded = self.encoder_.transform(X_cat)

            # Drop original categorical features
            X = X.drop(self.categorical_features)

            # Add encoded features
            for i, feat_name in enumerate(self.feature_names_out_):
                X = X.with_columns(pl.Series(feat_name, X_encoded[:, i].astype(int)))

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        np.ndarray
            Output feature names
        """
        return np.array(self.feature_names_out_ + list(self.binary_features.keys()))


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline combining selection, scaling, and encoding.

    Parameters
    ----------
    numeric_features : list of str
        Numeric features to scale
    categorical_features : list of str
        Categorical features to encode
    exclude_from_scaling : list of str, optional
        Features to exclude from scaling (e.g., cyclical)
    binary_features : dict, optional
        Binary features to map
    target_col : str, default='Usage_kWh'
        Target column name
    drop_ohe : str, default='first'
        OneHotEncoder drop strategy

    Attributes
    ----------
    selector_ : FeatureSelector
        Feature selector
    scaler_ : NumericScaler
        Numeric scaler
    encoder_ : CategoricalEncoder
        Categorical encoder
    feature_names_in_ : list of str
        Input feature names
    feature_names_out_ : list of str
        Output feature names

    Examples
    --------
    >>> pipeline = PreprocessingPipeline(
    ...     numeric_features=['NSM', 'CO2(tCO2)'],
    ...     categorical_features=['Load_Type'],
    ...     exclude_from_scaling=['hour_sin', 'hour_cos'],
    ...     binary_features={'WeekStatus': {'Weekday': 0, 'Weekend': 1}}
    ... )
    >>> X_preprocessed = pipeline.fit_transform(X_train)
    >>> pipeline.save('models/preprocessing_pipeline.pkl')
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        exclude_from_scaling: list[str] = None,
        binary_features: dict[str, dict[str, int]] = None,
        target_col: str = "Usage_kWh",
        drop_ohe: str = "first",
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.exclude_from_scaling = exclude_from_scaling if exclude_from_scaling else []
        self.binary_features = binary_features if binary_features else {}
        self.target_col = target_col
        self.drop_ohe = drop_ohe

    def fit(self, X, y=None):
        """
        Fit the complete pipeline.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data
        y : ignored

        Returns
        -------
        self
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        # Store input feature names
        self.feature_names_in_ = X.columns

        # 1. Feature Selection
        all_features = (
            self.numeric_features
            + self.categorical_features
            + list(self.binary_features.keys())
        )
        self.selector_ = FeatureSelector(
            features=all_features, drop_target=True, target_col=self.target_col
        )
        X = self.selector_.fit_transform(X)

        # 2. Numeric Scaling
        self.scaler_ = NumericScaler(
            features=self.numeric_features, exclude_features=self.exclude_from_scaling
        )
        X = self.scaler_.fit_transform(X)

        # 3. Categorical Encoding
        self.encoder_ = CategoricalEncoder(
            categorical_features=self.categorical_features,
            binary_features=self.binary_features,
            drop=self.drop_ohe,
        )
        X = self.encoder_.fit_transform(X)

        # Store output feature names
        self.feature_names_out_ = X.columns

        return self

    def transform(self, X):
        """
        Apply complete preprocessing pipeline.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Preprocessed data
        """
        # Convert to polars if pandas
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        # Apply transformations in order
        X = self.selector_.transform(X)
        X = self.scaler_.transform(X)
        X = self.encoder_.transform(X)

        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pl.DataFrame or pd.DataFrame
            Input data
        y : ignored

        Returns
        -------
        pl.DataFrame
            Preprocessed data
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        np.ndarray
            Output feature names
        """
        return np.array(self.feature_names_out_)

    def save(self, filepath: str | Path):
        """
        Save pipeline to disk using joblib.

        Parameters
        ----------
        filepath : str or Path
            Path to save pipeline

        Examples
        --------
        >>> pipeline.save('models/preprocessing_pipeline.pkl')
        """
        import joblib

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str | Path):
        """
        Load pipeline from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to load pipeline from

        Returns
        -------
        PreprocessingPipeline
            Loaded pipeline

        Examples
        --------
        >>> pipeline = PreprocessingPipeline.load('models/preprocessing_pipeline.pkl')
        """
        import joblib

        return joblib.load(filepath)

    def get_preprocessing_summary(self) -> dict[str, Any]:
        """
        Get summary of preprocessing configuration.

        Returns
        -------
        dict
            Summary with keys:
            - n_features_in: Number of input features
            - n_features_out: Number of output features
            - numeric_features: List of numeric features
            - scaled_features: List of features that were scaled
            - categorical_features: List of categorical features
            - binary_features: Binary feature mappings
            - ohe_features: OneHotEncoded feature names
            - scaling_statistics: Mean and std per feature

        Examples
        --------
        >>> summary = pipeline.get_preprocessing_summary()
        >>> print(json.dumps(summary, indent=2))
        """
        summary = {
            "n_features_in": len(self.feature_names_in_),
            "n_features_out": len(self.feature_names_out_),
            "numeric_features": self.numeric_features,
            "scaled_features": self.scaler_.features_to_scale_,
            "excluded_from_scaling": self.exclude_from_scaling,
            "categorical_features": self.categorical_features,
            "binary_features": self.binary_features,
            "ohe_features": (
                self.encoder_.feature_names_out_ if self.encoder_.encoder_ else []
            ),
            "ohe_categories": self.encoder_.categories_,
            "scaling_statistics": self.scaler_.scaling_statistics_,
        }
        return summary
