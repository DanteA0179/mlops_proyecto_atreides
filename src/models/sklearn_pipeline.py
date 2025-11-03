"""
Scikit-Learn Pipeline for Energy Optimization

This module provides complete ML pipelines that combine:
1. Feature engineering
2. Preprocessing (scaling, encoding)
3. Model training

Following best practices for reproducibility and maintainability.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import create_feature_engineering_pipeline


def get_feature_groups():
    """
    Defines feature groups for preprocessing.
    
    Returns
    -------
    dict
        Dictionary with feature groups:
        - numeric: Continuous numerical features
        - categorical: Categorical features
        - temporal: Time-based features (will be engineered)
    """
    return {
        'numeric': [
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor'
        ],
        'categorical': [
            'Load_Type',
            'WeekStatus'
        ],
        'temporal': ['NSM', 'Day_of_week']
    }


def create_preprocessing_pipeline():
    """
    Creates preprocessing pipeline with feature engineering.
    
    This pipeline:
    1. Engineers temporal features
    2. Scales numerical features
    3. One-hot encodes categorical features
    
    Returns
    -------
    Pipeline
        Complete preprocessing pipeline
        
    Examples
    --------
    >>> from src.models.sklearn_pipeline import create_preprocessing_pipeline
    >>> preprocessor = create_preprocessing_pipeline()
    >>> X_transformed = preprocessor.fit_transform(X_train)
    """
    feature_groups = get_feature_groups()

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_groups['numeric']),
            ('cat', categorical_transformer, feature_groups['categorical'])
        ],
        remainder='passthrough'  # Keep other columns (temporal features)
    )

    return preprocessor



def create_full_pipeline(model, model_name='model'):
    """
    Creates complete ML pipeline: feature engineering + preprocessing + model.
    
    This is the MAIN pipeline that should be used for training and prediction.
    It ensures consistency between training and inference.
    
    Parameters
    ----------
    model : sklearn-compatible estimator
        Model to use (XGBoost, LightGBM, sklearn model, etc.)
    model_name : str, default='model'
        Name for the model step in the pipeline
        
    Returns
    -------
    Pipeline
        Complete ML pipeline
        
    Examples
    --------
    >>> from xgboost import XGBRegressor
    >>> from src.models.sklearn_pipeline import create_full_pipeline
    >>> 
    >>> model = XGBRegressor(n_estimators=100, random_state=42)
    >>> pipeline = create_full_pipeline(model, 'xgboost')
    >>> 
    >>> # Training
    >>> pipeline.fit(X_train, y_train)
    >>> 
    >>> # Prediction
    >>> y_pred = pipeline.predict(X_test)
    """
    # Feature engineering
    feature_engineer = create_feature_engineering_pipeline()

    # Preprocessing
    preprocessor = create_preprocessing_pipeline()

    # Complete pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', feature_engineer),
        ('preprocessing', preprocessor),
        (model_name, model)
    ])

    return full_pipeline


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test, cv=5):
    """
    Evaluates a pipeline with cross-validation and test metrics.
    
    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline to evaluate
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    cv : int, default=5
        Number of cross-validation folds
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - cv_rmse_mean: Mean RMSE from cross-validation
        - cv_rmse_std: Std RMSE from cross-validation
        - test_rmse: RMSE on test set
        - test_mae: MAE on test set
        - test_cv_percent: Coefficient of variation on test set
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    cv_rmse = -cv_scores

    # Train and evaluate on test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_cv = (test_rmse / y_test.mean()) * 100

    results = {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_cv_percent': test_cv
    }

    return results


def print_pipeline_structure(pipeline):
    """
    Prints the structure of a pipeline for debugging.
    
    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline to inspect
    """
    print("\n" + "="*60)
    print("PIPELINE STRUCTURE")
    print("="*60)

    for i, (name, step) in enumerate(pipeline.steps, 1):
        print(f"\nStep {i}: {name}")
        print(f"  Type: {type(step).__name__}")

        if hasattr(step, 'transformers'):
            print("  Transformers:")
            for trans_name, trans, cols in step.transformers:
                print(f"    - {trans_name}: {type(trans).__name__}")
                if isinstance(cols, list):
                    print(f"      Columns: {', '.join(cols)}")

        if hasattr(step, 'steps'):
            print("  Sub-steps:")
            for sub_name, sub_step in step.steps:
                print(f"    - {sub_name}: {type(sub_step).__name__}")

    print("\n" + "="*60)
