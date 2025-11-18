"""
Integration tests for feature engineering and model training.

Tests integration between feature transformers and model pipelines.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_training_data():
    """
    Create sample training data.

    Returns
    -------
    tuple
        (X, y) training data
    """
    np.random.seed(42)
    n_samples = 100

    X = pl.DataFrame(
        {
            "NSM": np.random.randint(0, 86400, n_samples),
            "Lagging_Current_Reactive.Power_kVarh": np.random.uniform(10, 50, n_samples),
            "Leading_Current_Reactive_Power_kVarh": np.random.uniform(5, 30, n_samples),
            "CO2(tCO2)": np.random.uniform(0.01, 0.1, n_samples),
            "Lagging_Current_Power_Factor": np.random.uniform(0.7, 0.95, n_samples),
            "Leading_Current_Power_Factor": np.random.uniform(0.8, 0.98, n_samples),
            "day": np.random.randint(0, 7, n_samples),
        }
    )

    y = np.random.uniform(30, 60, n_samples)

    return X, y


class TestFeatureEngineeringIntegration:
    """Test feature engineering integration with models"""

    def test_temporal_features_with_model(self, sample_training_data):
        """Test temporal feature engineering with model training"""
        from src.features.temporal_transformers import TemporalFeatureEngineer

        X, y = sample_training_data

        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        X_transformed = engineer.fit_transform(X)

        assert "hour" in X_transformed.columns

        X_array = X_transformed.select(["hour"]).to_numpy()

        model = LinearRegression()
        model.fit(X_array, y)

        predictions = model.predict(X_array)

        assert len(predictions) == len(y)
        assert all(pred >= 0 for pred in predictions[:10])

    def test_feature_pipeline_integration(self, sample_training_data):
        """Test complete feature pipeline with model"""
        from src.features.temporal_transformers import TemporalFeatureEngineer

        X, y = sample_training_data

        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        X_features = engineer.fit_transform(X)

        feature_cols = ["hour"]
        X_model = X_features.select(feature_cols).to_numpy()

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_model, y)

        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == len(feature_cols)


class TestModelPipelineIntegration:
    """Test sklearn pipeline integration"""

    def test_complete_sklearn_pipeline(self, sample_training_data):
        """Test complete sklearn pipeline"""
        from sklearn.preprocessing import StandardScaler

        X, y = sample_training_data

        X_array = X.select(
            [
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "CO2(tCO2)",
            ]
        ).to_numpy()

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        pipeline.fit(X_array, y)

        predictions = pipeline.predict(X_array[:5])

        assert len(predictions) == 5

    def test_feature_model_pipeline_integration(self, sample_training_data):
        """Test integration of custom features with sklearn pipeline"""
        from src.features.temporal_transformers import TemporalFeatureEngineer

        X, y = sample_training_data

        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        X_features = engineer.fit_transform(X)

        X_model = X_features.select(["hour"]).to_numpy()

        pipeline = Pipeline([("model", LinearRegression())])

        pipeline.fit(X_model, y)
        predictions = pipeline.predict(X_model)

        assert len(predictions) == len(y)


class TestTrainTestSplitIntegration:
    """Test train/test split with models"""

    def test_split_with_training(self, sample_training_data):
        """Test data split with model training"""
        from src.data.split_data import split_time_series_data

        X, y = sample_training_data
        X_array = X.to_numpy()

        X_train, X_test, y_train, y_test = split_time_series_data(X_array, y, test_size=0.2)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        assert train_score >= 0
        assert test_score >= -1

    def test_feature_engineering_with_split(self, sample_training_data):
        """Test feature engineering on split data"""
        from src.data.split_data import split_time_series_data

        from src.features.temporal_transformers import TemporalFeatureEngineer

        X, y = sample_training_data

        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        X_features = engineer.fit_transform(X)

        X_array = X_features.select(["hour"]).to_numpy()

        X_train, X_test, y_train, y_test = split_time_series_data(X_array, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        assert len(train_predictions) == len(y_train)
        assert len(test_predictions) == len(y_test)


class TestModelEvaluationIntegration:
    """Test model evaluation with training pipeline"""

    def test_train_evaluate_workflow(self, sample_training_data):
        """Test complete train and evaluate workflow"""
        from sklearn.metrics import mean_squared_error, r2_score
        from src.data.split_data import split_time_series_data

        X, y = sample_training_data
        X_array = X.select(
            ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh"]
        ).to_numpy()

        X_train, X_test, y_train, y_test = split_time_series_data(X_array, y, test_size=0.2)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        assert rmse >= 0
        assert -1 <= r2 <= 1

    def test_multiple_models_comparison(self, sample_training_data):
        """Test training and comparing multiple models"""
        from sklearn.metrics import mean_squared_error
        from src.data.split_data import split_time_series_data

        X, y = sample_training_data
        X_array = X.select(["Lagging_Current_Reactive.Power_kVarh", "CO2(tCO2)"]).to_numpy()

        X_train, X_test, y_train, y_test = split_time_series_data(X_array, y, test_size=0.2)

        models = {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=10, random_state=42),
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            results[name] = rmse

        assert len(results) == 2
        assert all(rmse >= 0 for rmse in results.values())


class TestFeatureImportanceIntegration:
    """Test feature importance analysis integration"""

    def test_feature_importance_extraction(self, sample_training_data):
        """Test extracting feature importance from trained model"""
        X, y = sample_training_data
        X_array = X.select(
            [
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "CO2(tCO2)",
            ]
        ).to_numpy()

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_array, y)

        importances = model.feature_importances_

        assert len(importances) == 3
        assert np.sum(importances) <= 1.0 + 1e-6


class TestModelPersistenceIntegration:
    """Test model saving and loading integration"""

    def test_model_save_load_predictions(self, sample_training_data, tmp_path):
        """Test saving model and loading for predictions"""
        import joblib

        X, y = sample_training_data
        X_array = X.select(["Lagging_Current_Reactive.Power_kVarh", "CO2(tCO2)"]).to_numpy()

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_array, y)

        predictions_before = model.predict(X_array[:5])

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        loaded_model = joblib.load(model_path)
        predictions_after = loaded_model.predict(X_array[:5])

        np.testing.assert_array_almost_equal(predictions_before, predictions_after)

    def test_pipeline_persistence(self, sample_training_data, tmp_path):
        """Test saving and loading complete pipeline"""
        import joblib
        from sklearn.preprocessing import StandardScaler

        X, y = sample_training_data
        X_array = X.select(["Lagging_Current_Reactive.Power_kVarh", "CO2(tCO2)"]).to_numpy()

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        pipeline.fit(X_array, y)

        pipeline_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)

        loaded_pipeline = joblib.load(pipeline_path)

        predictions_original = pipeline.predict(X_array[:5])
        predictions_loaded = loaded_pipeline.predict(X_array[:5])

        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)


class TestCrossValidationIntegration:
    """Test cross-validation integration"""

    def test_cross_validation_with_features(self, sample_training_data):
        """Test cross-validation with feature engineering"""
        from sklearn.model_selection import cross_val_score

        X, y = sample_training_data
        X_array = X.select(
            [
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "CO2(tCO2)",
            ]
        ).to_numpy()

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        scores = cross_val_score(model, X_array, y, cv=3, scoring="neg_mean_squared_error")

        assert len(scores) == 3
        assert all(score <= 0 for score in scores)


class TestEndToEndModelWorkflow:
    """Test complete end-to-end model workflow"""

    def test_complete_workflow(self, sample_training_data, tmp_path):
        """Test complete workflow: features -> split -> train -> evaluate -> save"""
        import joblib
        from sklearn.metrics import mean_squared_error
        from src.data.split_data import split_time_series_data

        from src.features.temporal_transformers import TemporalFeatureEngineer

        X, y = sample_training_data

        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        X_features = engineer.fit_transform(X)

        feature_cols = ["hour"]
        X_model = X_features.select(feature_cols).to_numpy()

        X_train, X_test, y_train, y_test = split_time_series_data(X_model, y, test_size=0.2)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        assert rmse >= 0

        model_path = tmp_path / "final_model.pkl"
        joblib.dump(model, model_path)

        assert model_path.exists()

        loaded_model = joblib.load(model_path)
        loaded_predictions = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(predictions, loaded_predictions)
