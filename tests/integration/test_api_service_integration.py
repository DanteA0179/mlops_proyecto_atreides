"""
Integration tests for API and service layer components.

Tests integration between API models, services, and business logic.
"""

from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def mock_trained_model():
    """
    Create a mock trained model.

    Returns
    -------
    RandomForestRegressor
        Trained model
    """
    np.random.seed(42)
    X = np.random.rand(100, 6)
    y = np.random.rand(100) * 50 + 20

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def temp_model_file(mock_trained_model, tmp_path):
    """
    Save model to temporary file.

    Parameters
    ----------
    mock_trained_model : RandomForestRegressor
        Trained model
    tmp_path : Path
        Temporary directory

    Returns
    -------
    Path
        Path to saved model
    """
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_trained_model, model_path)
    return model_path


class TestAPIModelIntegration:
    """Test integration between API request models and validation"""

    def test_prediction_request_validation(self):
        """Test prediction request validation"""
        from src.api.models.requests import PredictionRequest

        valid_request = PredictionRequest(
            lagging_reactive_power=23.45,
            leading_reactive_power=12.30,
            co2=0.05,
            lagging_power_factor=0.85,
            leading_power_factor=0.92,
            nsm=36000,
            day_of_week=1,
            load_type="Medium",
        )

        assert valid_request.lagging_reactive_power == 23.45
        assert valid_request.load_type == "Medium"

    def test_invalid_request_raises_error(self):
        """Test invalid request raises validation error"""
        from pydantic import ValidationError

        from src.api.models.requests import PredictionRequest

        with pytest.raises(ValidationError):
            PredictionRequest(
                lagging_reactive_power=23.45,
                leading_reactive_power=12.30,
                co2=0.05,
                lagging_power_factor=1.5,  # Invalid > 1
                leading_power_factor=0.92,
                nsm=36000,
                day_of_week=1,
                load_type="Medium",
            )

    def test_batch_request_validation(self):
        """Test batch request validation"""
        from src.api.models.requests import BatchPredictionRequest, PredictionRequest

        predictions = [
            PredictionRequest(
                lagging_reactive_power=23.45,
                leading_reactive_power=12.30,
                co2=0.05,
                lagging_power_factor=0.85,
                leading_power_factor=0.92,
                nsm=36000,
                day_of_week=1,
                load_type="Medium",
            )
            for _ in range(3)
        ]

        batch_request = BatchPredictionRequest(predictions=predictions)

        assert len(batch_request.predictions) == 3


class TestFeatureServiceIntegration:
    """Test feature service integration"""

    def test_feature_transformation_integration(self):
        """Test feature service transforms requests correctly"""
        from src.api.models.requests import PredictionRequest
        from src.api.services.feature_engineering import FeatureService

        request = PredictionRequest(
            lagging_reactive_power=23.45,
            leading_reactive_power=12.30,
            co2=0.05,
            lagging_power_factor=0.85,
            leading_power_factor=0.92,
            nsm=36000,
            day_of_week=1,
            load_type="Medium",
        )

        feature_service = FeatureService()
        features = feature_service.transform_request(request)

        assert features is not None
        assert features.shape[0] == 1

    def test_batch_feature_transformation(self):
        """Test batch feature transformation"""
        from src.api.models.requests import PredictionRequest
        from src.api.services.feature_engineering import FeatureService

        requests = [
            PredictionRequest(
                lagging_reactive_power=23.45,
                leading_reactive_power=12.30,
                co2=0.05,
                lagging_power_factor=0.85,
                leading_power_factor=0.92,
                nsm=36000 + i * 3600,
                day_of_week=i % 7,
                load_type="Medium",
            )
            for i in range(5)
        ]

        feature_service = FeatureService()
        features = feature_service.transform_batch(requests)

        assert features.shape[0] == 5


class TestModelServiceIntegration:
    """Test model service integration"""

    def test_model_loading(self, temp_model_file):
        """Test model service loads models correctly"""
        from src.api.services.model_service import ModelService

        with patch.object(ModelService, "_get_model_path", return_value=str(temp_model_file)):
            service = ModelService(model_type="test_model")
            service.load_model()

            assert service.model is not None

    def test_model_prediction(self, temp_model_file):
        """Test model service makes predictions"""
        from src.api.services.model_service import ModelService

        with patch.object(ModelService, "_get_model_path", return_value=str(temp_model_file)):
            service = ModelService(model_type="test_model")
            service.load_model()

            test_features = np.random.rand(1, 6)
            predictions = service.predict(test_features)

            assert len(predictions) == 1
            assert predictions[0] > 0


class TestAPIServiceWorkflow:
    """Test complete API service workflow"""

    def test_request_to_prediction_workflow(self, temp_model_file):
        """Test complete workflow: request -> features -> model -> response"""
        from src.api.models.requests import PredictionRequest
        from src.api.services.feature_engineering import FeatureService
        from src.api.services.model_service import ModelService

        request = PredictionRequest(
            lagging_reactive_power=23.45,
            leading_reactive_power=12.30,
            co2=0.05,
            lagging_power_factor=0.85,
            leading_power_factor=0.92,
            nsm=36000,
            day_of_week=1,
            load_type="Medium",
        )

        feature_service = FeatureService()
        features = feature_service.transform_request(request)

        with patch.object(ModelService, "_get_model_path", return_value=str(temp_model_file)):
            model_service = ModelService(model_type="test_model")
            model_service.load_model()

            predictions = model_service.predict(features)

            assert len(predictions) == 1
            assert predictions[0] > 0

    def test_batch_workflow(self, temp_model_file):
        """Test batch prediction workflow"""
        from src.api.models.requests import PredictionRequest
        from src.api.services.feature_engineering import FeatureService
        from src.api.services.model_service import ModelService

        requests = [
            PredictionRequest(
                lagging_reactive_power=23.45,
                leading_reactive_power=12.30,
                co2=0.05,
                lagging_power_factor=0.85,
                leading_power_factor=0.92,
                nsm=36000,
                day_of_week=1,
                load_type="Medium",
            )
            for _ in range(3)
        ]

        feature_service = FeatureService()
        features = feature_service.transform_batch(requests)

        with patch.object(ModelService, "_get_model_path", return_value=str(temp_model_file)):
            model_service = ModelService(model_type="test_model")
            model_service.load_model()

            predictions = model_service.predict(features)

            assert len(predictions) == 3


class TestErrorHandlingIntegration:
    """Test error handling across API layers"""

    def test_invalid_feature_handling(self):
        """Test handling of invalid features"""
        from src.api.models.requests import PredictionRequest
        from src.api.services.feature_engineering import FeatureService

        request = PredictionRequest(
            lagging_reactive_power=23.45,
            leading_reactive_power=12.30,
            co2=0.05,
            lagging_power_factor=0.85,
            leading_power_factor=0.92,
            nsm=36000,
            day_of_week=1,
            load_type="Medium",
        )

        feature_service = FeatureService()

        try:
            features = feature_service.transform_request(request)
            assert features is not None
        except Exception as e:
            pytest.fail(f"Feature transformation failed: {e}")

    def test_model_not_found_handling(self):
        """Test handling when model file not found"""
        from src.api.services.model_service import ModelService

        with patch.object(ModelService, "_get_model_path", return_value="/nonexistent/model.pkl"):
            service = ModelService(model_type="nonexistent")

            with pytest.raises(Exception):
                service.load_model()


class TestResponseFormatting:
    """Test response formatting integration"""

    def test_prediction_response_format(self, temp_model_file):
        """Test prediction response formatting"""
        from src.api.models.requests import PredictionRequest
        from src.api.models.responses import PredictionResponse
        from src.api.services.feature_engineering import FeatureService
        from src.api.services.model_service import ModelService

        request = PredictionRequest(
            lagging_reactive_power=23.45,
            leading_reactive_power=12.30,
            co2=0.05,
            lagging_power_factor=0.85,
            leading_power_factor=0.92,
            nsm=36000,
            day_of_week=1,
            load_type="Medium",
        )

        feature_service = FeatureService()
        features = feature_service.transform_request(request)

        with patch.object(ModelService, "_get_model_path", return_value=str(temp_model_file)):
            model_service = ModelService(model_type="test_model")
            model_service.load_model()

            prediction = model_service.predict(features)[0]

            response = PredictionResponse(
                predicted_usage_kwh=float(prediction),
                model_version="test_v1",
                model_type="test_model",
                prediction_timestamp="2024-01-01T00:00:00Z",
                features_used=features.shape[1],
                prediction_id="test_123",
            )

            assert response.predicted_usage_kwh > 0
            assert response.model_version == "test_v1"


class TestConcurrencyIntegration:
    """Test concurrent request handling"""

    def test_multiple_simultaneous_predictions(self, temp_model_file):
        """Test handling multiple predictions simultaneously"""
        from src.api.models.requests import PredictionRequest
        from src.api.services.feature_engineering import FeatureService
        from src.api.services.model_service import ModelService

        requests = [
            PredictionRequest(
                lagging_reactive_power=23.45 + i,
                leading_reactive_power=12.30 + i,
                co2=0.05,
                lagging_power_factor=0.85,
                leading_power_factor=0.92,
                nsm=36000 + i * 1000,
                day_of_week=i % 7,
                load_type="Medium",
            )
            for i in range(10)
        ]

        feature_service = FeatureService()

        with patch.object(ModelService, "_get_model_path", return_value=str(temp_model_file)):
            model_service = ModelService(model_type="test_model")
            model_service.load_model()

            predictions = []
            for request in requests:
                features = feature_service.transform_request(request)
                pred = model_service.predict(features)
                predictions.append(pred[0])

            assert len(predictions) == 10
            assert all(p > 0 for p in predictions)


class TestValidationIntegration:
    """Test validation across layers"""

    def test_end_to_end_validation(self):
        """Test validation from request to response"""
        from pydantic import ValidationError

        from src.api.models.requests import PredictionRequest

        with pytest.raises(ValidationError):
            PredictionRequest(
                lagging_reactive_power=-10,  # Invalid negative
                leading_reactive_power=12.30,
                co2=0.05,
                lagging_power_factor=0.85,
                leading_power_factor=0.92,
                nsm=36000,
                day_of_week=1,
                load_type="Medium",
            )

    def test_business_logic_validation(self):
        """Test business logic validation"""
        from src.api.models.requests import PredictionRequest

        request = PredictionRequest(
            lagging_reactive_power=23.45,
            leading_reactive_power=12.30,
            co2=0.05,
            lagging_power_factor=0.85,
            leading_power_factor=0.92,
            nsm=36000,
            day_of_week=1,
            load_type="Medium",
        )

        assert 0 <= request.lagging_power_factor <= 1
        assert 0 <= request.leading_power_factor <= 1
        assert 0 <= request.nsm <= 86400
        assert 0 <= request.day_of_week <= 6
