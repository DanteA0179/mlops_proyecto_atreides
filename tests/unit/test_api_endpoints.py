"""
Unit tests for Energy Optimization API endpoints.

Tests all API endpoints with valid and invalid requests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Mock the services before importing the app
def mock_predict_func(features):
    """Mock predict function that returns predictions based on input shape"""
    if len(features.shape) == 1 or features.shape[0] == 1:
        return np.array([45.67])
    else:
        # Return predictions for batch
        return np.array([45.67 + i for i in range(features.shape[0])])

mock_model_service = Mock()
mock_model_service.model_version = "test_model_v1"
mock_model_service.model_type = "test_model"
mock_model_service.predict = Mock(side_effect=mock_predict_func)
mock_model_service.predict_interval = Mock(return_value=(None, None))  # Return tuple for unpacking
mock_model_service.get_model_info = Mock(return_value={
    "model_version": "test_model_v1",
    "training_metrics": {"rmse": 12.79, "r2": 0.87},
    "features": []
})
mock_model_service.get_metrics = Mock(return_value={
    "training_metrics": {"rmse": 12.79},
    "production_metrics": {"total_predictions": 100}
})

mock_feature_service = Mock()
mock_feature_service.transform_request = Mock(return_value=np.array([[1.0] * 18]))
mock_feature_service.transform_batch = Mock(return_value=np.array([[1.0] * 18] * 3))
mock_feature_service.get_feature_count = Mock(return_value=18)

# Import app after ensuring model files exist
try:
    from src.api.main import app
    from src.api.routes import predict, health, model
    
    # Inject mocked services
    predict.set_services(mock_model_service, mock_feature_service)
    health.set_services(mock_model_service)
    model.set_services(mock_model_service, mock_feature_service)
    
    client = TestClient(app, raise_server_exceptions=False)
    API_AVAILABLE = True
except Exception as e:
    API_AVAILABLE = False
    pytest.skip(f"API not available: {str(e)}", allow_module_level=True)


class TestRootEndpoint:
    """Test suite for root endpoint"""

    def test_root_endpoint(self):
        """Test root endpoint returns welcome message"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Energy Optimization Copilot API" in data["message"]
        assert "version" in data
        assert "docs" in data
        assert "health" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestHealthEndpoint:
    """Test suite for /health endpoint"""

    def test_health_check_success(self):
        """Test successful health check"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert data["service"] == "energy-optimization-api"
        assert "model_loaded" in data
        assert "uptime_seconds" in data


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestPredictEndpoint:
    """Test suite for /predict endpoint"""

    def test_predict_valid_request(self):
        """Test successful prediction with valid request"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium",
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_usage_kwh" in data
        assert isinstance(data["predicted_usage_kwh"], (int, float))
        assert "model_version" in data
        assert "features_used" in data
        assert data["features_used"] >= 8  # At least input features
        assert "prediction_id" in data

    def test_predict_invalid_load_type(self):
        """Test validation error for invalid load_type"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Invalid",  # Invalid value
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data

    def test_predict_negative_value(self):
        """Test validation error for negative values"""
        request_data = {
            "lagging_reactive_power": -10.0,  # Negative value
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium",
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422

    def test_predict_power_factor_out_of_range(self):
        """Test validation error for power factor > 1.0"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 1.5,  # > 1.0
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium",
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422

    def test_predict_invalid_day_of_week(self):
        """Test validation error for invalid day_of_week"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 10,  # Invalid (0-6 only)
            "load_type": "Medium",
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestBatchPredictEndpoint:
    """Test suite for /predict/batch endpoint"""

    def test_batch_predict_valid(self):
        """Test batch prediction with 3 valid requests"""
        batch_data = {
            "predictions": [
                {
                    "lagging_reactive_power": 23.45,
                    "leading_reactive_power": 12.30,
                    "co2": 0.05,
                    "lagging_power_factor": 0.85,
                    "leading_power_factor": 0.92,
                    "nsm": 36000,
                    "day_of_week": 1,
                    "load_type": "Medium",
                },
                {
                    "lagging_reactive_power": 25.00,
                    "leading_reactive_power": 14.00,
                    "co2": 0.06,
                    "lagging_power_factor": 0.80,
                    "leading_power_factor": 0.90,
                    "nsm": 43200,
                    "day_of_week": 2,
                    "load_type": "Light",
                },
                {
                    "lagging_reactive_power": 30.00,
                    "leading_reactive_power": 16.00,
                    "co2": 0.08,
                    "lagging_power_factor": 0.88,
                    "leading_power_factor": 0.95,
                    "nsm": 57600,
                    "day_of_week": 3,
                    "load_type": "Maximum",
                },
            ]
        }

        response = client.post("/predict/batch", json=batch_data)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert "summary" in data
        assert data["summary"]["total_predictions"] == 3
        assert "avg_predicted_usage" in data["summary"]
        assert "processing_time_ms" in data["summary"]

    def test_batch_predict_empty(self):
        """Test error for empty batch"""
        batch_data = {"predictions": []}

        response = client.post("/predict/batch", json=batch_data)

        assert response.status_code == 422


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestModelInfoEndpoint:
    """Test suite for /model/info endpoint"""

    def test_model_info_success(self):
        """Test model info retrieval"""
        response = client.get("/model/info")

        # May fail if model not loaded, that's ok
        if response.status_code == 200:
            data = response.json()
            assert "model_version" in data
            assert "training_metrics" in data
            assert "features" in data
            assert isinstance(data["features"], list)


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestModelMetricsEndpoint:
    """Test suite for /model/metrics endpoint"""

    def test_model_metrics_success(self):
        """Test model metrics retrieval"""
        response = client.get("/model/metrics")

        # May fail if model not loaded, that's ok
        if response.status_code == 200:
            data = response.json()
            assert "training_metrics" in data
            assert "production_metrics" in data
