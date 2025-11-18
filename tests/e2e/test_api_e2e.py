"""
End-to-end tests for Energy Optimization API.

Tests the complete API workflow from startup to predictions,
including model loading, feature engineering, and error handling.
"""

import time

import pytest
import requests
from requests.exceptions import ConnectionError, Timeout


@pytest.fixture(scope="module")
def api_base_url() -> str:
    """
    API base URL fixture.

    Returns
    -------
    str
        Base URL for API endpoints
    """
    return "http://localhost:8000"


@pytest.fixture(scope="module")
def api_health_check(api_base_url: str):
    """
    Verify API is running before tests.

    Parameters
    ----------
    api_base_url : str
        Base URL for API

    Raises
    ------
    pytest.skip
        If API is not available
    """
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except (ConnectionError, Timeout):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                pytest.skip("API is not running. Start with: uvicorn src.api.main:app")

    pytest.skip("API health check failed")


@pytest.fixture
def valid_prediction_request() -> dict:
    """
    Valid prediction request fixture.

    Returns
    -------
    Dict
        Valid request payload for single prediction
    """
    return {
        "lagging_reactive_power": 23.45,
        "leading_reactive_power": 12.30,
        "co2": 0.05,
        "lagging_power_factor": 0.85,
        "leading_power_factor": 0.92,
        "nsm": 36000,
        "day_of_week": 1,
        "load_type": "Medium",
    }


@pytest.fixture
def valid_batch_request(valid_prediction_request: dict) -> dict:
    """
    Valid batch prediction request fixture.

    Parameters
    ----------
    valid_prediction_request : Dict
        Single prediction request

    Returns
    -------
    Dict
        Valid batch request with multiple predictions
    """
    predictions = []
    for i in range(3):
        pred = valid_prediction_request.copy()
        pred["nsm"] = 36000 + (i * 3600)
        predictions.append(pred)

    return {"predictions": predictions}


class TestAPILifecycle:
    """Test API lifecycle and initialization"""

    def test_api_is_running(self, api_base_url: str, api_health_check):
        """Test API server is running and accessible"""
        response = requests.get(api_base_url)
        assert response.status_code == 200

    def test_root_endpoint(self, api_base_url: str, api_health_check):
        """Test root endpoint returns correct information"""
        response = requests.get(api_base_url)
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "Energy Optimization Copilot API" in data["message"]
        assert "version" in data
        assert "docs" in data
        assert "endpoints" in data

    def test_openapi_docs_available(self, api_base_url: str, api_health_check):
        """Test OpenAPI documentation is accessible"""
        response = requests.get(f"{api_base_url}/docs")
        assert response.status_code == 200

        response = requests.get(f"{api_base_url}/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "paths" in openapi_spec


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_check_basic(self, api_base_url: str, api_health_check):
        """Test basic health check endpoint"""
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "service" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_check_model_loaded(self, api_base_url: str, api_health_check):
        """Test health check reports model status"""
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_check_details(self, api_base_url: str, api_health_check):
        """Test detailed health check endpoint"""
        response = requests.get(f"{api_base_url}/health/detailed")

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "components" in data


class TestModelEndpoints:
    """Test model information endpoints"""

    def test_model_info(self, api_base_url: str, api_health_check):
        """Test model info endpoint"""
        response = requests.get(f"{api_base_url}/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "training_date" in data or "created_at" in data

    def test_model_metrics(self, api_base_url: str, api_health_check):
        """Test model metrics endpoint"""
        response = requests.get(f"{api_base_url}/model/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "training_metrics" in data or "metrics" in data


class TestSinglePrediction:
    """Test single prediction endpoint"""

    def test_predict_valid_request(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test successful prediction with valid data"""
        response = requests.post(f"{api_base_url}/predict", json=valid_prediction_request)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_usage_kwh" in data
        assert isinstance(data["predicted_usage_kwh"], (int, float))
        assert data["predicted_usage_kwh"] > 0

        assert "model_version" in data
        assert "prediction_timestamp" in data
        assert "features_used" in data
        assert "prediction_id" in data

    def test_predict_light_load(self, api_base_url: str, api_health_check):
        """Test prediction with light load type"""
        request_data = {
            "lagging_reactive_power": 10.0,
            "leading_reactive_power": 5.0,
            "co2": 0.02,
            "lagging_power_factor": 0.90,
            "leading_power_factor": 0.95,
            "nsm": 28800,
            "day_of_week": 2,
            "load_type": "Light",
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["predicted_usage_kwh"] > 0

    def test_predict_maximum_load(self, api_base_url: str, api_health_check):
        """Test prediction with maximum load type"""
        request_data = {
            "lagging_reactive_power": 50.0,
            "leading_reactive_power": 30.0,
            "co2": 0.10,
            "lagging_power_factor": 0.75,
            "leading_power_factor": 0.80,
            "nsm": 43200,
            "day_of_week": 4,
            "load_type": "Maximum",
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["predicted_usage_kwh"] > 0

    def test_predict_confidence_intervals(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test prediction returns confidence intervals if available"""
        response = requests.post(f"{api_base_url}/predict", json=valid_prediction_request)
        assert response.status_code == 200

        data = response.json()
        if data.get("confidence_interval_lower") is not None:
            assert data["confidence_interval_lower"] <= data["predicted_usage_kwh"]
            assert data["confidence_interval_upper"] >= data["predicted_usage_kwh"]


class TestSinglePredictionValidation:
    """Test input validation for single predictions"""

    def test_predict_invalid_power_factor(self, api_base_url: str, api_health_check):
        """Test validation error for invalid power factor"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 1.5,  # Invalid > 1
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium",
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_negative_values(self, api_base_url: str, api_health_check):
        """Test validation error for negative values"""
        request_data = {
            "lagging_reactive_power": -10.0,  # Invalid negative
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Medium",
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_invalid_load_type(self, api_base_url: str, api_health_check):
        """Test validation error for invalid load type"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 1,
            "load_type": "Invalid",  # Invalid load type
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_invalid_day_of_week(self, api_base_url: str, api_health_check):
        """Test validation error for invalid day of week"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "day_of_week": 7,  # Invalid > 6
            "load_type": "Medium",
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_invalid_nsm(self, api_base_url: str, api_health_check):
        """Test validation error for invalid NSM"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 90000,  # Invalid > 86400
            "day_of_week": 1,
            "load_type": "Medium",
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_missing_fields(self, api_base_url: str, api_health_check):
        """Test validation error for missing required fields"""
        request_data = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            # Missing required fields
        }

        response = requests.post(f"{api_base_url}/predict", json=request_data)
        assert response.status_code == 422


class TestBatchPrediction:
    """Test batch prediction endpoint"""

    def test_batch_predict_valid_request(
        self, api_base_url: str, api_health_check, valid_batch_request: dict
    ):
        """Test successful batch prediction"""
        response = requests.post(f"{api_base_url}/predict/batch", json=valid_batch_request)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

        assert "summary" in data
        assert "total_predictions" in data["summary"]
        assert data["summary"]["total_predictions"] == 3
        assert "avg_predicted_usage" in data["summary"]
        assert "min_predicted_usage" in data["summary"]
        assert "max_predicted_usage" in data["summary"]
        assert "processing_time_ms" in data["summary"]

        assert "model_version" in data
        assert "batch_timestamp" in data

    def test_batch_predict_single_item(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test batch prediction with single item"""
        request_data = {"predictions": [valid_prediction_request]}

        response = requests.post(f"{api_base_url}/predict/batch", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["summary"]["total_predictions"] == 1

    def test_batch_predict_large_batch(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test batch prediction with larger batch"""
        predictions = []
        for i in range(50):
            pred = valid_prediction_request.copy()
            pred["nsm"] = (36000 + (i * 100)) % 86400
            predictions.append(pred)

        request_data = {"predictions": predictions}

        response = requests.post(f"{api_base_url}/predict/batch", json=request_data, timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 50
        assert data["summary"]["total_predictions"] == 50
        assert data["summary"]["processing_time_ms"] > 0

    def test_batch_predict_summary_statistics(
        self, api_base_url: str, api_health_check, valid_batch_request: dict
    ):
        """Test batch prediction summary statistics are correct"""
        response = requests.post(f"{api_base_url}/predict/batch", json=valid_batch_request)
        assert response.status_code == 200

        data = response.json()
        predictions = [p["predicted_usage_kwh"] for p in data["predictions"]]

        summary = data["summary"]
        assert summary["min_predicted_usage"] == min(predictions)
        assert summary["max_predicted_usage"] == max(predictions)
        assert abs(summary["avg_predicted_usage"] - (sum(predictions) / len(predictions))) < 0.01


class TestBatchPredictionValidation:
    """Test input validation for batch predictions"""

    def test_batch_predict_empty_list(self, api_base_url: str, api_health_check):
        """Test validation error for empty predictions list"""
        request_data = {"predictions": []}

        response = requests.post(f"{api_base_url}/predict/batch", json=request_data)
        assert response.status_code in [400, 422]

    def test_batch_predict_invalid_item(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test validation error for batch with invalid item"""
        invalid_request = valid_prediction_request.copy()
        invalid_request["lagging_power_factor"] = 1.5  # Invalid

        request_data = {"predictions": [valid_prediction_request, invalid_request]}

        response = requests.post(f"{api_base_url}/predict/batch", json=request_data)
        assert response.status_code == 422


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_complete_prediction_workflow(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test complete workflow: health check -> model info -> prediction"""
        # Check health
        health_response = requests.get(f"{api_base_url}/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] is True

        # Get model info
        model_response = requests.get(f"{api_base_url}/model/info")
        assert model_response.status_code == 200
        model_version = model_response.json()["model_version"]

        # Make prediction
        pred_response = requests.post(f"{api_base_url}/predict", json=valid_prediction_request)
        assert pred_response.status_code == 200

        pred_data = pred_response.json()
        assert pred_data["model_version"] == model_version
        assert pred_data["predicted_usage_kwh"] > 0

    def test_multiple_sequential_predictions(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test multiple sequential predictions maintain consistency"""
        predictions = []

        for _i in range(5):
            request = valid_prediction_request.copy()
            response = requests.post(f"{api_base_url}/predict", json=request)
            assert response.status_code == 200
            predictions.append(response.json()["predicted_usage_kwh"])

        # Same input should give same prediction
        assert all(abs(p - predictions[0]) < 0.01 for p in predictions)

    def test_mixed_load_types_workflow(self, api_base_url: str, api_health_check):
        """Test predictions across different load types"""
        load_types = ["Light", "Medium", "Maximum"]
        predictions = {}

        for load_type in load_types:
            request_data = {
                "lagging_reactive_power": 23.45,
                "leading_reactive_power": 12.30,
                "co2": 0.05,
                "lagging_power_factor": 0.85,
                "leading_power_factor": 0.92,
                "nsm": 36000,
                "day_of_week": 1,
                "load_type": load_type,
            }

            response = requests.post(f"{api_base_url}/predict", json=request_data)
            assert response.status_code == 200

            predictions[load_type] = response.json()["predicted_usage_kwh"]

        # All load types should return valid predictions
        assert all(p > 0 for p in predictions.values())

    def test_weekend_vs_weekday_workflow(self, api_base_url: str, api_health_check):
        """Test predictions for weekday vs weekend"""
        base_request = {
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92,
            "nsm": 36000,
            "load_type": "Medium",
        }

        # Weekday (Monday)
        weekday_request = base_request.copy()
        weekday_request["day_of_week"] = 0

        weekday_response = requests.post(f"{api_base_url}/predict", json=weekday_request)
        assert weekday_response.status_code == 200

        # Weekend (Saturday)
        weekend_request = base_request.copy()
        weekend_request["day_of_week"] = 5

        weekend_response = requests.post(f"{api_base_url}/predict", json=weekend_request)
        assert weekend_response.status_code == 200

        # Both should return valid predictions
        assert weekday_response.json()["predicted_usage_kwh"] > 0
        assert weekend_response.json()["predicted_usage_kwh"] > 0


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_endpoint(self, api_base_url: str, api_health_check):
        """Test 404 error for invalid endpoint"""
        response = requests.get(f"{api_base_url}/invalid_endpoint")
        assert response.status_code == 404

    def test_wrong_http_method(self, api_base_url: str, api_health_check):
        """Test error for wrong HTTP method"""
        response = requests.get(f"{api_base_url}/predict")
        assert response.status_code == 405

    def test_malformed_json(self, api_base_url: str, api_health_check):
        """Test error handling for malformed JSON"""
        response = requests.post(
            f"{api_base_url}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_content_type_validation(self, api_base_url: str, api_health_check):
        """Test content type validation"""
        response = requests.post(
            f"{api_base_url}/predict", data="some data", headers={"Content-Type": "text/plain"}
        )
        assert response.status_code in [415, 422]


class TestPerformance:
    """Test API performance characteristics"""

    def test_prediction_response_time(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test single prediction response time"""
        start_time = time.time()
        response = requests.post(f"{api_base_url}/predict", json=valid_prediction_request)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 2.0  # Should respond within 2 seconds

    def test_batch_prediction_efficiency(
        self, api_base_url: str, api_health_check, valid_prediction_request: dict
    ):
        """Test batch prediction is more efficient than individual calls"""
        # Batch prediction
        predictions = [valid_prediction_request.copy() for _ in range(10)]
        batch_request = {"predictions": predictions}

        batch_start = time.time()
        batch_response = requests.post(f"{api_base_url}/predict/batch", json=batch_request)
        batch_time = time.time() - batch_start

        assert batch_response.status_code == 200

        # Individual predictions
        individual_start = time.time()
        for _ in range(10):
            requests.post(f"{api_base_url}/predict", json=valid_prediction_request)
        individual_time = time.time() - individual_start

        # Batch should be faster or similar
        # Allow for some variance in timing
        assert batch_time < individual_time * 1.5
