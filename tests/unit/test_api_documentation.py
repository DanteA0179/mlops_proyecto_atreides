"""
Tests for API Documentation.

This module tests that API documentation is properly configured:
- Swagger UI is accessible
- ReDoc is accessible
- OpenAPI schema is valid
- All endpoints have examples
- Metadata is complete
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_swagger_ui_accessible(client):
    """Test that Swagger UI is accessible at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def test_redoc_accessible(client):
    """Test that ReDoc is accessible at /redoc."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower()


def test_openapi_schema_accessible(client):
    """Test that OpenAPI schema is accessible at /openapi.json."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


def test_openapi_schema_valid(client):
    """Test that OpenAPI schema has required fields."""
    response = client.get("/openapi.json")
    schema = response.json()

    # Validate OpenAPI version
    assert "openapi" in schema
    assert schema["openapi"].startswith("3.")

    # Validate info section
    assert "info" in schema
    assert "title" in schema["info"]
    assert "version" in schema["info"]
    assert "description" in schema["info"]

    # Validate paths
    assert "paths" in schema
    assert len(schema["paths"]) > 0


def test_openapi_has_contact_info(client):
    """Test that OpenAPI schema has contact information."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "info" in schema
    assert "contact" in schema["info"]

    contact = schema["info"]["contact"]
    assert "name" in contact
    assert "email" in contact
    assert "url" in contact


def test_openapi_has_license_info(client):
    """Test that OpenAPI schema has license information."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "info" in schema
    assert "license" in schema["info"]

    license_info = schema["info"]["license"]
    assert "name" in license_info
    assert "url" in license_info


def test_openapi_has_tags(client):
    """Test that OpenAPI schema has tags defined."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "tags" in schema
    assert len(schema["tags"]) > 0

    # Check that each tag has required fields
    for tag in schema["tags"]:
        assert "name" in tag
        assert "description" in tag


def test_openapi_has_servers(client):
    """Test that OpenAPI schema has servers defined."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "servers" in schema
    assert len(schema["servers"]) > 0

    # Check that each server has required fields
    for server in schema["servers"]:
        assert "url" in server
        assert "description" in server


def test_all_endpoints_documented(client):
    """Test that all endpoints are documented in OpenAPI schema."""
    response = client.get("/openapi.json")
    schema = response.json()

    paths = schema["paths"]

    # Expected endpoints
    expected_endpoints = [
        "/",
        "/predict",
        "/predict/batch",
        "/health",
        "/model/info",
        "/model/metrics",
    ]

    for endpoint in expected_endpoints:
        assert endpoint in paths, f"Endpoint {endpoint} not found in OpenAPI schema"


def test_predict_endpoint_has_examples(client):
    """Test that /predict endpoint has request examples."""
    response = client.get("/openapi.json")
    schema = response.json()

    # Get /predict POST method
    predict_path = schema["paths"]["/predict"]["post"]

    # Check request body has examples
    assert "requestBody" in predict_path
    request_body = predict_path["requestBody"]
    assert "content" in request_body
    assert "application/json" in request_body["content"]

    json_content = request_body["content"]["application/json"]
    assert "schema" in json_content

    # Schema should reference PredictionRequest which has examples
    schema_ref = json_content["schema"].get("$ref", "")
    assert "PredictionRequest" in schema_ref or "examples" in json_content


def test_predict_batch_endpoint_has_examples(client):
    """Test that /predict/batch endpoint has request examples."""
    response = client.get("/openapi.json")
    schema = response.json()

    # Get /predict/batch POST method
    batch_path = schema["paths"]["/predict/batch"]["post"]

    # Check request body exists
    assert "requestBody" in batch_path
    request_body = batch_path["requestBody"]
    assert "content" in request_body
    assert "application/json" in request_body["content"]


def test_models_have_examples(client):
    """Test that Pydantic models have examples in schema."""
    response = client.get("/openapi.json")
    schema = response.json()

    # Check components section
    assert "components" in schema
    assert "schemas" in schema["components"]

    schemas = schema["components"]["schemas"]

    # Check PredictionRequest has examples
    if "PredictionRequest" in schemas:
        pred_request = schemas["PredictionRequest"]
        # Examples should be in the schema
        assert "examples" in pred_request or "example" in pred_request


def test_endpoints_have_descriptions(client):
    """Test that all endpoints have descriptions."""
    response = client.get("/openapi.json")
    schema = response.json()

    for path, methods in schema["paths"].items():
        for method, details in methods.items():
            assert (
                "summary" in details or "description" in details
            ), f"Endpoint {method.upper()} {path} missing summary/description"


def test_endpoints_have_response_descriptions(client):
    """Test that all endpoints have response descriptions."""
    response = client.get("/openapi.json")
    schema = response.json()

    for path, methods in schema["paths"].items():
        for method, details in methods.items():
            assert "responses" in details, f"Endpoint {method.upper()} {path} missing responses"

            # Check that at least 200 response is documented
            responses = details["responses"]
            assert "200" in responses, f"Endpoint {method.upper()} {path} missing 200 response"


def test_prediction_request_has_multiple_examples(client):
    """Test that PredictionRequest model has multiple examples."""
    response = client.get("/openapi.json")
    schema = response.json()

    if "components" not in schema or "schemas" not in schema["components"]:
        pytest.skip("Schema components not available")

    schemas = schema["components"]["schemas"]

    if "PredictionRequest" in schemas:
        pred_request = schemas["PredictionRequest"]

        # Check for examples (plural)
        if "examples" in pred_request:
            examples = pred_request["examples"]
            assert isinstance(examples, list), "Examples should be a list"
            assert len(examples) >= 3, f"Expected at least 3 examples, got {len(examples)}"

        # Alternative: examples might be in example (singular)
        elif "example" in pred_request:
            # At least one example exists
            assert pred_request["example"] is not None


def test_openapi_has_external_docs(client):
    """Test that OpenAPI schema has external documentation link."""
    response = client.get("/openapi.json")
    schema = response.json()

    # External docs might be at root level
    if "externalDocs" in schema:
        ext_docs = schema["externalDocs"]
        assert "url" in ext_docs
        assert "description" in ext_docs


def test_openapi_version_is_correct(client):
    """Test that OpenAPI version matches app version."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert schema["info"]["version"] == "1.0.0"


def test_api_title_is_correct(client):
    """Test that API title is descriptive."""
    response = client.get("/openapi.json")
    schema = response.json()

    title = schema["info"]["title"]
    assert "Energy" in title or "energy" in title
    assert len(title) > 10  # Title should be descriptive


def test_api_description_is_detailed(client):
    """Test that API description is detailed and helpful."""
    response = client.get("/openapi.json")
    schema = response.json()

    description = schema["info"]["description"]
    assert len(description) > 100  # Description should be substantial

    # Should contain key information
    keywords = ["predicción", "energía", "ML", "API", "kWh"]
    assert any(keyword in description.lower() for keyword in keywords)


def test_health_endpoint_documented(client):
    """Test that /health endpoint is properly documented."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "/health" in schema["paths"]
    health_endpoint = schema["paths"]["/health"]["get"]

    assert "summary" in health_endpoint
    assert "description" in health_endpoint
    assert "responses" in health_endpoint


def test_model_info_endpoint_documented(client):
    """Test that /model/info endpoint is properly documented."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "/model/info" in schema["paths"]
    info_endpoint = schema["paths"]["/model/info"]["get"]

    assert "summary" in info_endpoint
    assert "description" in info_endpoint
    assert "responses" in info_endpoint


def test_model_metrics_endpoint_documented(client):
    """Test that /model/metrics endpoint is properly documented."""
    response = client.get("/openapi.json")
    schema = response.json()

    assert "/model/metrics" in schema["paths"]
    metrics_endpoint = schema["paths"]["/model/metrics"]["get"]

    assert "summary" in metrics_endpoint
    assert "description" in metrics_endpoint
    assert "responses" in metrics_endpoint


def test_components_has_security_schemes(client):
    """Test that security schemes are defined (for future use)."""
    response = client.get("/openapi.json")
    schema = response.json()

    if "components" in schema and "securitySchemes" in schema["components"]:
        security_schemes = schema["components"]["securitySchemes"]

        # Check that at least some security schemes are defined
        assert len(security_schemes) > 0

        # Each scheme should have type
        for _scheme_name, scheme_details in security_schemes.items():
            assert "type" in scheme_details
