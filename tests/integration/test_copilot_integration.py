"""Integration tests for copilot endpoint."""

import pytest
from fastapi.testclient import TestClient
import os


# Only run if Ollama is available
pytestmark = pytest.mark.skipif(
    os.getenv("OLLAMA_HOST") is None and os.getenv("LLM_PROVIDER") != "gemini",
    reason="LLM provider not configured"
)


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from src.api.main import app
    return TestClient(app)


class TestCopilotIntegration:
    """Integration tests with real LLM."""

    def test_chat_endpoint_exists(self, client):
        """Test that chat endpoint exists and accepts POST."""
        response = client.post("/copilot/chat", json={
            "message": "Hello"
        })
        # Should not return 404
        assert response.status_code != 404

    def test_chat_simple_query(self, client):
        """Test simple chat query."""
        response = client.post("/copilot/chat", json={
            "message": "What is energy consumption?"
        })

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "response" in data
        assert "conversation_id" in data
        assert "intent" in data
        assert "processing_time_ms" in data

        # Verify intent detection
        assert data["intent"] in ["explanation", "general"]

    def test_chat_prediction_query(self, client):
        """Test prediction query end-to-end."""
        response = client.post("/copilot/chat", json={
            "message": "What will be the consumption tomorrow at 10am with Medium load?"
        })

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "response" in data
        assert data["response"] != ""

        # Verify intent
        assert data["intent"] == "prediction"

        # Verify prediction data may be present
        # (might fail if model not loaded, which is acceptable)
        if "prediction_data" in data and data["prediction_data"] is not None:
            assert "predicted_usage_kwh" in data["prediction_data"]
            assert "model_version" in data["prediction_data"]

    def test_chat_validation_empty_message(self, client):
        """Test validation error for empty message."""
        response = client.post("/copilot/chat", json={
            "message": ""
        })

        assert response.status_code == 422  # Validation error

    def test_chat_validation_message_too_long(self, client):
        """Test validation error for message too long."""
        response = client.post("/copilot/chat", json={
            "message": "x" * 3000  # Exceeds 2000 char limit
        })

        assert response.status_code == 422

    def test_conversation_context(self, client):
        """Test conversation context is maintained."""
        # First message
        response1 = client.post("/copilot/chat", json={
            "message": "What is power factor?"
        })

        assert response1.status_code == 200
        conv_id = response1.json()["conversation_id"]

        # Second message with context
        response2 = client.post("/copilot/chat", json={
            "message": "How does it affect consumption?",
            "conversation_id": conv_id
        })

        assert response2.status_code == 200
        assert response2.json()["conversation_id"] == conv_id

    def test_get_conversation_history(self, client):
        """Test retrieving conversation history."""
        # Create a conversation
        response = client.post("/copilot/chat", json={
            "message": "Hello"
        })

        conv_id = response.json()["conversation_id"]

        # Get history
        history_response = client.get(f"/copilot/conversation/{conv_id}")

        assert history_response.status_code == 200
        history_data = history_response.json()

        assert "conversation_id" in history_data
        assert "messages" in history_data
        assert "message_count" in history_data
        assert history_data["conversation_id"] == conv_id
        assert history_data["message_count"] >= 2  # At least user + assistant

    def test_clear_conversation(self, client):
        """Test clearing conversation history."""
        # Create a conversation
        response = client.post("/copilot/chat", json={
            "message": "Hello"
        })

        conv_id = response.json()["conversation_id"]

        # Clear conversation
        clear_response = client.delete(f"/copilot/conversation/{conv_id}")

        assert clear_response.status_code == 200
        clear_data = clear_response.json()

        assert clear_data["message"] == "Conversation cleared successfully"
        assert clear_data["conversation_id"] == conv_id

        # Verify history is empty
        history_response = client.get(f"/copilot/conversation/{conv_id}")
        history_data = history_response.json()

        assert history_data["message_count"] == 0

    def test_intent_detection_analysis(self, client):
        """Test analysis intent detection."""
        response = client.post("/copilot/chat", json={
            "message": "Why was there a spike last Friday?"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "analysis"

    def test_intent_detection_whatif(self, client):
        """Test what-if intent detection."""
        response = client.post("/copilot/chat", json={
            "message": "What if I shift production to 2am?"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "what-if"

    def test_intent_detection_explanation(self, client):
        """Test explanation intent detection."""
        response = client.post("/copilot/chat", json={
            "message": "Explain reactive power"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "explanation"

    @pytest.mark.slow
    def test_latency_requirement(self, client):
        """Test latency is within acceptable range."""
        import time

        start = time.time()
        response = client.post("/copilot/chat", json={
            "message": "Explain energy consumption"
        })
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200

        # Latency should be reasonable
        # This is lenient to account for cold starts
        assert elapsed < 30000, f"Latency too high: {elapsed}ms"
