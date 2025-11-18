"""
Unit tests for OllamaClient.

These tests use mocks to avoid actual API calls to Ollama.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.ollama_client import OllamaClient


class TestOllamaClient:
    """Unit tests for OllamaClient class."""

    def test_client_initialization_defaults(self):
        """Test client initialization with default parameters."""
        with patch.dict("os.environ", {}, clear=True):
            client = OllamaClient()
            assert client.host == "http://localhost:11434"
            assert client.model == "llama3.2:3b"
            assert client.timeout == 120
            assert client.temperature == 0.7
            assert client.max_tokens == 1000

    def test_client_initialization_env_vars(self):
        """Test client initialization with environment variables."""
        env_vars = {
            "OLLAMA_HOST": "http://custom-host:8000",
            "OLLAMA_MODEL": "custom-model",
            "OLLAMA_TIMEOUT": "60",
            "OLLAMA_TEMPERATURE": "0.5",
            "OLLAMA_MAX_TOKENS": "500"
        }

        with patch.dict("os.environ", env_vars):
            client = OllamaClient()
            assert client.host == "http://custom-host:8000"
            assert client.model == "custom-model"
            assert client.timeout == 60
            assert client.temperature == 0.5
            assert client.max_tokens == 500

    def test_client_initialization_custom_params(self):
        """Test client initialization with custom parameters."""
        client = OllamaClient(
            host="http://test:9999",
            model="test-model",
            timeout=30,
            temperature=0.9,
            max_tokens=2000
        )
        assert client.host == "http://test:9999"
        assert client.model == "test-model"
        assert client.timeout == 30
        assert client.temperature == 0.9
        assert client.max_tokens == 2000

    @patch("requests.post")
    def test_generate_success(self, mock_post):
        """Test successful text generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Energy consumption is the amount of energy used.",
            "done": True,
            "eval_count": 10
        }
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.generate("What is energy consumption?")

        assert result == "Energy consumption is the amount of energy used."
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_generate_empty_prompt(self, mock_post):
        """Test generation with empty prompt raises ValueError."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.generate("")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.generate("   ")

    @patch("requests.post")
    def test_generate_timeout(self, mock_post):
        """Test generation with timeout raises TimeoutError."""
        from requests.exceptions import Timeout

        mock_post.side_effect = Timeout("Request timed out")

        client = OllamaClient(timeout=1)

        with pytest.raises(TimeoutError, match="timed out"):
            client.generate("Test prompt")

    @patch("requests.post")
    def test_generate_request_exception(self, mock_post):
        """Test generation with request exception raises RuntimeError."""
        from requests.exceptions import RequestException

        mock_post.side_effect = RequestException("Network error")

        client = OllamaClient()

        with pytest.raises(RuntimeError, match="API request failed"):
            client.generate("Test prompt")

    @patch("requests.post")
    def test_chat_success(self, mock_post):
        """Test successful chat generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient()
        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat(messages)

        assert result == "Hello! How can I help you?"
        mock_post.assert_called_once()

    def test_chat_empty_messages(self):
        """Test chat with empty messages raises ValueError."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="non-empty list"):
            client.chat([])

        with pytest.raises(ValueError, match="non-empty list"):
            client.chat(None)

    def test_chat_invalid_message_format(self):
        """Test chat with invalid message format raises ValueError."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="role.*content"):
            client.chat([{"invalid": "format"}])

    @patch("requests.get")
    def test_list_models_success(self, mock_get):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b", "size": 2000000000},
                {"name": "llama3.2:7b", "size": 4000000000}
            ]
        }
        mock_get.return_value = mock_response

        client = OllamaClient()
        models = client.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "llama3.2:3b"
        assert models[1]["name"] == "llama3.2:7b"

    @patch("requests.get")
    def test_list_models_failure(self, mock_get):
        """Test model listing failure returns empty list."""
        from requests.exceptions import RequestException

        mock_get.side_effect = RequestException("Network error")

        client = OllamaClient()
        models = client.list_models()

        assert models == []

    @patch("requests.get")
    def test_is_model_available_true(self, mock_get):
        """Test model availability check when model exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}]
        }
        mock_get.return_value = mock_response

        client = OllamaClient(model="llama3.2:3b")
        assert client.is_model_available() is True

    @patch("requests.get")
    def test_is_model_available_false(self, mock_get):
        """Test model availability check when model does not exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}]
        }
        mock_get.return_value = mock_response

        client = OllamaClient(model="llama3.2:3b")
        assert client.is_model_available() is False

    @patch("requests.get")
    def test_health_check_success(self, mock_get):
        """Test health check when API is accessible."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = OllamaClient()
        assert client.health_check() is True

    @patch("requests.get")
    def test_health_check_failure(self, mock_get):
        """Test health check when API is not accessible."""
        from requests.exceptions import RequestException

        mock_get.side_effect = RequestException("Connection refused")

        client = OllamaClient()
        assert client.health_check() is False

    @patch("requests.get")
    def test_get_model_info_success(self, mock_get):
        """Test getting model info when model exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2:3b",
                    "size": 2000000000,
                    "digest": "abc123"
                }
            ]
        }
        mock_get.return_value = mock_response

        client = OllamaClient(model="llama3.2:3b")
        info = client.get_model_info()

        assert info is not None
        assert info["name"] == "llama3.2:3b"
        assert info["size"] == 2000000000

    @patch("requests.get")
    def test_get_model_info_not_found(self, mock_get):
        """Test getting model info when model does not exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}]
        }
        mock_get.return_value = mock_response

        client = OllamaClient(model="llama3.2:3b")
        info = client.get_model_info()

        assert info is None
