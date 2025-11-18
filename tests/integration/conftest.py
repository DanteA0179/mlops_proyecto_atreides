"""
Pytest configuration for integration tests.

Provides shared fixtures and configuration for integration testing.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """
    Configure pytest for integration tests.

    Parameters
    ----------
    config : Config
        Pytest config object
    """
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_db: mark test as requiring database")
    config.addinivalue_line("markers", "requires_mlflow: mark test as requiring MLflow")


@pytest.fixture(scope="session", autouse=True)
def setup_integration_environment():
    """
    Setup environment for integration tests.

    Configures paths and environment variables for testing.
    """
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"

    yield

    if "TESTING" in os.environ:
        del os.environ["TESTING"]
