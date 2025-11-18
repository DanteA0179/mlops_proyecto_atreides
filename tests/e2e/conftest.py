"""
Pytest configuration for e2e tests.

Provides shared fixtures and configuration for end-to-end testing.
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
    Configure pytest for e2e tests.

    Parameters
    ----------
    config : Config
        Pytest config object
    """
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as API test (requires running API)")
    config.addinivalue_line("markers", "pipeline: mark test as pipeline test")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Setup test environment for e2e tests.

    Ensures project structure and environment is properly configured.
    """
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"

    yield

    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
