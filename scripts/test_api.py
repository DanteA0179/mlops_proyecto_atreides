#!/usr/bin/env python
"""
Quick test script for FastAPI health endpoint.
Run this to verify the API works before Docker build.
"""

import os
import sys
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    """Test the /health endpoint"""
    client = TestClient(app)
    response = client.get("/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()

    print("✅ Health endpoint test PASSED")


def test_root_endpoint():
    """Test the root endpoint"""
    client = TestClient(app)
    response = client.get("/")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    assert "message" in response.json()

    print("✅ Root endpoint test PASSED")


if __name__ == "__main__":
    print("Testing FastAPI endpoints...")
    print("-" * 50)

    try:
        test_health_endpoint()
        print()
        test_root_endpoint()
        print("-" * 50)
        print("✅ All tests PASSED!")
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        sys.exit(1)
