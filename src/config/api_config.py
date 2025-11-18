"""
API configuration settings.
"""

from typing import Final

# API settings
API_TITLE: Final[str] = "Energy Optimization API"
API_VERSION: Final[str] = "1.0.0"
API_DESCRIPTION: Final[str] = "API for energy consumption prediction in steel industry"

# Server settings
API_HOST: Final[str] = "0.0.0.0"
API_PORT: Final[int] = 8000

# Rate limiting
MAX_REQUESTS_PER_MINUTE: Final[int] = 60
MAX_BATCH_PREDICTIONS: Final[int] = 1000

# Timeouts
REQUEST_TIMEOUT_SECONDS: Final[int] = 30
MODEL_LOAD_TIMEOUT_SECONDS: Final[int] = 60

# CORS
ALLOWED_ORIGINS: Final[list[str]] = [
    "http://localhost:3000",
    "http://localhost:8501",
    "https://energy-copilot.streamlit.app",
]
