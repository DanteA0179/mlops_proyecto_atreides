"""
Centralized path configuration.

All file paths and directories used across the project.
"""

from pathlib import Path
from typing import Final

# Project root
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATA_RAW_DIR: Final[Path] = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
DATA_PRODUCTION_DIR: Final[Path] = DATA_DIR / "production"

# Database
DUCKDB_PATH: Final[Path] = DATA_DIR / "steel.duckdb"

# Models
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
MODELS_ENSEMBLES_DIR: Final[Path] = MODELS_DIR / "ensembles"
MODELS_ONNX_DIR: Final[Path] = MODELS_DIR / "onnx"

# Reports
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
REPORTS_MONITORING_DIR: Final[Path] = REPORTS_DIR / "monitoring"

# Logs
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
