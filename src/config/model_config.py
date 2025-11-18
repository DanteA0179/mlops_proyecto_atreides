"""
Model configuration and hyperparameters.
"""

from dataclasses import dataclass
from typing import Any, Final


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    name: str
    type: str
    version: str
    path: str
    hyperparameters: dict[str, Any]


# Default model configurations
DEFAULT_MODEL_TYPE: Final[str] = "stacking_ensemble"
DEFAULT_MODEL_VERSION: Final[str] = "v3"

LIGHTGBM_CONFIG = ModelConfig(
    name="lightgbm",
    type="gradient_boosting",
    version="v3",
    path="models/lightgbm/lightgbm_v3.pkl",
    hyperparameters={
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 31,
    },
)

XGBOOST_CONFIG = ModelConfig(
    name="xgboost",
    type="gradient_boosting",
    version="v2",
    path="models/xgboost/xgboost_v2.pkl",
    hyperparameters={
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 6,
    },
)
