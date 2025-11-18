"""
Business and domain constants.

All magic numbers and business rules centralized here.
"""

from typing import Final, Literal

# Dataset constants
SECONDS_PER_DAY: Final[int] = 86400
MINUTES_PER_HOUR: Final[int] = 60
HOURS_PER_DAY: Final[int] = 24

# Feature thresholds
OUTLIER_Z_SCORE_THRESHOLD: Final[float] = 3.0
DRIFT_DETECTION_THRESHOLD: Final[float] = 0.7
CORRELATION_THRESHOLD: Final[float] = 0.8

# Model thresholds
MAX_BATCH_SIZE: Final[int] = 1000
MODEL_TIMEOUT_SECONDS: Final[int] = 30
MIN_CONFIDENCE_SCORE: Final[float] = 0.5

# Load types
LoadType = Literal["Light", "Medium", "Maximum"]
VALID_LOAD_TYPES: Final[tuple[str, ...]] = ("Light", "Medium", "Maximum")

# Weekday status
WeekStatus = Literal["Weekday", "Weekend"]
VALID_WEEK_STATUS: Final[tuple[str, ...]] = ("Weekday", "Weekend")

# Power factor bounds
MIN_POWER_FACTOR: Final[float] = 0.0
MAX_POWER_FACTOR: Final[float] = 1.0
