"""
Dagster Pipeline for ML Training.

This package contains the Dagster implementation of the training pipeline,
replacing the previous Prefect implementation.
"""

from .definitions import defs

__all__ = ["defs"]
