"""
Dagster Definitions - Entry Point.

This module defines all Dagster objects (jobs, assets, resources, schedules, sensors)
that will be loaded by the Dagster system.
"""

from dagster import Definitions

from .chronos_jobs import (
    chronos_covariates_job,
    chronos_finetuned_job,
    chronos_zeroshot_job,
)
from .working_pipeline import complete_training_job

# Define all Dagster objects
defs = Definitions(
    jobs=[
        complete_training_job,
        chronos_zeroshot_job,
        chronos_finetuned_job,
        chronos_covariates_job,
    ],
)
