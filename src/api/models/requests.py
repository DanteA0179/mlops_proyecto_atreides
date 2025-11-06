"""
Pydantic Request Models for Energy Optimization API.

This module contains all request models used by the API endpoints,
with comprehensive validation using Pydantic.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    """
    Request model for single energy consumption prediction.

    All features validated against physical constraints and dataset ranges.

    Attributes
    ----------
    lagging_reactive_power : float
        Potencia reactiva en atraso (kVarh), must be >= 0
    leading_reactive_power : float
        Potencia reactiva en adelanto (kVarh), must be >= 0
    co2 : float
        Emisiones de CO2 (tCO2), must be >= 0
    lagging_power_factor : float
        Factor de potencia en atraso (0-1)
    leading_power_factor : float
        Factor de potencia en adelanto (0-1)
    nsm : int
        Segundos desde medianoche (0-86400)
    day_of_week : int
        Día de la semana (0=Monday, 6=Sunday)
    load_type : Literal["Light", "Medium", "Maximum"]
        Tipo de carga industrial

    Examples
    --------
    >>> request = PredictionRequest(
    ...     lagging_reactive_power=23.45,
    ...     leading_reactive_power=12.30,
    ...     co2=0.05,
    ...     lagging_power_factor=0.85,
    ...     leading_power_factor=0.92,
    ...     nsm=36000,
    ...     day_of_week=1,
    ...     load_type="Medium"
    ... )
    """

    lagging_reactive_power: float = Field(
        ...,
        ge=0.0,
        description="Potencia reactiva en atraso (kVarh)",
        examples=[23.45],
    )
    leading_reactive_power: float = Field(
        ...,
        ge=0.0,
        description="Potencia reactiva en adelanto (kVarh)",
        examples=[12.30],
    )
    co2: float = Field(
        ..., ge=0.0, description="Emisiones de CO2 (tCO2)", examples=[0.05]
    )
    lagging_power_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factor de potencia en atraso (0-1)",
        examples=[0.85],
    )
    leading_power_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factor de potencia en adelanto (0-1)",
        examples=[0.92],
    )
    nsm: int = Field(
        ...,
        ge=0,
        le=86400,
        description="Segundos desde medianoche (0-86400)",
        examples=[36000],
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Día de la semana (0=Lunes, 6=Domingo)",
        examples=[1],
    )
    load_type: Literal["Light", "Medium", "Maximum"] = Field(
        ..., description="Tipo de carga industrial", examples=["Medium"]
    )

    @field_validator("load_type")
    @classmethod
    def validate_load_type(cls, v: str) -> str:
        """
        Validate load_type is one of allowed values.

        Parameters
        ----------
        v : str
            Load type value to validate

        Returns
        -------
        str
            Validated load type

        Raises
        ------
        ValueError
            If load_type is not in valid values
        """
        valid_types = ["Light", "Medium", "Maximum"]
        if v not in valid_types:
            raise ValueError(f"load_type must be one of {valid_types}, got {v}")
        return v

    @model_validator(mode="after")
    def validate_power_factors(self) -> "PredictionRequest":
        """
        Validate power factor consistency.

        Returns
        -------
        PredictionRequest
            Self after validation

        Raises
        ------
        ValueError
            If power factors are invalid
        """
        if self.lagging_power_factor > 1.0:
            raise ValueError("lagging_power_factor cannot exceed 1.0")
        if self.leading_power_factor > 1.0:
            raise ValueError("leading_power_factor cannot exceed 1.0")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "lagging_reactive_power": 23.45,
                "leading_reactive_power": 12.30,
                "co2": 0.05,
                "lagging_power_factor": 0.85,
                "leading_power_factor": 0.92,
                "nsm": 36000,
                "day_of_week": 1,
                "load_type": "Medium",
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch energy consumption predictions.

    Accepts list of prediction requests with max limit of 1000.

    Attributes
    ----------
    predictions : List[PredictionRequest]
        List of prediction requests (max 1000)

    Examples
    --------
    >>> batch_request = BatchPredictionRequest(
    ...     predictions=[
    ...         PredictionRequest(...),
    ...         PredictionRequest(...),
    ...     ]
    ... )
    """

    predictions: List[PredictionRequest] = Field(
        ...,
        max_length=1000,
        description="List of prediction requests (max 1000)",
        examples=[
            [
                {
                    "lagging_reactive_power": 23.45,
                    "leading_reactive_power": 12.30,
                    "co2": 0.05,
                    "lagging_power_factor": 0.85,
                    "leading_power_factor": 0.92,
                    "nsm": 36000,
                    "day_of_week": 1,
                    "load_type": "Medium",
                }
            ]
        ],
    )

    @field_validator("predictions")
    @classmethod
    def validate_batch_not_empty(cls, v: List[PredictionRequest]) -> List[PredictionRequest]:
        """
        Ensure batch is not empty and within limits.

        Parameters
        ----------
        v : List[PredictionRequest]
            List of predictions to validate

        Returns
        -------
        List[PredictionRequest]
            Validated predictions list

        Raises
        ------
        ValueError
            If batch is empty or exceeds 1000 predictions
        """
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch cannot exceed 1000 predictions")
        return v
