"""
Data loader for drift monitoring.

This module handles loading reference and production data for drift analysis,
including conversion between Polars and Pandas formats as required by Evidently.

Author: Arthur (MLOps/SRE Engineer)
Date: 2025-11-16
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import polars as pl

from src.monitoring.config import MonitoringConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for drift monitoring.

    This class handles loading reference data (training set sample) and
    production data (recent predictions) for drift analysis.

    Attributes
    ----------
    config : MonitoringConfig
        Monitoring configuration
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize data loader.

        Parameters
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config

    def load_reference_data(self) -> pd.DataFrame:
        """
        Load reference data (training set sample).

        Returns
        -------
        pd.DataFrame
            Reference data as Pandas DataFrame (required by Evidently)

        Raises
        ------
        FileNotFoundError
            If reference data file not found
        ValueError
            If reference data is empty or invalid
        """
        ref_path = Path(self.config.reference_data.path)

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference data not found: {ref_path}")

        logger.info(f"Loading reference data from {ref_path}")

        # Load with Polars first (efficient)
        df_polars = pl.read_parquet(ref_path)

        if df_polars.height == 0:
            raise ValueError("Reference data is empty")

        logger.info(
            f"Loaded reference data: {df_polars.height} rows, {df_polars.width} columns"
        )

        # Convert to Pandas (required by Evidently)
        df_pandas = df_polars.to_pandas()

        return df_pandas

    def load_production_data(
        self, production_data_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load production data (recent predictions).

        Parameters
        ----------
        production_data_path : Optional[Path], default=None
            Path to production data file. If None, will be determined from config.

        Returns
        -------
        pd.DataFrame
            Production data as Pandas DataFrame

        Raises
        ------
        FileNotFoundError
            If production data file not found
        ValueError
            If production data is empty or has insufficient samples
        """
        if production_data_path is None:
            raise ValueError(
                "Production data path must be provided. "
                "Automatic loading from API logs/database not yet implemented."
            )

        if not production_data_path.exists():
            raise FileNotFoundError(f"Production data not found: {production_data_path}")

        logger.info(f"Loading production data from {production_data_path}")

        # Load with Polars first (efficient)
        df_polars = pl.read_parquet(production_data_path)

        if df_polars.height == 0:
            raise ValueError("Production data is empty")

        if df_polars.height < self.config.production_data.min_samples:
            logger.warning(
                f"Production data has only {df_polars.height} samples, "
                f"which is less than minimum {self.config.production_data.min_samples}. "
                f"Drift analysis may be unreliable."
            )

        logger.info(
            f"Loaded production data: {df_polars.height} rows, {df_polars.width} columns"
        )

        # Convert to Pandas (required by Evidently)
        df_pandas = df_polars.to_pandas()

        return df_pandas

    def validate_schemas(
        self, reference_data: pd.DataFrame, production_data: pd.DataFrame
    ) -> Tuple[bool, list]:
        """
        Validate that reference and production data have compatible schemas.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Reference data
        production_data : pd.DataFrame
            Production data

        Returns
        -------
        Tuple[bool, list]
            (is_valid, list_of_errors)
        """
        errors = []

        # Check column compatibility
        ref_columns = set(reference_data.columns)
        prod_columns = set(production_data.columns)

        missing_in_prod = ref_columns - prod_columns
        extra_in_prod = prod_columns - ref_columns

        if missing_in_prod:
            errors.append(
                f"Columns missing in production data: {sorted(missing_in_prod)}"
            )

        if extra_in_prod:
            logger.warning(
                f"Extra columns in production data (will be ignored): {sorted(extra_in_prod)}"
            )

        # Check data types compatibility for common columns
        common_columns = ref_columns & prod_columns

        for col in common_columns:
            ref_dtype = reference_data[col].dtype
            prod_dtype = production_data[col].dtype

            # Allow numeric type conversions (int -> float, etc.)
            if pd.api.types.is_numeric_dtype(ref_dtype) and pd.api.types.is_numeric_dtype(
                prod_dtype
            ):
                continue

            # Require exact match for non-numeric types
            if ref_dtype != prod_dtype:
                errors.append(
                    f"Data type mismatch for column '{col}': "
                    f"reference={ref_dtype}, production={prod_dtype}"
                )

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.error(f"Schema validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")

        return is_valid, errors

    def align_schemas(
        self, reference_data: pd.DataFrame, production_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align production data schema to match reference data.

        This method ensures production data has the same columns as reference data,
        in the same order, with compatible data types.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Reference data (schema template)
        production_data : pd.DataFrame
            Production data to align

        Returns
        -------
        pd.DataFrame
            Production data with aligned schema
        """
        logger.info("Aligning production data schema to reference data")

        # Select only columns present in reference data
        aligned_data = production_data[reference_data.columns].copy()

        # Convert data types to match reference where possible
        for col in aligned_data.columns:
            ref_dtype = reference_data[col].dtype
            prod_dtype = aligned_data[col].dtype

            if ref_dtype != prod_dtype:
                try:
                    aligned_data[col] = aligned_data[col].astype(ref_dtype)
                    logger.debug(
                        f"Converted column '{col}' from {prod_dtype} to {ref_dtype}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not convert column '{col}' to {ref_dtype}: {e}"
                    )

        return aligned_data

    def load_and_validate(
        self, production_data_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both reference and production data with validation.

        This is a convenience method that loads both datasets and validates
        their schemas are compatible.

        Parameters
        ----------
        production_data_path : Optional[Path], default=None
            Path to production data file

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (reference_data, production_data) with aligned schemas

        Raises
        ------
        FileNotFoundError
            If data files not found
        ValueError
            If data is invalid or schemas incompatible
        """
        # Load data
        reference_data = self.load_reference_data()
        production_data = self.load_production_data(production_data_path)

        # Validate schemas
        is_valid, errors = self.validate_schemas(reference_data, production_data)

        if not is_valid:
            # Try to align schemas automatically
            logger.warning("Attempting automatic schema alignment")
            production_data = self.align_schemas(reference_data, production_data)

            # Re-validate
            is_valid, errors = self.validate_schemas(reference_data, production_data)

            if not is_valid:
                raise ValueError(
                    f"Schema validation failed after alignment: {'; '.join(errors)}"
                )

        logger.info("Data loading and validation completed successfully")

        return reference_data, production_data
