"""
Data repository abstractions (Dependency Inversion Principle).

Abstract interfaces for data access following SOLID principles.
High-level modules depend on these abstractions, not concrete implementations.
"""

from abc import ABC, abstractmethod

import polars as pl


class DataRepository(ABC):
    """
    Abstract repository for data access.

    High-level modules depend on this abstraction, not concrete implementations.
    This follows the Dependency Inversion Principle (DIP) from SOLID.

    Examples
    --------
    >>> from src.utils.data_repository import DataRepository, DuckDBRepository
    >>> from src.config.paths import DUCKDB_PATH
    >>>
    >>> # Use through abstraction
    >>> repo: DataRepository = DuckDBRepository(DUCKDB_PATH)
    >>> df = repo.query("SELECT * FROM steel_cleaned LIMIT 10")
    >>> ref_data = repo.get_reference_data()
    """

    @abstractmethod
    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute SQL query.

        Parameters
        ----------
        sql : str
            SQL query string

        Returns
        -------
        pl.DataFrame
            Query results as Polars DataFrame
        """
        pass

    @abstractmethod
    def get_reference_data(self, limit: int | None = None) -> pl.DataFrame:
        """
        Get reference dataset for analysis.

        Parameters
        ----------
        limit : int | None, default=None
            Maximum number of rows to return. If None, returns all data.

        Returns
        -------
        pl.DataFrame
            Reference dataset
        """
        pass

    @abstractmethod
    def get_table_schema(self, table_name: str) -> dict[str, str]:
        """
        Get schema information for table.

        Parameters
        ----------
        table_name : str
            Name of table

        Returns
        -------
        dict[str, str]
            Mapping of column names to data types
        """
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in database.

        Parameters
        ----------
        table_name : str
            Name of table to check

        Returns
        -------
        bool
            True if table exists, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass
