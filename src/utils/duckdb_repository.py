"""
DuckDB implementation of DataRepository.

Concrete implementation of data access for DuckDB following DIP.
"""

import logging
from pathlib import Path

import duckdb
import polars as pl

from src.config.paths import DUCKDB_PATH
from src.utils.data_repository import DataRepository

logger = logging.getLogger(__name__)


class DuckDBRepository(DataRepository):
    """
    DuckDB implementation of DataRepository.

    Concrete implementation providing DuckDB-specific data access.
    Can be easily swapped with other implementations (PostgreSQL, SQLite, etc.)
    without changing high-level code.

    Parameters
    ----------
    db_path : Path | str, default=DUCKDB_PATH
        Path to DuckDB database file

    Examples
    --------
    >>> from src.utils.duckdb_repository import DuckDBRepository
    >>> from src.config.paths import DUCKDB_PATH
    >>>
    >>> repo = DuckDBRepository(DUCKDB_PATH)
    >>> df = repo.query("SELECT * FROM steel_cleaned LIMIT 10")
    >>> schema = repo.get_table_schema("steel_cleaned")
    >>> repo.close()

    Using as context manager:
    >>> with DuckDBRepository(DUCKDB_PATH) as repo:
    ...     df = repo.query("SELECT * FROM steel_cleaned LIMIT 5")
    ...     print(df)
    """

    def __init__(self, db_path: Path | str = DUCKDB_PATH):
        """Initialize repository with database path."""
        self.db_path = Path(db_path)
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._connect()

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self.conn = duckdb.connect(str(self.db_path))
            logger.debug(f"Connected to DuckDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute SQL query and return results as Polars DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to execute

        Returns
        -------
        pl.DataFrame
            Query results

        Raises
        ------
        RuntimeError
            If connection is not established
        duckdb.Error
            If query execution fails
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            return self.conn.execute(sql).pl()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def get_reference_data(self, limit: int | None = None) -> pl.DataFrame:
        """
        Get reference dataset from steel_cleaned table.

        Parameters
        ----------
        limit : int | None, default=None
            Maximum rows to return. If None, returns all data.

        Returns
        -------
        pl.DataFrame
            Reference dataset
        """
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        sql = f"SELECT * FROM steel_cleaned {limit_clause}"
        return self.query(sql)

    def get_table_schema(self, table_name: str) -> dict[str, str]:
        """
        Get schema information for specified table.

        Parameters
        ----------
        table_name : str
            Name of table

        Returns
        -------
        dict[str, str]
            Mapping of column names to data types

        Examples
        --------
        >>> repo = DuckDBRepository()
        >>> schema = repo.get_table_schema("steel_cleaned")
        >>> print(schema)
        {
            'Usage_kWh': 'DOUBLE',
            'Lagging_Current_Reactive_Power_kVarh': 'DOUBLE',
            'CO2(tCO2)': 'DOUBLE',
            ...
        }
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            return {row[0]: row[1] for row in result}
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            raise

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
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            tables = self.conn.execute("SHOW TABLES").fetchall()
            return any(table_name in str(table) for table in tables)
        except Exception as e:
            logger.error(f"Failed to check table existence: {e}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.debug("Database connection closed")

    def __enter__(self) -> "DuckDBRepository":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()
