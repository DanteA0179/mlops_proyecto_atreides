"""
DuckDB Utilities.

Reusable functions for working with DuckDB in notebooks and scripts.
Following clean code and reusability principles from AGENTS.md standards.
"""

import logging
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl

from src.config.paths import DATA_PROCESSED_DIR, DUCKDB_PATH

logger = logging.getLogger(__name__)


def get_connection(db_path: Path | str = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Get connection to DuckDB database.

    Simple wrapper for creating connections consistently across the project.

    Parameters
    ----------
    db_path : Path | str, default=DUCKDB_PATH
        Path to DuckDB database file

    Returns
    -------
    duckdb.DuckDBPyConnection
        Active database connection

    Examples
    --------
    >>> from src.utils.duckdb_utils import get_connection
    >>> conn = get_connection()
    >>> # Use connection...
    >>> conn.close()

    Using custom path:
    >>> conn = get_connection("custom/path/db.duckdb")
    >>> conn.close()
    """
    from src.data.load_to_duckdb import create_database

    return create_database(str(db_path))


def quick_query(
    query: str,
    db_path: Path | str = DUCKDB_PATH,
    output_format: Literal["polars", "pandas"] = "polars",
) -> pl.DataFrame:
    """
    Execute quick SQL query without manual connection management.

    Opens connection, executes query, and closes connection automatically.
    Ideal for single queries in notebooks.

    Parameters
    ----------
    query : str
        SQL query to execute
    db_path : Path | str, default=DUCKDB_PATH
        Path to database file
    output_format : Literal["polars", "pandas"], default="polars"
        Output format, either "polars" or "pandas"

    Returns
    -------
    pl.DataFrame
        Query results as DataFrame

    Examples
    --------
    >>> from src.utils.duckdb_utils import quick_query
    >>> df = quick_query("SELECT * FROM steel_cleaned LIMIT 10")
    >>> print(df)

    With custom path:
    >>> df = quick_query(
    ...     "SELECT * FROM data WHERE value > 100",
    ...     db_path="custom/db.duckdb"
    ... )
    """
    from src.data.load_to_duckdb import query_to_dataframe

    conn = get_connection(db_path)
    try:
        df = query_to_dataframe(conn, query, output_format)
        return df
    finally:
        conn.close()


def get_stats_by_column(
    table_name: str,
    group_by_column: str,
    agg_column: str,
    db_path: Path | str = DUCKDB_PATH,
) -> pl.DataFrame:
    """
    Get aggregated statistics grouped by column.

    Convenience function for quick exploratory analysis.

    Parameters
    ----------
    table_name : str
        Name of database table
    group_by_column : str
        Column to group by
    agg_column : str
        Column to aggregate
    db_path : Path | str, default=DUCKDB_PATH
        Path to database file

    Returns
    -------
    pl.DataFrame
        DataFrame with statistics (count, avg, min, max, std)

    Examples
    --------
    >>> from src.utils.duckdb_utils import get_stats_by_column
    >>> stats = get_stats_by_column('steel_cleaned', 'Load_Type', 'Usage_kWh')
    >>> print(stats)
    shape: (3, 6)
    ┌───────────┬───────┬──────┬───────┬───────┬──────┐
    │ Load_Type │ count │ avg  │ min   │ max   │ std  │
    │ ---       │ ---   │ ---  │ ---   │ ---   │ ---  │
    │ str       │ i64   │ f64  │ f64   │ f64   │ f64  │
    ╞═══════════╪═══════╪══════╪═══════╪═══════╪══════╡
    │ Light     │ 12000 │ 23.4 │ 10.2  │ 45.6  │ 5.23 │
    │ Medium    │ 18000 │ 47.9 │ 30.1  │ 78.9  │ 8.45 │
    │ Maximum   │ 5040  │ 95.7 │ 70.4  │ 120.3 │ 12.3 │
    └───────────┴───────┴──────┴───────┴───────┴──────┘

    With special characters in column names:
    >>> stats = get_stats_by_column('steel_cleaned', 'Load_Type', 'CO2(tCO2)')
    """
    group_col_quoted = _quote_column_if_needed(group_by_column)
    agg_col_quoted = _quote_column_if_needed(agg_column)

    safe_group_name = group_by_column.replace(".", "_").replace("(", "_").replace(")", "_")

    query = f"""
        SELECT
            {group_col_quoted} as {safe_group_name},
            COUNT(*) as count,
            ROUND(AVG({agg_col_quoted}), 2) as avg,
            ROUND(MIN({agg_col_quoted}), 2) as min,
            ROUND(MAX({agg_col_quoted}), 2) as max,
            ROUND(STDDEV({agg_col_quoted}), 2) as std
        FROM {table_name}
        GROUP BY {group_col_quoted}
        ORDER BY avg DESC
    """
    return quick_query(query, db_path)


def get_temporal_stats(
    table_name: str,
    time_column: str,
    agg_column: str,
    time_unit: str = "hour",
    db_path: str = "data/steel.duckdb",
) -> pl.DataFrame:
    """
    Obtiene estadísticas temporales agregadas.

    Parameters
    ----------
    table_name : str
        Nombre de la tabla
    time_column : str
        Columna temporal (ej: 'NSM')
    agg_column : str
        Columna para agregar
    time_unit : str, default='hour'
        Unidad temporal: 'hour', 'day', 'minute'
    db_path : str
        Ruta a la base de datos

    Returns
    -------
    pl.DataFrame
        DataFrame con estadísticas temporales

    Examples
    --------
    >>> from src.utils.duckdb_utils import get_temporal_stats
    >>> hourly = get_temporal_stats('steel_cleaned', 'NSM', 'Usage_kWh', 'hour')
    >>> print(hourly)
    """
    # Agregar comillas a columnas con caracteres especiales
    time_col_quoted = _quote_column_if_needed(time_column)
    agg_col_quoted = _quote_column_if_needed(agg_column)

    # Convertir NSM a unidad temporal
    if time_unit == "hour":
        time_expr = f"FLOOR({time_col_quoted} / 3600)"
    elif time_unit == "minute":
        time_expr = f"FLOOR({time_col_quoted} / 60)"
    elif time_unit == "day":
        time_expr = f"FLOOR({time_col_quoted} / 86400)"
    else:
        raise ValueError(f"Invalid time_unit: {time_unit}")

    # Limpiar nombre de columna para alias
    agg_col_clean = agg_column.replace(".", "_").replace("(", "_").replace(")", "_")

    query = f"""
        SELECT
            {time_expr} as {time_unit},
            COUNT(*) as records,
            ROUND(AVG({agg_col_quoted}), 2) as avg_{agg_col_clean},
            ROUND(STDDEV({agg_col_quoted}), 2) as std_{agg_col_clean}
        FROM {table_name}
        GROUP BY {time_unit}
        ORDER BY {time_unit}
    """
    return quick_query(query, db_path)


def _quote_column_if_needed(col: str) -> str:
    """
    Agrega comillas dobles a nombres de columnas con caracteres especiales.

    Parameters
    ----------
    col : str
        Nombre de columna

    Returns
    -------
    str
        Nombre de columna con comillas si es necesario
    """
    # Si ya tiene comillas, retornar tal cual
    if col.startswith('"') and col.endswith('"'):
        return col

    # Si tiene paréntesis u otros caracteres especiales, agregar comillas
    special_chars = ["(", ")", " ", "-", "."]
    if any(char in col for char in special_chars):
        return f'"{col}"'

    return col


def get_top_n(
    table_name: str,
    order_by_column: str,
    n: int = 10,
    ascending: bool = False,
    columns: list[str] | None = None,
    db_path: str = "data/steel.duckdb",
) -> pl.DataFrame:
    """
    Obtiene los top N registros ordenados por una columna.

    Parameters
    ----------
    table_name : str
        Nombre de la tabla
    order_by_column : str
        Columna para ordenar
    n : int, default=10
        Número de registros a retornar
    ascending : bool, default=False
        Si True, orden ascendente. Si False, descendente.
    columns : list of str, optional
        Columnas a seleccionar. Si None, selecciona todas.
        Nota: Los nombres con caracteres especiales (paréntesis, espacios)
        se manejan automáticamente.
    db_path : str
        Ruta a la base de datos

    Returns
    -------
    pl.DataFrame
        Top N registros

    Examples
    --------
    >>> from src.utils.duckdb_utils import get_top_n
    >>> top_10 = get_top_n('steel_cleaned', 'Usage_kWh', n=10)
    >>>
    >>> # Con columnas que tienen caracteres especiales
    >>> top_10 = get_top_n(
    ...     'steel_cleaned',
    ...     'Usage_kWh',
    ...     n=10,
    ...     columns=['Usage_kWh', 'CO2(tCO2)']  # Se maneja automáticamente
    ... )
    """
    if columns:
        # Agregar comillas a columnas con caracteres especiales
        quoted_cols = [_quote_column_if_needed(col) for col in columns]
        cols = ", ".join(quoted_cols)
    else:
        cols = "*"

    order = "ASC" if ascending else "DESC"

    query = f"""
        SELECT {cols}
        FROM {table_name}
        ORDER BY {order_by_column} {order}
        LIMIT {n}
    """
    return quick_query(query, db_path)


def get_correlation(
    table_name: str, column1: str, column2: str, db_path: str = "data/steel.duckdb"
) -> float:
    """
    Calcula la correlación entre dos columnas.

    Parameters
    ----------
    table_name : str
        Nombre de la tabla
    column1 : str
        Primera columna
    column2 : str
        Segunda columna
    db_path : str
        Ruta a la base de datos

    Returns
    -------
    float
        Coeficiente de correlación de Pearson

    Examples
    --------
    >>> from src.utils.duckdb_utils import get_correlation
    >>> corr = get_correlation('steel_cleaned', 'Usage_kWh', 'CO2(tCO2)')
    >>> print(f"Correlation: {corr:.4f}")
    """
    # Agregar comillas a columnas con caracteres especiales
    col1_quoted = _quote_column_if_needed(column1)
    col2_quoted = _quote_column_if_needed(column2)

    query = f"""
        SELECT CORR({col1_quoted}, {col2_quoted}) as correlation
        FROM {table_name}
    """
    df = quick_query(query, db_path)
    return float(df["correlation"][0])


def get_weekend_vs_weekday_stats(
    table_name: str, agg_column: str, db_path: str = "data/steel.duckdb"
) -> pl.DataFrame:
    """
    Compara estadísticas entre fin de semana y días laborales.

    Parameters
    ----------
    table_name : str
        Nombre de la tabla
    agg_column : str
        Columna para agregar
    db_path : str
        Ruta a la base de datos

    Returns
    -------
    pl.DataFrame
        Estadísticas comparativas

    Examples
    --------
    >>> from src.utils.duckdb_utils import get_weekend_vs_weekday_stats
    >>> stats = get_weekend_vs_weekday_stats('steel_cleaned', 'Usage_kWh')
    >>> print(stats)
    """
    # Agregar comillas a columnas con caracteres especiales
    agg_col_quoted = _quote_column_if_needed(agg_column)

    query = f"""
        SELECT
            WeekStatus,
            COUNT(*) as records,
            ROUND(AVG({agg_col_quoted}), 2) as avg,
            ROUND(STDDEV({agg_col_quoted}), 2) as std,
            ROUND(MIN({agg_col_quoted}), 2) as min,
            ROUND(MAX({agg_col_quoted}), 2) as max
        FROM {table_name}
        GROUP BY WeekStatus
    """
    return quick_query(query, db_path)


def execute_custom_query(
    query: str,
    db_path: Path | str = DUCKDB_PATH,
    output_format: Literal["polars", "pandas"] = "polars",
) -> pl.DataFrame:
    """
    Execute custom SQL query.

    Alias for quick_query with more semantic clarity.

    Parameters
    ----------
    query : str
        SQL query to execute
    db_path : Path | str, default=DUCKDB_PATH
        Path to database file
    output_format : Literal["polars", "pandas"], default="polars"
        Output format

    Returns
    -------
    pl.DataFrame
        Query results

    Examples
    --------
    >>> from src.utils.duckdb_utils import execute_custom_query
    >>> df = execute_custom_query('''
    ...     SELECT Load_Type, AVG(Usage_kWh) as avg_usage
    ...     FROM steel_cleaned
    ...     GROUP BY Load_Type
    ... ''')
    """
    return quick_query(query, db_path, output_format)


def setup_database(
    db_path: Path | str = DUCKDB_PATH,
    parquet_path: Path | str = DATA_PROCESSED_DIR / "steel_cleaned.parquet",
    force_reload: bool = False,
) -> duckdb.DuckDBPyConnection:
    """
    Configura la base de datos y carga datos si es necesario.

    Esta función es ideal para notebooks - verifica si los datos ya están
    cargados y solo los carga si es necesario.

    Parameters
    ----------
    db_path : str
        Ruta a la base de datos DuckDB
    parquet_path : str
        Ruta al archivo Parquet con datos limpios
    force_reload : bool, default=False
        Si True, recarga los datos incluso si ya existen

    Returns
    -------
    duckdb.DuckDBPyConnection
        Conexión a la base de datos con datos cargados

    Examples
    --------
    >>> from src.utils.duckdb_utils import setup_database
    >>> conn = setup_database("../../data/steel.duckdb", "../../data/processed/steel_cleaned.parquet")
    >>> # Ahora puedes usar conn para queries
    """
    from src.data.load_to_duckdb import load_parquet_to_table

    conn = get_connection(db_path)

    # Verificar si la tabla ya existe
    tables = conn.execute("SHOW TABLES").fetchall()
    table_exists = any("steel_cleaned" in str(table) for table in tables)

    if not table_exists or force_reload:
        # Verificar que el archivo Parquet existe
        if not Path(parquet_path).exists():
            logger.warning(f"⚠️ Archivo Parquet no encontrado: {parquet_path}")
            logger.info("💡 Tip: Ejecuta primero 'poetry run python src/data/clean_data.py'")
        else:
            logger.info(f"📊 Cargando datos desde {parquet_path}...")
            load_parquet_to_table(conn, parquet_path, "steel_cleaned", replace=force_reload)
            logger.info("✅ Datos cargados exitosamente")
    else:
        logger.info("✅ Tabla 'steel_cleaned' ya existe en la base de datos")

    return conn


# Context manager para manejo automático de conexiones
class DuckDBConnection:
    """
    Context manager para manejar conexiones DuckDB automáticamente.

    Asegura que la conexión se cierre correctamente incluso si hay errores.

    Examples
    --------
    >>> from src.utils.duckdb_utils import DuckDBConnection
    >>>
    >>> with DuckDBConnection() as conn:
    ...     df = conn.execute("SELECT * FROM steel_cleaned LIMIT 10").pl()
    ...     print(df)
    """

    def __init__(self, db_path: Path | str = DUCKDB_PATH):
        """Initialize context manager with database path."""
        self.db_path = db_path
        self.conn: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> duckdb.DuckDBPyConnection:
        """Context manager entry."""
        self.conn = get_connection(self.db_path)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures connection is closed."""
        if self.conn:
            self.conn.close()
