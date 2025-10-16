"""
DuckDB Data Loader

This module provides functions to load Parquet files into DuckDB for
interactive SQL-based exploration and analysis.

DuckDB advantages:
- 10-100x faster than Pandas for analytical queries
- SQL interface for familiar data exploration
- Reads Parquet directly without loading into memory
- Portable single-file database

Example Usage:
    >>> from src.data.load_to_duckdb import create_database, load_data, query_to_dataframe
    >>> 
    >>> # Create/connect to database
    >>> conn = create_database("data/steel.duckdb")
    >>> 
    >>> # Load data
    >>> load_data(conn)
    >>> 
    >>> # Query
    >>> df = query_to_dataframe(conn, "SELECT * FROM steel_cleaned LIMIT 10")
"""

import duckdb
import polars as pl
from pathlib import Path
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database(db_path: str = "data/steel.duckdb") -> duckdb.DuckDBPyConnection:
    """
    Creates or connects to a DuckDB database.
    
    Parameters
    ----------
    db_path : str, default='data/steel.duckdb'
        Path to the DuckDB database file.
        Use ':memory:' for in-memory database (testing).
        
    Returns
    -------
    duckdb.DuckDBPyConnection
        Connection to the DuckDB database
        
    Examples
    --------
    >>> conn = create_database("data/steel.duckdb")
    >>> conn = create_database(":memory:")  # For testing
    """
    try:
        # Ensure directory exists
        if db_path != ":memory:":
            db_path_obj = Path(db_path)
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        logger.info(f"✅ Connected to DuckDB: {db_path}")
        return conn
    
    except Exception as e:
        logger.error(f"❌ Failed to create database: {e}")
        raise


def load_parquet_to_table(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    table_name: str,
    replace: bool = True
) -> None:
    """
    Loads a Parquet file into a DuckDB table.
    
    DuckDB can read Parquet files directly without loading into memory first,
    making this operation very efficient.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    parquet_path : str
        Path to the Parquet file
    table_name : str
        Name of the table to create
    replace : bool, default=True
        If True, replaces existing table. If False, appends data.
        
    Raises
    ------
    FileNotFoundError
        If the Parquet file doesn't exist
    Exception
        If loading fails
        
    Examples
    --------
    >>> conn = create_database(":memory:")
    >>> load_parquet_to_table(conn, "data/processed/steel_cleaned.parquet", "steel_cleaned")
    """
    parquet_path_obj = Path(parquet_path)
    
    if not parquet_path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    try:
        # Drop table if exists and replace=True
        if replace:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table from Parquet
        # DuckDB reads Parquet directly - very efficient!
        conn.execute(f"""
            CREATE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        logger.info(f"✅ Loaded {row_count:,} rows into table '{table_name}'")
    
    except Exception as e:
        logger.error(f"❌ Failed to load Parquet to table: {e}")
        raise



def query_to_dataframe(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    output_format: str = "polars"
) -> Union[pl.DataFrame, 'pd.DataFrame']:
    """
    Executes a SQL query and returns results as a DataFrame.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    query : str
        SQL query to execute
    output_format : str, default='polars'
        Output format: 'polars' or 'pandas'
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        Query results as DataFrame
        
    Examples
    --------
    >>> conn = create_database("data/steel.duckdb")
    >>> df = query_to_dataframe(conn, "SELECT * FROM steel_cleaned LIMIT 10")
    >>> df_pandas = query_to_dataframe(conn, "SELECT * FROM steel_cleaned", output_format="pandas")
    """
    try:
        if output_format == "polars":
            # DuckDB → Arrow → Polars (zero-copy, very fast)
            df = conn.execute(query).pl()
        elif output_format == "pandas":
            df = conn.execute(query).df()
        else:
            raise ValueError(f"Invalid output_format: {output_format}. Use 'polars' or 'pandas'")
        
        logger.info(f"✅ Query executed: {len(df)} rows returned")
        return df
    
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise


def get_table_info(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    """
    Gets information about a table (schema, row count, size).
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    table_name : str
        Name of the table
        
    Returns
    -------
    dict
        Dictionary with table information:
        - row_count: Number of rows
        - columns: List of column names
        - schema: List of (column_name, data_type) tuples
        
    Examples
    --------
    >>> conn = create_database("data/steel.duckdb")
    >>> info = get_table_info(conn, "steel_cleaned")
    >>> print(f"Rows: {info['row_count']}")
    >>> print(f"Columns: {info['columns']}")
    """
    try:
        # Row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Schema
        schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        schema = [(row[0], row[1]) for row in schema_result]
        columns = [row[0] for row in schema_result]
        
        info = {
            'table_name': table_name,
            'row_count': row_count,
            'column_count': len(columns),
            'columns': columns,
            'schema': schema
        }
        
        return info
    
    except Exception as e:
        logger.error(f"❌ Failed to get table info: {e}")
        raise


def load_data(
    conn: duckdb.DuckDBPyConnection,
    cleaned_parquet: str = "data/processed/steel_cleaned.parquet"
) -> None:
    """
    Loads all project data into DuckDB.
    
    This is the main function to populate the database with all tables.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    cleaned_parquet : str
        Path to cleaned Parquet file
        
    Examples
    --------
    >>> conn = create_database("data/steel.duckdb")
    >>> load_data(conn)
    """
    logger.info("📊 Loading data into DuckDB...")
    
    # Load cleaned data
    if Path(cleaned_parquet).exists():
        load_parquet_to_table(conn, cleaned_parquet, "steel_cleaned")
    else:
        logger.warning(f"⚠️ File not found: {cleaned_parquet}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DATABASE SUMMARY")
    logger.info("="*60)
    
    tables = conn.execute("SHOW TABLES").fetchall()
    for table in tables:
        table_name = table[0]
        info = get_table_info(conn, table_name)
        logger.info(f"\nTable: {table_name}")
        logger.info(f"  Rows: {info['row_count']:,}")
        logger.info(f"  Columns: {info['column_count']}")
    
    logger.info("\n" + "="*60)


def example_queries(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Demonstrates example queries for data exploration.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    """
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)
    
    # Query 1: Statistics by Load_Type
    print("\n1️⃣ Statistics by Load_Type:")
    print("-" * 60)
    df1 = query_to_dataframe(conn, """
        SELECT 
            Load_Type,
            COUNT(*) as count,
            ROUND(AVG(Usage_kWh), 2) as avg_usage,
            ROUND(MIN(Usage_kWh), 2) as min_usage,
            ROUND(MAX(Usage_kWh), 2) as max_usage,
            ROUND(STDDEV(Usage_kWh), 2) as std_usage
        FROM steel_cleaned
        GROUP BY Load_Type
        ORDER BY avg_usage DESC
    """)
    print(df1)
    
    # Query 2: Consumption by Day of Week
    print("\n2️⃣ Average Consumption by Day of Week:")
    print("-" * 60)
    df2 = query_to_dataframe(conn, """
        SELECT 
            Day_of_week,
            ROUND(AVG(Usage_kWh), 2) as avg_usage,
            ROUND(AVG(CO2), 4) as avg_co2
        FROM steel_cleaned
        GROUP BY Day_of_week
        ORDER BY 
            CASE Day_of_week
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
            END
    """)
    print(df2)
    
    # Query 3: Top 10 Peak Consumption
    print("\n3️⃣ Top 10 Peak Consumption Events:")
    print("-" * 60)
    df3 = query_to_dataframe(conn, """
        SELECT 
            ROUND(Usage_kWh, 2) as Usage_kWh,
            Load_Type,
            Day_of_week,
            NSM,
            ROUND(CO2, 4) as CO2
        FROM steel_cleaned
        ORDER BY Usage_kWh DESC
        LIMIT 10
    """)
    print(df3)
    
    # Query 4: Hourly Analysis
    print("\n4️⃣ Average Consumption by Hour:")
    print("-" * 60)
    df4 = query_to_dataframe(conn, """
        SELECT 
            FLOOR(NSM / 3600) as hour,
            COUNT(*) as records,
            ROUND(AVG(Usage_kWh), 2) as avg_usage,
            ROUND(AVG(Lagging_Current_Power_Factor), 3) as avg_power_factor
        FROM steel_cleaned
        GROUP BY hour
        ORDER BY hour
        LIMIT 24
    """)
    print(df4)
    
    print("\n" + "="*60)


def main():
    """
    Main function to create database and load data.
    
    Run this script directly to set up the DuckDB database:
        python src/data/load_to_duckdb.py
    """
    # Create database
    conn = create_database("data/steel.duckdb")
    
    # Load data
    load_data(conn)
    
    # Run example queries
    example_queries(conn)
    
    # Close connection
    conn.close()
    logger.info("\n✅ Database setup complete!")


if __name__ == "__main__":
    main()
