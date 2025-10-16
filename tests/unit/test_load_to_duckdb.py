"""
Unit Tests for DuckDB Data Loader

Tests the functionality of loading Parquet files into DuckDB.
"""

import pytest
import duckdb
import polars as pl
from pathlib import Path
import tempfile
import shutil

from src.data.load_to_duckdb import (
    create_database,
    load_parquet_to_table,
    query_to_dataframe,
    get_table_info,
    load_data
)


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_parquet(temp_dir):
    """Creates a sample Parquet file for testing."""
    # Create sample data
    df = pl.DataFrame({
        'Usage_kWh': [10.5, 12.3, 15.7, 11.2, 13.8],
        'Load_Type': ['Light', 'Medium', 'Maximum', 'Light', 'Medium'],
        'Day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'NSM': [3600, 7200, 10800, 14400, 18000],
        'CO2': [0.05, 0.06, 0.08, 0.05, 0.07]
    })
    
    # Save to Parquet
    parquet_path = Path(temp_dir) / "test_data.parquet"
    df.write_parquet(parquet_path)
    
    return str(parquet_path)


class TestCreateDatabase:
    """Tests for create_database function."""
    
    def test_create_memory_database(self):
        """Test creating an in-memory database."""
        conn = create_database(":memory:")
        assert conn is not None
        assert isinstance(conn, duckdb.DuckDBPyConnection)
        conn.close()
    
    def test_create_file_database(self, temp_dir):
        """Test creating a file-based database."""
        db_path = Path(temp_dir) / "test.duckdb"
        conn = create_database(str(db_path))
        
        assert conn is not None
        assert db_path.exists()
        conn.close()
    
    def test_create_database_creates_directory(self, temp_dir):
        """Test that create_database creates parent directories."""
        db_path = Path(temp_dir) / "subdir" / "test.duckdb"
        conn = create_database(str(db_path))
        
        assert db_path.parent.exists()
        assert db_path.exists()
        conn.close()


class TestLoadParquetToTable:
    """Tests for load_parquet_to_table function."""
    
    def test_load_parquet_success(self, sample_parquet):
        """Test successful loading of Parquet file."""
        conn = create_database(":memory:")
        load_parquet_to_table(conn, sample_parquet, "test_table")
        
        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        assert ('test_table',) in tables
        
        # Verify row count
        row_count = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        assert row_count == 5
        
        conn.close()
    
    def test_load_parquet_file_not_found(self):
        """Test loading non-existent Parquet file."""
        conn = create_database(":memory:")
        
        with pytest.raises(FileNotFoundError):
            load_parquet_to_table(conn, "nonexistent.parquet", "test_table")
        
        conn.close()
    
    def test_load_parquet_replace_table(self, sample_parquet):
        """Test replacing existing table."""
        conn = create_database(":memory:")
        
        # Load first time
        load_parquet_to_table(conn, sample_parquet, "test_table")
        row_count_1 = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        
        # Load again (replace)
        load_parquet_to_table(conn, sample_parquet, "test_table", replace=True)
        row_count_2 = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        
        assert row_count_1 == row_count_2 == 5
        
        conn.close()


class TestQueryToDataFrame:
    """Tests for query_to_dataframe function."""
    
    def test_query_to_polars(self, sample_parquet):
        """Test querying to Polars DataFrame."""
        conn = create_database(":memory:")
        load_parquet_to_table(conn, sample_parquet, "test_table")
        
        df = query_to_dataframe(conn, "SELECT * FROM test_table", output_format="polars")
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert 'Usage_kWh' in df.columns
        
        conn.close()
    
    def test_query_to_pandas(self, sample_parquet):
        """Test querying to Pandas DataFrame."""
        conn = create_database(":memory:")
        load_parquet_to_table(conn, sample_parquet, "test_table")
        
        df = query_to_dataframe(conn, "SELECT * FROM test_table", output_format="pandas")
        
        # Check it's a pandas DataFrame
        assert hasattr(df, 'iloc')  # Pandas-specific attribute
        assert len(df) == 5
        
        conn.close()
    
    def test_query_with_filter(self, sample_parquet):
        """Test query with WHERE clause."""
        conn = create_database(":memory:")
        load_parquet_to_table(conn, sample_parquet, "test_table")
        
        df = query_to_dataframe(conn, "SELECT * FROM test_table WHERE Load_Type = 'Light'")
        
        assert len(df) == 2
        assert all(df['Load_Type'] == 'Light')
        
        conn.close()
    
    def test_query_invalid_format(self, sample_parquet):
        """Test query with invalid output format."""
        conn = create_database(":memory:")
        load_parquet_to_table(conn, sample_parquet, "test_table")
        
        with pytest.raises(ValueError):
            query_to_dataframe(conn, "SELECT * FROM test_table", output_format="invalid")
        
        conn.close()


class TestGetTableInfo:
    """Tests for get_table_info function."""
    
    def test_get_table_info(self, sample_parquet):
        """Test getting table information."""
        conn = create_database(":memory:")
        load_parquet_to_table(conn, sample_parquet, "test_table")
        
        info = get_table_info(conn, "test_table")
        
        assert info['table_name'] == 'test_table'
        assert info['row_count'] == 5
        assert info['column_count'] == 5
        assert 'Usage_kWh' in info['columns']
        assert len(info['schema']) == 5
        
        conn.close()
    
    def test_get_table_info_nonexistent(self):
        """Test getting info for non-existent table."""
        conn = create_database(":memory:")
        
        with pytest.raises(Exception):
            get_table_info(conn, "nonexistent_table")
        
        conn.close()


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_with_existing_file(self, sample_parquet, temp_dir):
        """Test load_data with existing Parquet file."""
        conn = create_database(":memory:")
        
        # Use sample_parquet as cleaned data
        load_data(conn, cleaned_parquet=sample_parquet)
        
        # Verify table was created
        tables = conn.execute("SHOW TABLES").fetchall()
        assert ('steel_cleaned',) in tables
        
        conn.close()
    
    def test_load_data_with_missing_file(self):
        """Test load_data with missing Parquet file (should warn, not fail)."""
        conn = create_database(":memory:")
        
        # Should not raise exception, just log warning
        load_data(conn, cleaned_parquet="nonexistent.parquet")
        
        conn.close()


# Integration test
class TestIntegration:
    """Integration tests for the full workflow."""
    
    def test_full_workflow(self, sample_parquet, temp_dir):
        """Test complete workflow: create DB → load data → query."""
        # Create database
        db_path = Path(temp_dir) / "test.duckdb"
        conn = create_database(str(db_path))
        
        # Load data
        load_parquet_to_table(conn, sample_parquet, "steel_cleaned")
        
        # Query data
        df = query_to_dataframe(conn, """
            SELECT Load_Type, AVG(Usage_kWh) as avg_usage
            FROM steel_cleaned
            GROUP BY Load_Type
        """)
        
        # Verify results
        assert len(df) == 3  # Light, Medium, Maximum
        assert 'Load_Type' in df.columns
        assert 'avg_usage' in df.columns
        
        # Get table info
        info = get_table_info(conn, "steel_cleaned")
        assert info['row_count'] == 5
        
        conn.close()
        
        # Verify database file exists
        assert db_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
