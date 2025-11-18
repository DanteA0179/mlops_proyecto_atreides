"""
Integration tests for data pipeline components.

Tests integration between data loading, cleaning, and DuckDB storage.
"""

import tempfile
from pathlib import Path

import duckdb
import polars as pl
import pytest


@pytest.fixture(scope="module")
def sample_data():
    """
    Create sample data for testing.

    Returns
    -------
    pl.DataFrame
        Sample steel industry data
    """
    return pl.DataFrame(
        {
            "Usage_kWh": [45.2, 50.1, 38.9, 55.3, 42.0],
            "Lagging_Current_Reactive.Power_kVarh": [23.5, 25.1, 20.3, 28.2, 22.1],
            "Leading_Current_Reactive_Power_kVarh": [12.3, 13.5, 10.8, 14.2, 11.5],
            "CO2(tCO2)": [0.05, 0.06, 0.04, 0.07, 0.05],
            "Lagging_Current_Power_Factor": [0.85, 0.82, 0.88, 0.80, 0.86],
            "Leading_Current_Power_Factor": [0.92, 0.90, 0.95, 0.89, 0.93],
            "NSM": [36000, 43200, 50400, 57600, 64800],
            "day": [15, 15, 15, 15, 15],
            "Load_Type": ["Medium", "Light", "Medium", "Maximum", "Light"],
        }
    )


@pytest.fixture
def temp_csv_file(sample_data):
    """
    Create temporary CSV file.

    Parameters
    ----------
    sample_data : pl.DataFrame
        Sample data

    Yields
    ------
    Path
        Path to temporary CSV file
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.write_csv(f.name)
        yield Path(f.name)

    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_db_path():
    """
    Create temporary database path.

    Yields
    ------
    Path
        Path to temporary database
    """
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = Path(f.name)

    yield db_path

    db_path.unlink(missing_ok=True)


class TestDataLoadingIntegration:
    """Test integration between data loading components"""

    def test_csv_to_duckdb_integration(self, temp_csv_file: Path, temp_db_path: Path):
        """Test loading CSV into DuckDB"""
        from src.data.load_to_duckdb import load_csv_to_duckdb

        load_csv_to_duckdb(
            csv_path=str(temp_csv_file), db_path=str(temp_db_path), table_name="test_steel"
        )

        conn = duckdb.connect(str(temp_db_path))
        result = conn.execute("SELECT COUNT(*) FROM test_steel").fetchone()

        assert result[0] == 5
        conn.close()

    def test_load_clean_store_workflow(self, temp_csv_file: Path, temp_db_path: Path):
        """Test complete workflow: load -> clean -> store"""
        # Load
        df = pl.read_csv(temp_csv_file)
        assert len(df) == 5

        # Clean
        from src.utils.data_cleaning import handle_null_values, remove_duplicates

        df_cleaned = handle_null_values(df)
        df_cleaned = remove_duplicates(df_cleaned)

        # Store in DuckDB
        conn = duckdb.connect(str(temp_db_path))
        conn.execute("CREATE TABLE steel_cleaned AS SELECT * FROM df_cleaned")

        # Verify
        result = conn.execute("SELECT COUNT(*) FROM steel_cleaned").fetchone()
        assert result[0] == len(df_cleaned)

        conn.close()


class TestDataQualityIntegration:
    """Test integration between data quality components"""

    def test_quality_checks_with_duckdb(self, sample_data: pl.DataFrame, temp_db_path: Path):
        """Test data quality checks on DuckDB data"""
        from src.utils.data_quality import analyze_nulls

        conn = duckdb.connect(str(temp_db_path))
        conn.execute("CREATE TABLE test_data AS SELECT * FROM sample_data")

        df = conn.execute("SELECT * FROM test_data").pl()

        null_report = analyze_nulls(df)

        assert "column" in null_report.columns
        assert "null_count" in null_report.columns

        conn.close()

    def test_outlier_detection_with_storage(self, sample_data: pl.DataFrame, temp_db_path: Path):
        """Test outlier detection and storage"""
        from src.utils.outlier_detection import detect_outliers_iqr

        outliers = detect_outliers_iqr(sample_data, "Usage_kWh")

        conn = duckdb.connect(str(temp_db_path))
        conn.execute("CREATE TABLE original_data AS SELECT * FROM sample_data")

        if len(outliers) > 0:
            conn.execute("CREATE TABLE outliers AS SELECT * FROM outliers")

        original_count = conn.execute("SELECT COUNT(*) FROM original_data").fetchone()[0]
        assert original_count == 5

        conn.close()


class TestDataTransformationIntegration:
    """Test integration between transformation components"""

    def test_load_transform_validate_workflow(self, temp_csv_file: Path):
        """Test workflow: load -> transform -> validate"""
        from src.utils.data_cleaning import convert_data_types

        df = pl.read_csv(temp_csv_file)

        df_transformed = convert_data_types(df)

        assert df_transformed["NSM"].dtype in [pl.Int64, pl.Int32]
        assert len(df_transformed) == len(df)

    def test_multiple_transformations_chain(self, sample_data: pl.DataFrame):
        """Test chaining multiple transformations"""
        from src.utils.data_cleaning import (
            convert_data_types,
            handle_null_values,
            remove_duplicates,
        )

        df = sample_data
        df = handle_null_values(df)
        df = remove_duplicates(df)
        df = convert_data_types(df)

        assert len(df) == 5
        assert df["NSM"].dtype in [pl.Int64, pl.Int32]


class TestDuckDBUtilsIntegration:
    """Test integration of DuckDB utility functions"""

    def test_utils_with_real_database(self, sample_data: pl.DataFrame, temp_db_path: Path):
        """Test DuckDB utils with actual database"""
        from src.utils.duckdb_utils import setup_database

        conn = duckdb.connect(str(temp_db_path))
        conn.execute("CREATE TABLE steel_data AS SELECT * FROM sample_data")
        conn.close()

        conn = setup_database(str(temp_db_path))

        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

        conn.close()

    def test_query_functions_integration(self, sample_data: pl.DataFrame, temp_db_path: Path):
        """Test query functions work with database"""
        conn = duckdb.connect(str(temp_db_path))
        conn.execute("CREATE TABLE test_table AS SELECT * FROM sample_data")
        conn.close()

        conn = duckdb.connect(str(temp_db_path))
        result = conn.execute("SELECT * FROM test_table LIMIT 3").pl()

        assert len(result) == 3
        assert "Usage_kWh" in result.columns

        conn.close()


class TestDataPersistenceIntegration:
    """Test data persistence across operations"""

    def test_save_load_consistency(self, sample_data: pl.DataFrame, temp_db_path: Path):
        """Test data consistency after save and load"""
        conn = duckdb.connect(str(temp_db_path))

        conn.execute("CREATE TABLE persistent_data AS SELECT * FROM sample_data")

        original_sum = sample_data["Usage_kWh"].sum()

        conn.close()

        conn = duckdb.connect(str(temp_db_path))
        loaded_df = conn.execute("SELECT * FROM persistent_data").pl()

        loaded_sum = loaded_df["Usage_kWh"].sum()

        assert abs(original_sum - loaded_sum) < 0.01

        conn.close()

    def test_multiple_table_operations(self, sample_data: pl.DataFrame, temp_db_path: Path):
        """Test operations across multiple tables"""
        conn = duckdb.connect(str(temp_db_path))

        conn.execute("CREATE TABLE raw_data AS SELECT * FROM sample_data")

        conn.execute(
            """
            CREATE TABLE cleaned_data AS
            SELECT * FROM raw_data
            WHERE "Usage_kWh" > 40
        """
        )

        raw_count = conn.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        cleaned_count = conn.execute("SELECT COUNT(*) FROM cleaned_data").fetchone()[0]

        assert raw_count == 5
        assert cleaned_count <= raw_count

        conn.close()


class TestErrorHandlingIntegration:
    """Test error handling across components"""

    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        from src.utils.data_cleaning import handle_null_values

        invalid_df = pl.DataFrame({"col1": [1, 2, None, 4], "col2": [None, 2, 3, 4]})

        result = handle_null_values(invalid_df)

        assert result is not None

    def test_database_error_handling(self, temp_db_path: Path):
        """Test database error handling"""
        conn = duckdb.connect(str(temp_db_path))

        with pytest.raises(Exception):
            conn.execute("SELECT * FROM nonexistent_table")

        conn.close()
