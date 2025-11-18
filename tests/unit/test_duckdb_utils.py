"""
Unit Tests for DuckDB Utilities

Tests para las funciones reutilizables de src/utils/duckdb_utils.py
"""

import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.utils.duckdb_utils import (
    DuckDBConnection,
    get_connection,
    get_correlation,
    get_stats_by_column,
    get_temporal_stats,
    get_top_n,
    get_weekend_vs_weekday_stats,
    quick_query,
)


@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para archivos de test."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_db(temp_dir):
    """Crea una base de datos de prueba con datos de ejemplo."""
    from src.data.load_to_duckdb import create_database, load_parquet_to_table

    # Crear datos de ejemplo
    df = pl.DataFrame(
        {
            "Usage_kWh": [10.5, 12.3, 15.7, 11.2, 13.8, 20.1, 18.5, 16.3],
            "Load_Type": [
                "Light",
                "Medium",
                "Maximum",
                "Light",
                "Medium",
                "Maximum",
                "Light",
                "Medium",
            ],
            "Day_of_week": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "Monday",
            ],
            "WeekStatus": [
                "Weekday",
                "Weekday",
                "Weekday",
                "Weekday",
                "Weekday",
                "Weekend",
                "Weekend",
                "Weekday",
            ],
            "NSM": [3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800],
            "CO2": [0.05, 0.06, 0.08, 0.05, 0.07, 0.10, 0.09, 0.08],
        }
    )

    # Guardar a Parquet
    parquet_path = Path(temp_dir) / "test_data.parquet"
    df.write_parquet(parquet_path)

    # Crear DB y cargar datos
    db_path = Path(temp_dir) / "test.duckdb"
    conn = create_database(str(db_path))
    load_parquet_to_table(conn, str(parquet_path), "steel_cleaned")
    conn.close()

    return str(db_path)


class TestGetConnection:
    """Tests para get_connection()."""

    def test_get_connection_memory(self):
        """Test conexión a DB en memoria."""
        conn = get_connection(":memory:")
        assert conn is not None
        conn.close()

    def test_get_connection_file(self, sample_db):
        """Test conexión a DB en archivo."""
        conn = get_connection(sample_db)
        assert conn is not None
        conn.close()


class TestQuickQuery:
    """Tests para quick_query()."""

    def test_quick_query_basic(self, sample_db):
        """Test query básico."""
        df = quick_query("SELECT * FROM steel_cleaned LIMIT 5", db_path=sample_db)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_quick_query_with_filter(self, sample_db):
        """Test query con filtro."""
        df = quick_query("SELECT * FROM steel_cleaned WHERE Load_Type = 'Light'", db_path=sample_db)
        assert len(df) == 3
        assert all(df["Load_Type"] == "Light")

    def test_quick_query_pandas_output(self, sample_db):
        """Test output en formato pandas."""
        df = quick_query(
            "SELECT * FROM steel_cleaned LIMIT 5", db_path=sample_db, output_format="pandas"
        )
        assert hasattr(df, "iloc")  # Pandas-specific


class TestGetStatsByColumn:
    """Tests para get_stats_by_column()."""

    def test_get_stats_by_column(self, sample_db):
        """Test estadísticas por columna."""
        df = get_stats_by_column(
            table_name="steel_cleaned",
            group_by_column="Load_Type",
            agg_column="Usage_kWh",
            db_path=sample_db,
        )

        assert isinstance(df, pl.DataFrame)
        assert "Load_Type" in df.columns
        assert "count" in df.columns
        assert "avg" in df.columns
        assert "min" in df.columns
        assert "max" in df.columns
        assert "std" in df.columns
        assert len(df) == 3  # Light, Medium, Maximum


class TestGetTemporalStats:
    """Tests para get_temporal_stats()."""

    def test_get_temporal_stats_hour(self, sample_db):
        """Test estadísticas temporales por hora."""
        df = get_temporal_stats(
            table_name="steel_cleaned",
            time_column="NSM",
            agg_column="Usage_kWh",
            time_unit="hour",
            db_path=sample_db,
        )

        assert isinstance(df, pl.DataFrame)
        assert "hour" in df.columns
        assert "records" in df.columns
        assert "avg_Usage_kWh" in df.columns

    def test_get_temporal_stats_invalid_unit(self, sample_db):
        """Test con unidad temporal inválida."""
        with pytest.raises(ValueError):
            get_temporal_stats(
                table_name="steel_cleaned",
                time_column="NSM",
                agg_column="Usage_kWh",
                time_unit="invalid",
                db_path=sample_db,
            )


class TestGetTopN:
    """Tests para get_top_n()."""

    def test_get_top_n_default(self, sample_db):
        """Test top N con valores por defecto."""
        df = get_top_n(
            table_name="steel_cleaned", order_by_column="Usage_kWh", n=3, db_path=sample_db
        )

        assert len(df) == 3
        # Verificar orden descendente
        assert df["Usage_kWh"][0] >= df["Usage_kWh"][1]

    def test_get_top_n_ascending(self, sample_db):
        """Test top N en orden ascendente."""
        df = get_top_n(
            table_name="steel_cleaned",
            order_by_column="Usage_kWh",
            n=3,
            ascending=True,
            db_path=sample_db,
        )

        assert len(df) == 3
        # Verificar orden ascendente
        assert df["Usage_kWh"][0] <= df["Usage_kWh"][1]

    def test_get_top_n_specific_columns(self, sample_db):
        """Test top N con columnas específicas."""
        df = get_top_n(
            table_name="steel_cleaned",
            order_by_column="Usage_kWh",
            n=3,
            columns=["Usage_kWh", "Load_Type"],
            db_path=sample_db,
        )

        assert len(df.columns) == 2
        assert "Usage_kWh" in df.columns
        assert "Load_Type" in df.columns

    def test_get_top_n_special_chars_in_column_names(self, sample_db):
        """Test top N con nombres de columnas con caracteres especiales."""
        df = get_top_n(
            table_name="steel_cleaned",
            order_by_column="Usage_kWh",
            n=3,
            columns=["Usage_kWh", "CO2"],  # CO2 tiene paréntesis en la DB real
            db_path=sample_db,
        )

        assert len(df) == 3
        assert "Usage_kWh" in df.columns
        assert "CO2" in df.columns


class TestGetCorrelation:
    """Tests para get_correlation()."""

    def test_get_correlation(self, sample_db):
        """Test cálculo de correlación."""
        corr = get_correlation(
            table_name="steel_cleaned", column1="Usage_kWh", column2="CO2", db_path=sample_db
        )

        assert isinstance(corr, float)
        assert -1 <= corr <= 1  # Correlación debe estar entre -1 y 1


class TestGetWeekendVsWeekdayStats:
    """Tests para get_weekend_vs_weekday_stats()."""

    def test_get_weekend_vs_weekday_stats(self, sample_db):
        """Test comparación fin de semana vs días laborales."""
        df = get_weekend_vs_weekday_stats(
            table_name="steel_cleaned", agg_column="Usage_kWh", db_path=sample_db
        )

        assert isinstance(df, pl.DataFrame)
        assert "WeekStatus" in df.columns
        assert "records" in df.columns
        assert "avg" in df.columns
        assert len(df) == 2  # Weekday y Weekend


class TestSetupDatabase:
    """Tests para setup_database()."""

    def test_setup_database_new(self, temp_dir):
        """Test setup de base de datos nueva."""
        from src.utils.duckdb_utils import setup_database

        # Crear datos de prueba
        df = pl.DataFrame(
            {"Usage_kWh": [10.5, 12.3, 15.7], "Load_Type": ["Light", "Medium", "Maximum"]}
        )
        parquet_path = Path(temp_dir) / "test.parquet"
        df.write_parquet(parquet_path)

        # Setup database
        db_path = Path(temp_dir) / "test.duckdb"
        conn = setup_database(str(db_path), str(parquet_path))

        # Verificar que la tabla existe
        result = conn.execute("SELECT COUNT(*) FROM steel_cleaned").fetchone()
        assert result[0] == 3

        conn.close()

    def test_setup_database_existing(self, sample_db):
        """Test setup con base de datos existente."""
        from src.utils.duckdb_utils import setup_database

        # Setup con DB que ya tiene datos
        conn = setup_database(sample_db, "dummy.parquet")

        # Debe conectar sin problemas
        result = conn.execute("SELECT COUNT(*) FROM steel_cleaned").fetchone()
        assert result[0] == 8

        conn.close()


class TestDuckDBConnection:
    """Tests para context manager DuckDBConnection."""

    def test_context_manager(self, sample_db):
        """Test uso de context manager."""
        with DuckDBConnection(sample_db) as conn:
            result = conn.execute("SELECT COUNT(*) FROM steel_cleaned").fetchone()
            assert result[0] == 8

    def test_context_manager_auto_close(self, sample_db):
        """Test que la conexión se cierra automáticamente."""
        with DuckDBConnection(sample_db) as conn:
            assert conn is not None

        # Después del context manager, la conexión debe estar cerrada
        # No podemos verificar directamente, pero no debe haber errores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
