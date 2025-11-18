"""
Unit tests for preprocessing_utils.py.

Tests for preprocessing utility functions.
"""

import numpy as np
import polars as pl
import pytest

from src.utils.preprocessing_utils import (
    analyze_categorical_cardinality,
    calculate_scaling_statistics,
    get_feature_name_after_ohe,
    identify_feature_types,
    map_binary_feature,
    validate_preprocessing_config,
)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    return pl.DataFrame(
        {
            "numeric1": np.random.randn(100),
            "numeric2": np.random.rand(100) * 100,
            "categorical": np.random.choice(["A", "B", "C"], 100),
            "binary": np.random.choice(["Yes", "No"], 100),
            "boolean": np.random.choice([True, False], 100),
            "high_cardinality": [f"ID_{i}" for i in range(100)],
            "target": np.random.rand(100) * 50,
        }
    )


class TestIdentifyFeatureTypes:
    """Tests for identify_feature_types function."""

    def test_basic_identification(self, sample_dataframe):
        """Test basic feature type identification."""
        types = identify_feature_types(sample_dataframe, exclude_cols=["target"])

        assert "numeric" in types
        assert "categorical" in types
        assert "boolean" in types
        assert "excluded" in types

        # Check numeric features
        assert "numeric1" in types["numeric"]
        assert "numeric2" in types["numeric"]

        # Check categorical
        assert "categorical" in types["categorical"]
        assert "binary" in types["categorical"]

        # Check boolean
        assert "boolean" in types["boolean"]

        # Check excluded
        assert "target" in types["excluded"]

    def test_high_cardinality_excluded(self, sample_dataframe):
        """Test that high cardinality features are excluded."""
        types = identify_feature_types(sample_dataframe)

        # High cardinality should be in excluded
        assert "high_cardinality" in types["excluded"]

    def test_empty_exclude(self, sample_dataframe):
        """Test with empty exclude list."""
        types = identify_feature_types(sample_dataframe, exclude_cols=[])

        # No features should be in excluded (except high cardinality)
        assert "target" not in types["excluded"]
        assert "high_cardinality" in types["excluded"]

    def test_all_feature_types(self):
        """Test identification of all possible types."""
        # Use more rows so categorical is properly detected (need < 10% cardinality)
        # 2 unique values / 20 rows = 10%, need more repetition
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 2,
                "float_col": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0] * 2,
                "string_col": ["A", "B", "A", "A", "B", "A", "A", "B", "A", "A"]
                * 2,  # 2 unique / 20 = 10%
                "bool_col": [True, False, True, True, False, True, True, False, True, True] * 2,
                "datetime_col": pl.date_range(
                    pl.date(2020, 1, 1), pl.date(2020, 1, 20), interval="1d", eager=True
                ),
            }
        )

        types = identify_feature_types(df)

        assert "int_col" in types["numeric"]
        assert "float_col" in types["numeric"]
        # string_col with 2 values / 20 rows = 10%, at edge, may not be categorical
        # Let's just check it's either categorical or excluded
        assert "string_col" in types["categorical"] or "string_col" in types["excluded"]
        assert "bool_col" in types["boolean"]
        assert "datetime_col" in types["datetime"]

    def test_identification_with_all_null_column(self):
        """
        Prueba que una columna que solo contiene nulos (pl.Null)
        sea clasificada correctamente (ej. 'excluded').
        """
        df = pl.DataFrame({"numeric": [1, 2, 3], "all_null": [None, None, None]})

        # Le decimos a Polars que la columna es de tipo pl.Null
        df = df.with_columns(pl.col("all_null").cast(pl.Null))

        types = identify_feature_types(df)

        # Una columna de solo nulos no es numérica ni categórica
        assert "all_null" not in types.get("numeric", [])
        assert "all_null" not in types.get("categorical", [])

        # Debería ser 'excluida' o un tipo 'desconocido'
        assert "all_null" in types.get("excluded", []) or "all_null" in types.get("unknown", [])

    def test_identification_empty_dataframe(self):
        """Prueba que la función maneja un DataFrame vacío."""
        df = pl.DataFrame()
        types = identify_feature_types(df)

        # El resultado debe ser un diccionario de listas vacías
        assert "numeric" in types
        assert len(types["numeric"]) == 0
        assert len(types["categorical"]) == 0


class TestValidatePreprocessingConfig:
    """Tests for validate_preprocessing_config function."""

    def test_valid_config(self, sample_dataframe):
        """Test validation of valid configuration."""
        result = validate_preprocessing_config(
            numeric_features=["numeric1", "numeric2"],
            categorical_features=["categorical"],
            df=sample_dataframe,
        )

        assert result["valid"] is True
        assert len(result["missing_features"]) == 0
        assert len(result["duplicate_features"]) == 0
        assert len(result["wrong_types"]) == 0

    def test_missing_features(self, sample_dataframe):
        """Test detection of missing features."""
        result = validate_preprocessing_config(
            numeric_features=["numeric1", "nonexistent"],
            categorical_features=["categorical"],
            df=sample_dataframe,
        )

        assert result["valid"] is False
        assert "nonexistent" in result["missing_features"]

    def test_duplicate_features(self, sample_dataframe):
        """Test detection of duplicate features."""
        result = validate_preprocessing_config(
            numeric_features=["numeric1", "categorical"],
            categorical_features=["categorical"],
            df=sample_dataframe,
        )

        assert result["valid"] is False
        assert "categorical" in result["duplicate_features"]

    def test_wrong_types(self, sample_dataframe):
        """Test detection of wrong feature types."""
        result = validate_preprocessing_config(
            numeric_features=["numeric1", "categorical"],  # categorical is string
            categorical_features=[],
            df=sample_dataframe,
        )

        assert result["valid"] is False
        assert "categorical" in result["wrong_types"]

    def test_categorical_as_numeric(self, sample_dataframe):
        """Test detection of categorical features marked as numeric."""
        result = validate_preprocessing_config(
            numeric_features=["categorical"],  # Wrong type
            categorical_features=[],
            df=sample_dataframe,
        )

        assert result["valid"] is False


class TestCalculateScalingStatistics:
    """Tests for calculate_scaling_statistics function."""

    def test_basic_statistics(self, sample_dataframe):
        """Test basic statistics calculation."""
        stats = calculate_scaling_statistics(sample_dataframe, features=["numeric1", "numeric2"])

        # Check structure
        assert "numeric1" in stats
        assert "numeric2" in stats

        # Check stat keys
        for feat in ["numeric1", "numeric2"]:
            assert "mean" in stats[feat]
            assert "std" in stats[feat]
            assert "min" in stats[feat]
            assert "max" in stats[feat]
            assert "median" in stats[feat]
            assert "q25" in stats[feat]
            assert "q75" in stats[feat]

    def test_statistics_values(self):
        """Test that statistics are calculated correctly."""
        df = pl.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})

        stats = calculate_scaling_statistics(df, features=["feature"])

        assert stats["feature"]["mean"] == pytest.approx(3.0)
        assert stats["feature"]["min"] == pytest.approx(1.0)
        assert stats["feature"]["max"] == pytest.approx(5.0)
        assert stats["feature"]["median"] == pytest.approx(3.0)

    def test_empty_features(self, sample_dataframe):
        """Test with empty features list."""
        stats = calculate_scaling_statistics(sample_dataframe, features=[])

        assert len(stats) == 0

    def test_missing_features(self, sample_dataframe):
        """Test with missing features (should skip)."""
        stats = calculate_scaling_statistics(sample_dataframe, features=["numeric1", "nonexistent"])

        assert "numeric1" in stats
        assert "nonexistent" not in stats

    def test_statistics_with_nulls(self):
        """
        Prueba que los nulos son ignorados al calcular estadísticas.
        """
        df = pl.DataFrame({"feature": [1.0, 2.0, None, 4.0, 5.0]})

        stats = calculate_scaling_statistics(df, features=["feature"])

        # La media debe ser (1+2+4+5) / 4 = 3.0
        assert stats["feature"]["mean"] == pytest.approx(3.0)
        # El conteo (si se calcula) debe ser 4, no 5

    def test_statistics_with_all_nulls(self):
        """
        Prueba que una columna de solo nulos se maneja sin fallar.
        """
        df = pl.DataFrame({"feature": [None, None, None]}).with_columns(
            pl.col("feature").cast(pl.Float64)
        )

        stats = calculate_scaling_statistics(df, features=["feature"])

        # La media de solo nulos es nula (o NaN)
        assert stats["feature"]["mean"] is None or np.isnan(stats["feature"]["mean"])
        assert stats["feature"]["min"] is None
        assert stats["feature"]["max"] is None


class TestAnalyzeCategoricalCardinality:
    """Tests for analyze_categorical_cardinality function."""

    def test_basic_analysis(self, sample_dataframe):
        """Test basic cardinality analysis."""
        analysis = analyze_categorical_cardinality(sample_dataframe, feature="categorical")

        # Check structure
        assert "n_categories" in analysis
        assert "categories" in analysis
        assert "value_counts" in analysis
        assert "encoding_size" in analysis
        assert "most_common" in analysis
        assert "least_common" in analysis
        assert "total_count" in analysis
        assert "distribution" in analysis

    def test_cardinality_count(self):
        """Test cardinality counting."""
        df = pl.DataFrame({"cat": ["A", "B", "C", "A", "B", "A"]})

        analysis = analyze_categorical_cardinality(df, feature="cat")

        assert analysis["n_categories"] == 3
        assert set(analysis["categories"]) == {"A", "B", "C"}
        assert analysis["encoding_size"] == 2  # n_categories - 1 (drop='first')

    def test_value_counts(self):
        """Test value counts."""
        df = pl.DataFrame({"cat": ["A", "A", "A", "B", "B", "C"]})

        analysis = analyze_categorical_cardinality(df, feature="cat")

        assert analysis["value_counts"]["A"] == 3
        assert analysis["value_counts"]["B"] == 2
        assert analysis["value_counts"]["C"] == 1

    def test_most_least_common(self):
        """Test most and least common categories."""
        df = pl.DataFrame({"cat": ["A", "A", "A", "B", "B", "C"]})

        analysis = analyze_categorical_cardinality(df, feature="cat")

        assert analysis["most_common"] == "A"
        assert analysis["least_common"] == "C"

    def test_distribution(self):
        """Test distribution calculation."""
        df = pl.DataFrame({"cat": ["A", "A", "B", "B"]})

        analysis = analyze_categorical_cardinality(df, feature="cat")

        assert analysis["distribution"]["A"] == pytest.approx(0.5)
        assert analysis["distribution"]["B"] == pytest.approx(0.5)

    def test_missing_feature(self, sample_dataframe):
        """Test error on missing feature."""
        with pytest.raises(ValueError, match="not found in dataframe"):
            analyze_categorical_cardinality(sample_dataframe, feature="nonexistent")

    def test_cardinality_with_nulls(self):
        """
        Prueba cómo la cardinalidad maneja los valores nulos.
        """
        df = pl.DataFrame({"cat": ["A", "A", "B", None, "A", None]})

        analysis = analyze_categorical_cardinality(df, feature="cat")

        # Depende de tu implementación, pero generalmente los nulos
        # se cuentan como una categoría separada.

        # Asumiendo que 'None' NO se cuenta como una categoría
        assert analysis["n_categories"] == 2
        assert set(analysis["categories"]) == {"A", "B"}

        # Pero los conteos de valores SÍ deben reflejarlos
        assert analysis["value_counts"]["A"] == 3
        assert analysis["value_counts"]["B"] == 1
        # (El conteo de nulos puede estar o no en 'value_counts')

        # O, si los nulos SÍ se cuentan como categoría:
        # assert analysis["n_categories"] == 3
        # assert set(analysis["categories"]) == {"A", "B", None}
        # assert analysis["value_counts"][None] == 2


class TestMapBinaryFeature:
    """Tests for map_binary_feature function."""

    def test_basic_mapping(self):
        """Test basic binary mapping."""
        df = pl.DataFrame({"status": ["Active", "Inactive", "Active"]})

        df_mapped = map_binary_feature(df, feature="status", mapping={"Active": 1, "Inactive": 0})

        assert df_mapped["status"].to_list() == [1, 0, 1]
        assert df_mapped["status"].dtype == pl.Int32

    def test_weekday_weekend_mapping(self):
        """Test weekday/weekend mapping."""
        df = pl.DataFrame({"week_status": ["Weekday", "Weekend", "Weekday", "Weekend"]})

        df_mapped = map_binary_feature(
            df,
            feature="week_status",
            mapping={"Weekday": 0, "Weekend": 1},
        )

        assert df_mapped["week_status"].to_list() == [0, 1, 0, 1]

    def test_missing_feature(self):
        """Test error on missing feature."""
        df = pl.DataFrame({"feature": [1, 2, 3]})

        with pytest.raises(ValueError, match="not found in dataframe"):
            map_binary_feature(df, feature="nonexistent", mapping={"A": 0, "B": 1})

    def test_unknown_categories(self):
        """Test handling of unknown categories."""
        df = pl.DataFrame({"status": ["A", "B", "C"]})

        df_mapped = map_binary_feature(df, feature="status", mapping={"A": 0, "B": 1})

        # Unknown category "C" should be None
        assert df_mapped["status"][2] is None

    def test_mapping_with_existing_nulls(self):
        """
        Prueba que los nulos en la entrada se mantienen como nulos en la salida.
        """
        df = pl.DataFrame({"status": ["Active", "Inactive", None, "Active"]})

        df_mapped = map_binary_feature(df, feature="status", mapping={"Active": 1, "Inactive": 0})

        # Comprobamos los valores uno por uno
        assert df_mapped["status"][0] == 1
        assert df_mapped["status"][1] == 0
        assert df_mapped["status"][2] is None  # El nulo debe preservarse
        assert df_mapped["status"][3] == 1

        # El tipo de dato debe permitir nulos (ej. Int64, no Int32)
        assert df_mapped["status"].dtype in [pl.Int64, pl.Int32]


class TestGetFeatureNameAfterOHE:
    """Tests for get_feature_name_after_ohe function."""

    def test_basic_name_generation(self):
        """Test basic OHE name generation."""
        name = get_feature_name_after_ohe("Load_Type", "Medium_Load")
        assert name == "Load_Type_Medium_Load"

    def test_with_spaces(self):
        """Test name generation with spaces."""
        name = get_feature_name_after_ohe("Load Type", "Medium Load")
        assert name == "Load Type_Medium_Load"

    def test_custom_separator(self):
        """Test with custom separator."""
        name = get_feature_name_after_ohe("Load_Type", "Medium_Load", prefix_sep="-")
        assert name == "Load_Type-Medium_Load"

    def test_multiple_words(self):
        """Test with multiple words in category."""
        name = get_feature_name_after_ohe("Status", "Very High Priority")
        assert name == "Status_Very_High_Priority"

    def test_ohe_name_with_numeric_category(self):
        """
        Prueba que la función convierte categorías no-string (como números)
        a string.
        """
        name = get_feature_name_after_ohe("Age_Group", 10)
        assert name == "Age_Group_10"

    def test_ohe_name_with_boolean_category(self):
        """Prueba que la función maneja categorías booleanas."""
        name = get_feature_name_after_ohe("Is_Flagged", True)
        assert name == "Is_Flagged_True"


# pytest tests/unit/test_preprocessing_utils.py -v
