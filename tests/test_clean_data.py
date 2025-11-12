"""
Unit tests for data cleaning functions
US-006: Clean Data Pipeline
"""

import pytest
import polars as pl
from pathlib import Path
from polars.testing import assert_frame_equal

from src.utils.data_cleaning import (
    convert_data_types,
    correct_range_violations,
    treat_outliers,
    remove_duplicates,
    validate_cleaned_data,
)
from src.utils.data_quality import analyze_nulls


class TestConvertDataTypes:
    """Tests for convert_data_types function"""
    
    def test_convert_string_to_float(self):
        """Test conversion from String to Float64"""
        df = pl.DataFrame({
            'col1': ['1.5', '2.5', '3.5'],
            'col2': ['a', 'b', 'c']
        })
        
        schema = {'col1': pl.Float64, 'col2': pl.Utf8}
        result = convert_data_types(df, schema)
        
        assert result['col1'].dtype == pl.Float64
        assert result['col2'].dtype == pl.Utf8
    
    def test_convert_string_to_int(self):
        """Test conversion from String to Int64"""
        df = pl.DataFrame({
            'col1': ['1', '2', '3']
        })

        schema = {'col1': pl.Int64}
        result = convert_data_types(df, schema)

        assert result['col1'].dtype == pl.Int64

    def test_convert_decimal_string_to_int(self):
        """Test conversion from decimal String to Int64 (via Float64)"""
        df = pl.DataFrame({
            'col1': ['900.0', '1800.0', '2700.0']
        })

        schema = {'col1': pl.Int64}
        result = convert_data_types(df, schema)

        assert result['col1'].dtype == pl.Int64
        assert result['col1'][0] == 900
        assert result['col1'][1] == 1800
    
    def test_drop_columns(self):
        """Test dropping unwanted columns"""
        df = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [4, 5, 6]
        })
        
        schema = {'col1': pl.Int64, 'col2': pl.Utf8}
        result = convert_data_types(df, schema, drop_columns=['col3'])
        
        assert 'col3' not in result.columns
        assert 'col1' in result.columns
        assert 'col2' in result.columns
    
    def test_handle_invalid_values(self):
        """Test handling of invalid values during conversion"""
        df = pl.DataFrame({
            'col1': ['1.5', 'invalid', '3.5']
        })
        
        schema = {'col1': pl.Float64}
        result = convert_data_types(df, schema)
        
        assert result['col1'].dtype == pl.Float64
        assert result['col1'].null_count() == 1  # 'invalid' becomes null

    #new iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
    def test_convert_empty_dataframe(self):
        """
        Prueba qué pasa si el DataFrame está vacío.
        Esperamos que devuelva un DataFrame también vacío, pero con los tipos correctos.
        """
        # 1. Preparar datos: un DataFrame vacío con un esquema inicial
        df = pl.DataFrame(schema={'col1': pl.Utf8, 'col2': pl.Utf8})
        
        # 2. Definir el esquema objetivo
        schema = {'col1': pl.Int64, 'col2': pl.Float64}
        
        # 3. Ejecutar la función
        result = convert_data_types(df, schema)
        
        # 4. Comprobar
        assert len(result) == 0  # Sigue estando vacío
        assert result['col1'].dtype == pl.Int64  # Tipo de col1 es correcto
        assert result['col2'].dtype == pl.Float64 # Tipo de col2 es correcto
    
    def test_convert_with_existing_nulls(self):
        """
        Prueba que los nulos que ya existen en los datos se mantienen como nulos.
        La función no debe fallar, simplemente debe convertir lo que pueda.
        """
        # 1. Preparar datos: 'col1' tiene un nulo
        df = pl.DataFrame({'col1': ['1.0', None, '3.0']})
        schema = {'col1': pl.Float64}
        
        # 2. Ejecutar la función
        result = convert_data_types(df, schema)
        
        # 3. Comprobar
        assert result['col1'].dtype == pl.Float64 # Tipo es correcto
        assert result['col1'].null_count() == 1 # El nulo original se preservó
        assert result['col1'][2] == 3.0 # El valor no nulo se convirtió


class TestAnalyzeNulls:
    """Tests for analyze_nulls function"""

    def test_analyze_nulls_basic(self):
        """Test basic null analysis"""
        df = pl.DataFrame({
            'col1': [1, None, 3],
            'col2': [None, 2, 3],
            'col3': [1, 2, 3]
        })

        result = analyze_nulls(df, 'Test Dataset')

        assert len(result) == 3
        assert 'column' in result.columns
        assert 'null_count' in result.columns
        assert 'null_percentage' in result.columns

    def test_analyze_nulls_empty_dataframe(self):
        """Test null analysis with empty DataFrame"""
        df = pl.DataFrame({
            'col1': [],
            'col2': []
        })

        result = analyze_nulls(df, 'Empty Dataset')

        assert len(result) == 2
        assert result['null_count'].sum() == 0

    def test_analyze_nulls_no_nulls(self):
        """Test null analysis with no nulls"""
        df = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        result = analyze_nulls(df, 'Clean Dataset')

        assert result['null_count'].sum() == 0


class TestCorrectRangeViolations:
    """Tests for correct_range_violations function"""

    def test_make_absolute(self):
        """Test conversion of negative values to absolute"""
        df = pl.DataFrame({
            'col1': [-5.0, 10.0, -20.0]
        })

        result = correct_range_violations(df, {}, make_absolute=['col1'])

        assert result['col1'][0] == 5.0
        assert result['col1'][1] == 10.0
        assert result['col1'][2] == 20.0

    def test_range_min_max_capping(self):
        """Test range validation with min and max (capping)"""
        df = pl.DataFrame({
            'col1': [-10.0, 50.0, 150.0]
        })

        range_rules = {'col1': {'min': 0, 'max': 100}}
        result = correct_range_violations(df, range_rules)

        # Now it caps instead of setting to null
        assert result['col1'][0] == 0.0  # capped to min
        assert result['col1'][1] == 50.0  # valid
        assert result['col1'][2] == 100.0  # capped to max

    def test_range_min_only(self):
        """Test range validation with min only"""
        df = pl.DataFrame({
            'col1': [-10.0, 50.0, 150.0]
        })

        range_rules = {'col1': {'min': 0}}
        result = correct_range_violations(df, range_rules)

        assert result['col1'][0] == 0.0  # capped to min
        assert result['col1'][1] == 50.0  # valid
        assert result['col1'][2] == 150.0  # valid (no max)
    
    def test_range_no_violations(self):
        """
        Prueba el "camino feliz": ¿Qué pasa si todos los datos ya están bien?
        El DataFrame de salida debe ser idéntico al de entrada.
        """
        # 1. Preparar datos: todos los valores están entre 0 y 100
        df = pl.DataFrame({'col1': [10.0, 50.0, 90.0]})
        range_rules = {'col1': {'min': 0, 'max': 100}}
        
        # Guardamos una copia para comparar
        original_df = df.clone()
        
        # 2. Ejecutar la función
        result = correct_range_violations(df, range_rules)
        
        # 3. Comprobar
        # Comparamos que los DataFrames sean idénticos
        assert_frame_equal(original_df, result)
    
    def test_range_with_nulls(self):
        """
        Prueba que la función ignora los nulos y los deja como nulos.
        No debería intentar comparar 'None' con 0 o 100.
        """
        # 1. Preparar datos: un valor bajo, uno bueno, un nulo y uno alto
        df = pl.DataFrame({'col1': [-10.0, 50.0, None, 150.0]})
        range_rules = {'col1': {'min': 0, 'max': 100}}
        
        # 2. Ejecutar la función
        result = correct_range_violations(df, range_rules)
        
        # 3. Comprobar
        assert result['col1'][0] == 0.0   # Acotado a min
        assert result['col1'][1] == 50.0  # Válido, se queda igual
        assert result['col1'][2] is None  # Nulo, se queda nulo
        assert result['col1'][3] == 100.0 # Acotado a max


class TestTreatOutliers:
    """Tests for treat_outliers function"""
    
    def test_cap_outliers(self):
        """Test capping outliers at percentiles"""
        df = pl.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]  # 100 is outlier
        })

        result = treat_outliers(df, ['col1'], method='cap', lower_percentile=0.1, upper_percentile=0.9)

        # Check that extreme value was capped (should be capped to 90th percentile)
        assert result['col1'].max() <= 10.0  # 90th percentile of [1-9, 100]
    
    def test_remove_outliers(self):
        """Test setting outliers to null"""
        df = pl.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        })
        
        result = treat_outliers(df, ['col1'], method='remove', lower_percentile=0.1, upper_percentile=0.9)
        
        # Check that extreme value was set to null
        assert result['col1'].null_count() > 0
    
    def test_cap_outliers_low(self):
        """
        Prueba que el acotamiento (capping) funciona para outliers *bajos*.
        Tu prueba original 'test_cap_outliers' solo probaba valores altos.
        """
        # 1. Preparar datos: -100.0 es un outlier bajo
        df = pl.DataFrame({
            'col1': [-100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        
        # 2. Ejecutar la función
        # Acotamos al percentil 10 (que sería 2.0) y 90 (que sería 10.0)
        result = treat_outliers(df, ['col1'], method='cap', lower_percentile=0.1, upper_percentile=0.9)
        
        # 3. Comprobar
        # El valor mínimo en el DataFrame resultante debe ser 2.0
        assert result['col1'].min() == 2.0
        # Y el valor que era -100.0 ahora debe ser 2.0
        assert result['col1'][0] == 2.0
    
    def test_treat_outliers_no_outliers(self):
        """
        Prueba el "camino feliz": ¿Qué pasa si no hay outliers?
        El DataFrame de salida debe ser idéntico al de entrada.
        """
        # 1. Preparar datos: todos los valores están "normales"
        df = pl.DataFrame({'col1': [1.0, 2.0, 3.0, 4.0, 5.0]})
        original_df = df.clone()
        
        # 2. Ejecutar la función
        result = treat_outliers(df, ['col1'], method='cap', lower_percentile=0.1, upper_percentile=0.9)
        
        # 3. Comprobar
        assert_frame_equal(original_df, result)


class TestRemoveDuplicates:
    """Tests for remove_duplicates function"""
    
    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicate rows"""
        df = pl.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        result = remove_duplicates(df)
        
        assert len(result) == 3
    
    def test_remove_duplicates_subset(self):
        """Test removal of duplicates based on subset of columns"""
        df = pl.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        result = remove_duplicates(df, subset=['col1'])
        
        assert len(result) == 3
    
    def test_keep_first(self):
        """Test keeping first occurrence of duplicates"""
        df = pl.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'd']  # Make row 2 and 3 truly identical
        })

        result = remove_duplicates(df, keep='first')

        # Should keep first '2' with 'b', remove second
        assert len(result) == 3
    
    def test_remove_duplicates_keep_last(self):
        """
        Prueba que se conserva la *última* aparición de un duplicado.
        """
        # 1. Preparar datos: el '2' está duplicado
        df = pl.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b_primero', 'b_ultimo', 'c'] # Para saber cuál se quedó
        })
        
        # 2. Ejecutar la función
        result = remove_duplicates(df, subset=['col1'], keep='last')
        
        # 3. Comprobar
        assert len(result) == 3 # Quedan 3 filas
        # Nos aseguramos de que la fila que quedó fue la última
        assert 'b_ultimo' in result['col2']
        assert 'b_primero' not in result['col2']
    
    def test_remove_duplicates_keep_none(self):
        """
        Prueba que se eliminan *TODAS* las filas que tenían algún duplicado.
        Si '2' aparece dos veces, se eliminan ambas filas.
        """
        # 1. Preparar datos: 2 y 4 están duplicados
        df = pl.DataFrame({
            'col1': [1, 2, 2, 3, 4, 4, 5]
        })
        
        # 2. Ejecutar la función
        result = remove_duplicates(df, subset=['col1'], keep='none')
        
        # 3. Comprobar
        # El resultado esperado es un DF que solo tiene [1, 3, 5]
        expected = pl.DataFrame({'col1': [1, 3, 5]})
        
        # Comparamos que los DataFrames sean idénticos
        # (Usamos .sort() por si la función no mantiene el orden)
        assert_frame_equal(result.sort('col1'), expected)



class TestValidateCleanedData:
    """Tests for validate_cleaned_data function"""
    
    def test_perfect_match(self):
        """Test validation with perfect match"""
        df1 = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        df2 = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = validate_cleaned_data(df1, df2)
        
        assert result['shape_match'] is True
        assert result['schema_match'] is True
        assert result['null_count'] == 0
        assert result['duplicate_count'] == 0
        assert result['all_checks_passed'] is True
    
    def test_shape_mismatch(self):
        """Test validation with shape mismatch"""
        df1 = pl.DataFrame({
            'col1': [1, 2, 3, 4]
        })
        df2 = pl.DataFrame({
            'col1': [1, 2, 3]
        })
        
        result = validate_cleaned_data(df1, df2, tolerance=0)
        
        assert result['shape_match'] is False
        assert result['row_count_diff'] == 1
    
    def test_type_mismatch(self):
        """Test validation with type mismatch"""
        df1 = pl.DataFrame({
            'col1': [1, 2, 3]
        })
        df2 = pl.DataFrame({
            'col1': [1.0, 2.0, 3.0]
        })
        
        result = validate_cleaned_data(df1, df2)
        
        assert len(result['type_mismatches']) > 0
        assert result['all_checks_passed'] is False



# Run tests with: pytest tests/test_clean_data.py -v
