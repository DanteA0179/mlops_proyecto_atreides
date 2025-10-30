"""
Unit tests for data cleaning functions
US-006: Clean Data Pipeline
"""

import pytest
import polars as pl
from pathlib import Path

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
