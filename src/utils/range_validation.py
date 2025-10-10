"""
Range Validation Utilities

This module provides functions for validating value ranges:
- Numeric range validation
- Categorical value validation
- Date/time validation
"""

import polars as pl


def validate_ranges(df: pl.DataFrame, range_rules: dict, dataset_name: str = "Dataset") -> dict:
    """
    Validate that values are within expected ranges.
    
    Args:
        df: Input DataFrame
        range_rules: Dictionary mapping column names to range specifications
                    Format: {column: {'min': value, 'max': value}}
        dataset_name: Name of the dataset for logging
        
    Returns:
        dict: Validation results with violations per column
    """
    validation_results = {}
    total_violations = 0
    
    for column, rules in range_rules.items():
        if column not in df.columns:
            validation_results[column] = {
                'status': 'column_not_found',
                'violations': 0,
                'violation_percentage': 0.0,
                'total_count': 0,
                'min_expected': rules.get('min'),
                'max_expected': rules.get('max'),
                'examples': []
            }
            continue

        min_val = rules.get('min')
        max_val = rules.get('max')
        
        try:
            # Check if column is string type and try to cast to numeric
            if df[column].dtype == pl.Utf8:
                # Try to cast to float for comparison
                df_temp = df.with_columns([
                    pl.col(column).cast(pl.Float64, strict=False).alias(column + '_numeric')
                ])
                compare_col = column + '_numeric'
            else:
                df_temp = df
                compare_col = column
            
            # Build filter condition
            violations_filter = None
            if min_val is not None and max_val is not None:
                violations_filter = (pl.col(compare_col) < min_val) | (pl.col(compare_col) > max_val)
            elif min_val is not None:
                violations_filter = pl.col(compare_col) < min_val
            elif max_val is not None:
                violations_filter = pl.col(compare_col) > max_val
            
            if violations_filter is None:
                continue
            
            # Find violations (excluding nulls)
            violations = df_temp.filter(violations_filter & pl.col(compare_col).is_not_null())
            violation_count = len(violations)
            
            # Count non-null values for percentage calculation
            non_null_count = df_temp[compare_col].drop_nulls().shape[0]
            violation_percentage = (violation_count / non_null_count) * 100 if non_null_count > 0 else 0
            
            # Get examples (up to 5) from original column
            if violation_count > 0:
                examples = violations[column].unique().head(5).to_list()
            else:
                examples = []
            
            validation_results[column] = {
                'status': 'violations_found' if violation_count > 0 else 'valid',
                'violations': violation_count,
                'violation_percentage': round(violation_percentage, 2),
                'total_count': non_null_count,
                'min_expected': min_val,
                'max_expected': max_val,
                'examples': examples
            }
            
            total_violations += violation_count
            
        except Exception as e:
            # If casting fails, mark as error
            validation_results[column] = {
                'status': 'error',
                'violations': 0,
                'violation_percentage': 0.0,
                'total_count': len(df),
                'min_expected': min_val,
                'max_expected': max_val,
                'examples': [],
                'error': str(e)
            }
    
    print(f"\n{dataset_name} - Range Validation:")
    print(f"  Columns validated: {len(range_rules)}")
    print(f"  Columns with violations: {sum(1 for v in validation_results.values() if v['violations'] > 0)}")
    print(f"  Total violations: {total_violations:,}")
    
    return validation_results


def show_range_violation_examples(validation_results: dict, dataset_name: str):
    """
    Display examples of range violations with frequencies.
    
    Args:
        validation_results: Results from validate_ranges()
        dataset_name: Name of the dataset for logging
    """
    print(f"\n{dataset_name} - Range Violation Examples:")
    
    has_violations = False
    for column, results in validation_results.items():
        if results['violations'] > 0:
            has_violations = True
            print(f"\n  Column: {column}")
            print(f"    Expected range: [{results.get('min_expected', 'N/A')}, {results.get('max_expected', 'N/A')}]")
            print(f"    Violations: {results['violations']:,} ({results['violation_percentage']:.2f}%)")
            print(f"    Example values: {results['examples'][:5]}")
    
    if not has_violations:
        print("  ✓ No range violations found")


def compare_range_violations(dirty_validation: dict, clean_validation: dict) -> pl.DataFrame:
    """
    Compare range violations between dirty and clean datasets.
    
    Args:
        dirty_validation: Validation results from dirty dataset
        clean_validation: Validation results from clean dataset
        
    Returns:
        DataFrame comparing range violations
    """
    comparison_data = []
    
    # Get all columns from both validations
    all_columns = set(dirty_validation.keys()) | set(clean_validation.keys())
    
    for column in sorted(all_columns):
        dirty_result = dirty_validation.get(column, {})
        clean_result = clean_validation.get(column, {})
        
        dirty_violations = dirty_result.get('violations', 0)
        clean_violations = clean_result.get('violations', 0)
        
        comparison_data.append({
            'column': column,
            'dirty_violations': dirty_violations,
            'dirty_pct': dirty_result.get('violation_percentage', 0.0),
            'clean_violations': clean_violations,
            'clean_pct': clean_result.get('violation_percentage', 0.0),
            'difference': dirty_violations - clean_violations,
            'issue_introduced': clean_violations == 0 and dirty_violations > 0
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    comparison_df = comparison_df.sort('difference', descending=True)
    
    return comparison_df


def validate_categorical_values(
    df: pl.DataFrame, 
    column: str, 
    valid_values: list,
    dataset_name: str = "Dataset"
) -> dict:
    """
    Validate that categorical column contains only expected values.
    
    Args:
        df: Input DataFrame
        column: Column name to validate
        valid_values: List of valid categorical values
        dataset_name: Name of the dataset for logging
        
    Returns:
        dict: Validation results with invalid values
    """
    if column not in df.columns:
        return {
            'status': 'column_not_found',
            'invalid_count': 0,
            'invalid_values': []
        }
    
    # Find invalid values
    invalid_rows = df.filter(~pl.col(column).is_in(valid_values))
    invalid_count = len(invalid_rows)
    invalid_values = invalid_rows[column].unique().to_list() if invalid_count > 0 else []
    
    total_count = len(df)
    invalid_percentage = (invalid_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n{dataset_name} - Categorical Validation ({column}):")
    print(f"  Valid values: {valid_values}")
    print(f"  Invalid rows: {invalid_count:,} ({invalid_percentage:.2f}%)")
    if invalid_values:
        print(f"  Invalid values found: {invalid_values}")
    
    return {
        'status': 'invalid_values_found' if invalid_count > 0 else 'valid',
        'column': column,
        'invalid_count': invalid_count,
        'invalid_percentage': round(invalid_percentage, 2),
        'total_count': total_count,
        'valid_values': valid_values,
        'invalid_values': invalid_values
    }
