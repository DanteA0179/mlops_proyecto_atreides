"""Utility functions and helpers for data quality analysis."""

# Data Loading
from .load_datasets import load_dataset

# Data Quality
from .data_quality import (
    analyze_nulls,
    compare_null_patterns,
    compare_schemas,
    compare_type_validation,
    get_numeric_columns,
    validate_types,
)

# Outlier Detection
from .outlier_detection import (
    analyze_outliers_all_columns,
    analyze_outliers_zscore_all_columns,
    detect_outliers_iqr,
    detect_outliers_zscore,
)

# Duplicate Detection
from .duplicate_detection import (
    compare_duplicate_detection,
    detect_duplicates_exact,
    detect_duplicates_partial,
)

# Range Validation
from .range_validation import (
    compare_range_violations,
    show_range_violation_examples,
    validate_categorical_values,
    validate_ranges,
)

# Visualization
from .visualization import (
    COLORS,
    visualize_distribution_comparison,
    visualize_duplicate_comparison,
    visualize_null_comparison,
    visualize_nulls,
    visualize_outliers_boxplots,
    visualize_range_violations,
    visualize_type_validation,
)

# Data Cleaning
from .data_cleaning import (
    convert_data_types,
    correct_range_violations,
    handle_null_values,
    remove_duplicates,
    treat_outliers,
    validate_cleaned_data,
)

# Temporal Features
from .temporal_features import (
    create_all_temporal_features,
    create_cyclical_encoding,
    create_is_weekend,
    extract_day_of_week_numeric,
    extract_hour_from_nsm,
    validate_temporal_features,
)

__all__ = [
    # Data Quality
    "load_dataset",
    "compare_schemas",
    "analyze_nulls",
    "compare_null_patterns",
    "validate_types",
    "compare_type_validation",
    "get_numeric_columns",
    # Outlier Detection
    "detect_outliers_iqr",
    "analyze_outliers_all_columns",
    "detect_outliers_zscore",
    "analyze_outliers_zscore_all_columns",
    # Duplicate Detection
    "detect_duplicates_exact",
    "detect_duplicates_partial",
    "compare_duplicate_detection",
    # Range Validation
    "validate_ranges",
    "show_range_violation_examples",
    "compare_range_violations",
    "validate_categorical_values",
    # Visualization
    "COLORS",
    "visualize_nulls",
    "visualize_null_comparison",
    "visualize_outliers_boxplots",
    "visualize_type_validation",
    "visualize_duplicate_comparison",
    "visualize_range_violations",
    "visualize_distribution_comparison",
    # Data Cleaning
    "convert_data_types",
    "handle_null_values",
    "correct_range_violations",
    "treat_outliers",
    "remove_duplicates",
    "validate_cleaned_data",
    # Temporal Features
    "extract_hour_from_nsm",
    "extract_day_of_week_numeric",
    "create_is_weekend",
    "create_cyclical_encoding",
    "create_all_temporal_features",
    "validate_temporal_features",
]
