"""
Duplicate Detection Utilities

This module provides functions for detecting duplicate records:
- Exact duplicates (all columns)
- Partial duplicates (subset of columns)
"""

import polars as pl


def detect_duplicates_exact(df: pl.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Detect exact duplicate rows (all columns identical).
    
    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset for logging
        
    Returns:
        dict: Duplicate statistics including count, percentage, and duplicate rows
    """
    # Find duplicates
    duplicates = df.filter(pl.struct(df.columns).is_duplicated())
    
    duplicate_count = len(duplicates)
    total_count = len(df)
    duplicate_percentage = (duplicate_count / total_count) * 100 if total_count > 0 else 0
    
    # Get unique duplicate groups
    unique_duplicates = df.filter(pl.struct(df.columns).is_duplicated()).unique()
    
    print(f"\n{dataset_name} - Exact Duplicate Detection:")
    print(f"  Total rows: {total_count:,}")
    print(f"  Duplicate rows: {duplicate_count:,} ({duplicate_percentage:.2f}%)")
    print(f"  Unique duplicate patterns: {len(unique_duplicates):,}")
    
    return {
        'dataset_name': dataset_name,
        'duplicate_count': duplicate_count,
        'duplicate_percentage': round(duplicate_percentage, 2),
        'total_count': total_count,
        'unique_duplicate_patterns': len(unique_duplicates),
        'duplicate_rows': duplicates
    }


def detect_duplicates_partial(
    df: pl.DataFrame, 
    subset: list[str], 
    dataset_name: str = "Dataset"
) -> dict:
    """
    Detect partial duplicate rows based on a subset of columns.
    
    Args:
        df: Input DataFrame
        subset: List of column names to check for duplicates
        dataset_name: Name of the dataset for logging
        
    Returns:
        dict: Duplicate statistics including count, percentage, and duplicate rows
    """
    # Verify subset columns exist
    missing_cols = [col for col in subset if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Find duplicates based on subset
    duplicates = df.filter(pl.struct(subset).is_duplicated())
    
    duplicate_count = len(duplicates)
    total_count = len(df)
    duplicate_percentage = (duplicate_count / total_count) * 100 if total_count > 0 else 0
    
    # Get unique duplicate groups
    unique_duplicates = df.filter(pl.struct(subset).is_duplicated()).unique(subset=subset)
    
    print(f"\n{dataset_name} - Partial Duplicate Detection (subset: {subset}):")
    print(f"  Total rows: {total_count:,}")
    print(f"  Duplicate rows: {duplicate_count:,} ({duplicate_percentage:.2f}%)")
    print(f"  Unique duplicate patterns: {len(unique_duplicates):,}")
    
    return {
        'dataset_name': dataset_name,
        'subset': subset,
        'duplicate_count': duplicate_count,
        'duplicate_percentage': round(duplicate_percentage, 2),
        'total_count': total_count,
        'unique_duplicate_patterns': len(unique_duplicates),
        'duplicate_rows': duplicates
    }


def compare_duplicate_detection(
    dirty_exact: dict, 
    clean_exact: dict,
    dirty_partial: dict, 
    clean_partial: dict
) -> pl.DataFrame:
    """
    Compare duplicate detection results between dirty and clean datasets.
    
    Args:
        dirty_exact: Exact duplicate results from dirty dataset
        clean_exact: Exact duplicate results from clean dataset
        dirty_partial: Partial duplicate results from dirty dataset
        clean_partial: Partial duplicate results from clean dataset
        
    Returns:
        DataFrame comparing duplicate detection results
    """
    comparison_data = [
        {
            'type': 'Exact Duplicates',
            'dirty_count': dirty_exact['duplicate_count'],
            'dirty_pct': dirty_exact['duplicate_percentage'],
            'clean_count': clean_exact['duplicate_count'],
            'clean_pct': clean_exact['duplicate_percentage'],
            'difference': dirty_exact['duplicate_count'] - clean_exact['duplicate_count']
        },
        {
            'type': f"Partial Duplicates ({', '.join(dirty_partial['subset'])})",
            'dirty_count': dirty_partial['duplicate_count'],
            'dirty_pct': dirty_partial['duplicate_percentage'],
            'clean_count': clean_partial['duplicate_count'],
            'clean_pct': clean_partial['duplicate_percentage'],
            'difference': dirty_partial['duplicate_count'] - clean_partial['duplicate_count']
        }
    ]
    
    return pl.DataFrame(comparison_data)
