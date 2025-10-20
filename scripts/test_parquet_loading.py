"""
Test script to verify Parquet data loading and compatibility with feature importance functions.

This script validates:
1. Parquet file can be loaded successfully
2. Special characters in column names are handled correctly (e.g., CO2(tCO2))
3. Polars DataFrame is compatible with all feature importance functions
4. All functions work end-to-end with real data
"""

import sys
from pathlib import Path

import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_importance import (
    calculate_mutual_information,
    calculate_pearson_correlation,
    compare_importance_methods,
    get_top_features,
    plot_feature_importance,
)


def test_parquet_loading():
    """Test loading Parquet file and verify data structure."""
    print("=" * 80)
    print("TEST 1: Parquet Data Loading")
    print("=" * 80)

    parquet_path = "data/processed/steel_cleaned.parquet"

    try:
        # Load Parquet file
        df = pl.read_parquet(parquet_path)
        print(f"✓ Successfully loaded Parquet file from: {parquet_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        print()

        # Display column names
        print("Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col} ({df[col].dtype})")
        print()

        # Check for special characters in column names
        special_char_cols = [col for col in df.columns if any(c in col for c in "()[]{}")]
        if special_char_cols:
            print(f"✓ Found {len(special_char_cols)} column(s) with special characters:")
            for col in special_char_cols:
                print(f"  - {col}")
        else:
            print("  No columns with special characters found")
        print()

        # Verify target column exists
        target_col = "Usage_kWh"
        if target_col in df.columns:
            print(f"✓ Target column '{target_col}' found")
            print(f"  Type: {df[target_col].dtype}")
            print(f"  Null count: {df[target_col].null_count()}")
        else:
            print(f"✗ Target column '{target_col}' NOT found")
            return False
        print()

        # Display sample data
        print("Sample data (first 3 rows):")
        print(df.head(3))
        print()

        return True

    except FileNotFoundError:
        print(f"✗ ERROR: Parquet file not found at: {parquet_path}")
        return False
    except Exception as e:
        print(f"✗ ERROR loading Parquet file: {e}")
        return False


def test_mutual_information_compatibility():
    """Test mutual information calculation with Parquet data."""
    print("=" * 80)
    print("TEST 2: Mutual Information Compatibility")
    print("=" * 80)

    try:
        df = pl.read_parquet("data/processed/steel_cleaned.parquet")

        # Calculate mutual information
        mi_scores = calculate_mutual_information(df, target_column="Usage_kWh")
        print(f"✓ Successfully calculated mutual information scores")
        print(f"  Number of features: {len(mi_scores)}")
        print()

        # Display top 5 features
        print("Top 5 features by mutual information:")
        print(mi_scores.head(5))
        print()

        # Verify data types
        assert "feature" in mi_scores.columns, "Missing 'feature' column"
        assert "mi_score" in mi_scores.columns, "Missing 'mi_score' column"
        print("✓ Output DataFrame has correct columns")
        print()

        # Verify MI scores are non-negative
        min_score = mi_scores["mi_score"].min()
        assert min_score >= 0, f"MI scores should be non-negative, got min={min_score}"
        print(f"✓ All MI scores are non-negative (min={min_score:.4f})")
        print()

        return True

    except Exception as e:
        print(f"✗ ERROR in mutual information calculation: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pearson_correlation_compatibility():
    """Test Pearson correlation calculation with Parquet data."""
    print("=" * 80)
    print("TEST 3: Pearson Correlation Compatibility")
    print("=" * 80)

    try:
        df = pl.read_parquet("data/processed/steel_cleaned.parquet")

        # Calculate Pearson correlations
        correlations = calculate_pearson_correlation(df, target_column="Usage_kWh")
        print(f"✓ Successfully calculated Pearson correlations")
        print(f"  Number of features: {len(correlations)}")
        print()

        # Display top 5 features
        print("Top 5 features by absolute correlation:")
        print(correlations.head(5))
        print()

        # Verify data types
        assert "feature" in correlations.columns, "Missing 'feature' column"
        assert "correlation" in correlations.columns, "Missing 'correlation' column"
        assert "abs_correlation" in correlations.columns, "Missing 'abs_correlation' column"
        print("✓ Output DataFrame has correct columns")
        print()

        # Verify correlation range
        min_corr = correlations["correlation"].min()
        max_corr = correlations["correlation"].max()
        assert -1 <= min_corr <= 1, f"Correlation out of range: min={min_corr}"
        assert -1 <= max_corr <= 1, f"Correlation out of range: max={max_corr}"
        print(f"✓ All correlations are in valid range [-1, 1]")
        print(f"  Range: [{min_corr:.4f}, {max_corr:.4f}]")
        print()

        return True

    except Exception as e:
        print(f"✗ ERROR in Pearson correlation calculation: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_special_characters_handling():
    """Test that special characters in column names are handled correctly."""
    print("=" * 80)
    print("TEST 4: Special Characters Handling")
    print("=" * 80)

    try:
        df = pl.read_parquet("data/processed/steel_cleaned.parquet")

        # Find columns with special characters
        special_char_cols = [col for col in df.columns if any(c in col for c in "()[]{}")]

        if not special_char_cols:
            print("  No columns with special characters to test")
            return True

        print(f"Testing {len(special_char_cols)} column(s) with special characters:")
        for col in special_char_cols:
            print(f"  - {col}")
        print()

        # Calculate MI scores
        mi_scores = calculate_mutual_information(df)

        # Check if special character columns are in results
        mi_features = set(mi_scores["feature"].to_list())
        for col in special_char_cols:
            if col in mi_features:
                score = mi_scores.filter(pl.col("feature") == col)["mi_score"][0]
                print(f"✓ Column '{col}' handled correctly (MI score: {score:.4f})")
            else:
                print(f"  Column '{col}' not in MI results (might be target or non-numeric)")
        print()

        # Calculate correlations
        correlations = calculate_pearson_correlation(df)

        # Check if special character columns are in results
        corr_features = set(correlations["feature"].to_list())
        for col in special_char_cols:
            if col in corr_features:
                corr = correlations.filter(pl.col("feature") == col)["correlation"][0]
                print(f"✓ Column '{col}' handled correctly (correlation: {corr:.4f})")
            else:
                print(f"  Column '{col}' not in correlation results (might be target or non-numeric)")
        print()

        return True

    except Exception as e:
        print(f"✗ ERROR handling special characters: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_all_functions_integration():
    """Test all functions work together end-to-end."""
    print("=" * 80)
    print("TEST 5: Full Integration Test")
    print("=" * 80)

    try:
        df = pl.read_parquet("data/processed/steel_cleaned.parquet")

        # Calculate both importance methods
        mi_scores = calculate_mutual_information(df)
        correlations = calculate_pearson_correlation(df)
        print("✓ Both importance methods calculated successfully")
        print()

        # Get top features
        top_mi = get_top_features(mi_scores, "mi_score", top_n=5)
        top_corr = get_top_features(correlations, "abs_correlation", top_n=5)
        print(f"✓ Top features extracted successfully")
        print(f"  Top 5 by MI: {top_mi}")
        print(f"  Top 5 by correlation: {top_corr}")
        print()

        # Compare methods
        comparison = compare_importance_methods(mi_scores, correlations, top_n=10)
        print(f"✓ Methods compared successfully")
        print(f"  Common features: {len(comparison['common_features'])}")
        print(f"  MI-only features: {len(comparison['mi_only'])}")
        print(f"  Correlation-only features: {len(comparison['corr_only'])}")
        print(f"  Overlap: {comparison['overlap_percentage']:.1f}%")
        print()

        # Test plotting (without saving)
        fig_mi = plot_feature_importance(
            mi_scores, score_column="mi_score", method_name="Mutual Information", top_n=5
        )
        print("✓ MI plot created successfully")

        fig_corr = plot_feature_importance(
            correlations,
            score_column="correlation",
            method_name="Pearson Correlation",
            top_n=5,
            color_by_sign=True,
        )
        print("✓ Correlation plot created successfully")
        print()

        # Close figures to free memory
        import matplotlib.pyplot as plt

        plt.close(fig_mi)
        plt.close(fig_corr)

        return True

    except Exception as e:
        print(f"✗ ERROR in integration test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PARQUET DATA LOADING VALIDATION TEST" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []

    # Run all tests
    results.append(("Parquet Loading", test_parquet_loading()))
    results.append(("Mutual Information", test_mutual_information_compatibility()))
    results.append(("Pearson Correlation", test_pearson_correlation_compatibility()))
    results.append(("Special Characters", test_special_characters_handling()))
    results.append(("Integration", test_all_functions_integration()))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print()
        print("🎉 ALL TESTS PASSED! Parquet data loading is fully compatible.")
        return 0
    else:
        print()
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
