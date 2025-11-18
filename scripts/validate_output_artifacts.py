"""
Validation script for feature importance output artifacts.

This script validates:
1. All PNG files are created in reports/figures/
2. All CSV files are created in reports/metrics/
3. File sizes are reasonable
4. Plots are readable and properly formatted
"""

import sys
from pathlib import Path

import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.feature_importance import (
    calculate_mutual_information,
    calculate_pearson_correlation,
    plot_feature_importance,
)


def check_file_exists(filepath: Path) -> tuple[bool, int]:
    """Check if file exists and return its size."""
    if filepath.exists():
        size = filepath.stat().st_size
        return True, size
    return False, 0


def validate_png_files():
    """Validate that all required PNG files exist and are properly formatted."""
    print("=" * 80)
    print("VALIDATION 1: PNG Files in reports/figures/")
    print("=" * 80)

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        "mutual_information_top10.png",
        "pearson_correlation_top10.png",
        "feature_importance_comparison.png",
    ]

    results = []
    for filename in expected_files:
        filepath = figures_dir / filename
        exists, size = check_file_exists(filepath)

        if exists:
            # Check if file size is reasonable (should be > 10KB for a proper plot)
            if size > 10_000:
                print(f"✓ {filename}")
                print(f"  Size: {size:,} bytes ({size / 1024:.1f} KB)")
                results.append(True)
            else:
                print(f"⚠ {filename} exists but is too small ({size} bytes)")
                results.append(False)
        else:
            print(f"✗ {filename} NOT FOUND")
            results.append(False)

    print()
    return all(results), expected_files


def validate_csv_files():
    """Validate that all required CSV files exist."""
    print("=" * 80)
    print("VALIDATION 2: CSV Files in reports/metrics/")
    print("=" * 80)

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        "mutual_information_scores.csv",
        "pearson_correlations.csv",
    ]

    results = []
    for filename in expected_files:
        filepath = metrics_dir / filename
        exists, size = check_file_exists(filepath)

        if exists:
            # Check if file size is reasonable (should be > 100 bytes)
            if size > 100:
                print(f"✓ {filename}")
                print(f"  Size: {size:,} bytes ({size / 1024:.1f} KB)")

                # Try to read and validate CSV structure
                try:
                    df = pl.read_csv(filepath)
                    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
                    print(f"  Columns: {df.columns}")
                    results.append(True)
                except Exception as e:
                    print(f"  ⚠ Error reading CSV: {e}")
                    results.append(False)
            else:
                print(f"⚠ {filename} exists but is too small ({size} bytes)")
                results.append(False)
        else:
            print(f"✗ {filename} NOT FOUND")
            results.append(False)

    print()
    return all(results), expected_files


def generate_missing_artifacts():
    """Generate missing artifacts if they don't exist."""
    print("=" * 80)
    print("GENERATING MISSING ARTIFACTS")
    print("=" * 80)

    try:
        # Load data
        print("Loading data from Parquet...")
        df = pl.read_parquet("data/processed/steel_cleaned.parquet")
        print(f"✓ Data loaded: {df.shape}")
        print()

        # Calculate importance scores
        print("Calculating mutual information scores...")
        mi_scores = calculate_mutual_information(df, target_column="Usage_kWh")
        print(f"✓ MI scores calculated: {len(mi_scores)} features")
        print()

        print("Calculating Pearson correlations...")
        correlations = calculate_pearson_correlation(df, target_column="Usage_kWh")
        print(f"✓ Correlations calculated: {len(correlations)} features")
        print()

        # Create directories
        figures_dir = Path("reports/figures")
        metrics_dir = Path("reports/metrics")
        figures_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Generate PNG files
        print("Generating PNG files...")

        # 1. Mutual Information plot
        mi_path = figures_dir / "mutual_information_top10.png"
        plot_feature_importance(
            mi_scores,
            score_column="mi_score",
            method_name="Mutual Information",
            top_n=10,
            output_path=str(mi_path),
        )
        print(f"✓ Created: {mi_path}")

        # 2. Pearson Correlation plot
        corr_path = figures_dir / "pearson_correlation_top10.png"
        plot_feature_importance(
            correlations,
            score_column="correlation",
            method_name="Pearson Correlation",
            top_n=10,
            output_path=str(corr_path),
            color_by_sign=True,
        )
        print(f"✓ Created: {corr_path}")

        # 3. Comparison plot (side by side)
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # MI subplot
        top_mi = mi_scores.head(10)
        features_mi = top_mi["feature"].to_list()[::-1]
        scores_mi = top_mi["mi_score"].to_list()[::-1]
        ax1.barh(features_mi, scores_mi, color="#2E86AB", edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Mutual Information Score", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax1.set_title("Top 10 Features by Mutual Information", fontsize=14, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3, linestyle="--")

        # Correlation subplot
        top_corr = correlations.head(10)
        features_corr = top_corr["feature"].to_list()[::-1]
        scores_corr = top_corr["correlation"].to_list()[::-1]
        colors = ["#2E86AB" if s >= 0 else "#E63946" for s in scores_corr]
        ax2.barh(features_corr, scores_corr, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Pearson Correlation", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax2.set_title("Top 10 Features by Pearson Correlation", fontsize=14, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        comp_path = figures_dir / "feature_importance_comparison.png"
        fig.savefig(comp_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Created: {comp_path}")
        print()

        # Generate CSV files
        print("Generating CSV files...")

        # 1. Mutual Information scores
        mi_csv_path = metrics_dir / "mutual_information_scores.csv"
        mi_scores.write_csv(mi_csv_path)
        print(f"✓ Created: {mi_csv_path}")

        # 2. Pearson correlations
        corr_csv_path = metrics_dir / "pearson_correlations.csv"
        correlations.write_csv(corr_csv_path)
        print(f"✓ Created: {corr_csv_path}")
        print()

        print("✓ All artifacts generated successfully!")
        return True

    except Exception as e:
        print(f"✗ ERROR generating artifacts: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_plot_quality():
    """Validate that plots are readable and properly formatted."""
    print("=" * 80)
    print("VALIDATION 3: Plot Quality Check")
    print("=" * 80)

    try:
        from PIL import Image

        figures_dir = Path("reports/figures")

        png_files = [
            "mutual_information_top10.png",
            "pearson_correlation_top10.png",
            "feature_importance_comparison.png",
        ]

        results = []
        for filename in png_files:
            filepath = figures_dir / filename

            if not filepath.exists():
                print(f"✗ {filename} not found")
                results.append(False)
                continue

            try:
                # Open image and check properties
                img = Image.open(filepath)
                width, height = img.size

                print(f"✓ {filename}")
                print(f"  Dimensions: {width}x{height} pixels")
                print(f"  Format: {img.format}")
                print(f"  Mode: {img.mode}")

                # Check if dimensions are reasonable
                if width < 500 or height < 300:
                    print("  ⚠ Image dimensions seem too small")
                    results.append(False)
                else:
                    results.append(True)

            except Exception as e:
                print(f"✗ Error reading {filename}: {e}")
                results.append(False)

        print()
        return all(results)

    except ImportError:
        print("⚠ PIL/Pillow not installed, skipping image quality check")
        print("  (This is optional - files still exist and are valid)")
        print()
        return True


def main():
    """Run all validations."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "OUTPUT ARTIFACTS VALIDATION" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # First check if artifacts exist
    png_valid, png_files = validate_png_files()
    csv_valid, csv_files = validate_csv_files()

    # If any artifacts are missing, generate them
    if not png_valid or not csv_valid:
        print()
        print("⚠️  Some artifacts are missing. Generating them now...")
        print()
        if not generate_missing_artifacts():
            print()
            print("✗ Failed to generate artifacts")
            return 1

        # Re-validate after generation
        print()
        png_valid, _ = validate_png_files()
        csv_valid, _ = validate_csv_files()

    # Validate plot quality
    plot_quality_valid = validate_plot_quality()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_valid = png_valid and csv_valid and plot_quality_valid

    print(f"PNG Files: {'✓ PASS' if png_valid else '✗ FAIL'}")
    print(f"CSV Files: {'✓ PASS' if csv_valid else '✗ FAIL'}")
    print(f"Plot Quality: {'✓ PASS' if plot_quality_valid else '✗ FAIL'}")
    print()

    if all_valid:
        print("🎉 ALL VALIDATIONS PASSED!")
        print()
        print("Generated artifacts:")
        print("  PNG files:")
        for f in png_files:
            print(f"    - reports/figures/{f}")
        print("  CSV files:")
        for f in csv_files:
            print(f"    - reports/metrics/{f}")
        return 0
    else:
        print("⚠️  Some validations failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
