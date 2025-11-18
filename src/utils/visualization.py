"""
Visualization Utilities for Data Quality Analysis

This module provides functions for creating visualizations:
- Null value visualizations
- Outlier visualizations
- Type validation visualizations
- Duplicate detection visualizations
- Range violation visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Color scheme
COLORS = {"clean": "#2E86AB", "dirty": "#A23B72", "neutral": "#6C757D"}


def visualize_nulls(null_df: pl.DataFrame, dataset_name: str, color: str):
    """
    Generate horizontal bar chart of null counts.

    Args:
        null_df: DataFrame with null analysis results
        dataset_name: Name of the dataset
        color: Color for the bars
    """
    # Filter to only columns with nulls
    null_df_filtered = null_df.filter(pl.col("null_count") > 0)

    if len(null_df_filtered) == 0:
        print(f"No null values found in {dataset_name}")
        return

    plt.figure(figsize=(12, max(6, len(null_df_filtered) * 0.4)))

    # Convert to pandas for plotting
    plot_data = null_df_filtered.to_pandas()

    # Create horizontal bar chart
    plt.barh(plot_data["column"], plot_data["null_count"], color=color, alpha=0.7)

    # Add percentage labels
    for i, (count, pct) in enumerate(
        zip(plot_data["null_count"], plot_data["null_percentage"], strict=False)
    ):
        plt.text(count, i, f" {count:,} ({pct:.1f}%)", va="center", fontsize=10)

    plt.xlabel("Null Count", fontsize=12)
    plt.ylabel("Column", fontsize=12)
    plt.title(f"{dataset_name} - Null Values by Column", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_null_comparison(comparison_df: pl.DataFrame):
    """
    Create side-by-side bar chart comparing null counts.

    Args:
        comparison_df: DataFrame with null comparison results
    """
    # Filter to columns with nulls in either dataset
    comparison_filtered = comparison_df.filter(
        (pl.col("dirty_null_count") > 0) | (pl.col("clean_null_count") > 0)
    )

    if len(comparison_filtered) == 0:
        print("No null values found in either dataset")
        return

    plt.figure(figsize=(14, max(6, len(comparison_filtered) * 0.5)))

    # Convert to pandas for plotting
    plot_data = comparison_filtered.to_pandas()

    # Create grouped bar chart
    x = range(len(plot_data))
    width = 0.35

    plt.barh(
        [i - width / 2 for i in x],
        plot_data["clean_null_count"],
        width,
        label="Clean",
        color=COLORS["clean"],
        alpha=0.7,
    )
    plt.barh(
        [i + width / 2 for i in x],
        plot_data["dirty_null_count"],
        width,
        label="Dirty",
        color=COLORS["dirty"],
        alpha=0.7,
    )

    plt.yticks(x, plot_data["column"])
    plt.xlabel("Null Count", fontsize=12)
    plt.ylabel("Column", fontsize=12)
    plt.title("Null Values Comparison: Clean vs Dirty", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_outliers_boxplots(
    df: pl.DataFrame, numeric_cols: list[str], dataset_name: str, color: str
):
    """
    Generate boxplots for numeric columns to visualize outliers.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        dataset_name: Name of the dataset
        color: Color for the boxplots
    """
    if not numeric_cols:
        print(f"No numeric columns to plot for {dataset_name}")
        return

    # Calculate grid dimensions
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Convert to pandas for plotting
    df_pandas = df.to_pandas()

    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row][col_idx]

        # Create boxplot
        sns.boxplot(y=df_pandas[col], ax=ax, color=color, width=0.5)
        ax.set_title(f"{col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row][col_idx].axis("off")

    plt.suptitle(
        f"{dataset_name} - Outlier Detection (Boxplots)", fontsize=16, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    plt.show()


def visualize_type_validation(comparison_df: pl.DataFrame):
    """
    Create visualization of type validation results.

    Args:
        comparison_df: DataFrame with type validation comparison results
    """
    # Count status categories
    clean_valid = (comparison_df["clean_valid"]).sum()
    clean_invalid = (not comparison_df["clean_valid"]).sum()
    dirty_valid = (comparison_df["dirty_valid"]).sum()
    dirty_invalid = (not comparison_df["dirty_valid"]).sum()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Clean dataset
    clean_data = [clean_valid, clean_invalid]
    clean_labels = [f"Valid\n({clean_valid})", f"Invalid\n({clean_invalid})"]
    colors_clean = [COLORS["clean"], "#FF6B6B"]

    ax1.pie(
        clean_data,
        labels=clean_labels,
        colors=colors_clean,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax1.set_title("Clean Dataset - Type Validation", fontsize=14, fontweight="bold")

    # Plot 2: Dirty dataset
    dirty_data = [dirty_valid, dirty_invalid]
    dirty_labels = [f"Valid\n({dirty_valid})", f"Invalid\n({dirty_invalid})"]
    colors_dirty = [COLORS["dirty"], "#FF6B6B"]

    ax2.pie(
        dirty_data,
        labels=dirty_labels,
        colors=colors_dirty,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax2.set_title("Dirty Dataset - Type Validation", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Create bar chart for comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Valid", "Invalid"]
    clean_counts = [clean_valid, clean_invalid]
    dirty_counts = [dirty_valid, dirty_invalid]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, clean_counts, width, label="Clean Dataset", color=COLORS["clean"], alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2, dirty_counts, width, label="Dirty Dataset", color=COLORS["dirty"], alpha=0.7
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_xlabel("Validation Status", fontsize=12)
    ax.set_ylabel("Number of Columns", fontsize=12)
    ax.set_title(
        "Type Validation Comparison: Clean vs Dirty Dataset", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_duplicate_comparison(comparison_df: pl.DataFrame):
    """
    Create visualization comparing duplicate detection results.

    Args:
        comparison_df: DataFrame with duplicate comparison results
    """
    plt.figure(figsize=(12, 6))

    # Convert to pandas
    plot_data = comparison_df.to_pandas()

    # Create grouped bar chart
    x = range(len(plot_data))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        plot_data["clean_count"],
        width,
        label="Clean",
        color=COLORS["clean"],
        alpha=0.7,
    )
    plt.bar(
        [i + width / 2 for i in x],
        plot_data["dirty_count"],
        width,
        label="Dirty",
        color=COLORS["dirty"],
        alpha=0.7,
    )

    # Add value labels
    for i, (clean, dirty) in enumerate(
        zip(plot_data["clean_count"], plot_data["dirty_count"], strict=False)
    ):
        plt.text(i - width / 2, clean, f"{clean:,}", ha="center", va="bottom", fontsize=10)
        plt.text(i + width / 2, dirty, f"{dirty:,}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, plot_data["type"], rotation=15, ha="right")
    plt.ylabel("Duplicate Count", fontsize=12)
    plt.title("Duplicate Detection Comparison: Clean vs Dirty", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_range_violations(comparison_df: pl.DataFrame):
    """
    Create visualization of range violations comparison.

    Args:
        comparison_df: DataFrame with range violation comparison
    """
    # Filter to columns with violations
    violations_df = comparison_df.filter(
        (pl.col("dirty_violations") > 0) | (pl.col("clean_violations") > 0)
    )

    if len(violations_df) == 0:
        print("No range violations found in either dataset")
        return

    plt.figure(figsize=(14, max(6, len(violations_df) * 0.5)))

    # Convert to pandas
    plot_data = violations_df.to_pandas()

    # Create grouped bar chart
    x = range(len(plot_data))
    width = 0.35

    plt.barh(
        [i - width / 2 for i in x],
        plot_data["clean_violations"],
        width,
        label="Clean",
        color=COLORS["clean"],
        alpha=0.7,
    )
    plt.barh(
        [i + width / 2 for i in x],
        plot_data["dirty_violations"],
        width,
        label="Dirty",
        color=COLORS["dirty"],
        alpha=0.7,
    )

    plt.yticks(x, plot_data["column"])
    plt.xlabel("Violation Count", fontsize=12)
    plt.ylabel("Column", fontsize=12)
    plt.title("Range Violations Comparison: Clean vs Dirty", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_distribution_comparison(dirty_df: pl.DataFrame, clean_df: pl.DataFrame, column: str):
    """
    Create side-by-side histograms comparing distributions.

    Args:
        dirty_df: Dirty dataset
        clean_df: Clean dataset
        column: Column name to compare
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convert to pandas
    clean_data = clean_df[column].to_pandas()
    dirty_data = dirty_df[column].to_pandas()

    # Clean dataset histogram
    ax1.hist(clean_data, bins=30, color=COLORS["clean"], alpha=0.7, edgecolor="black")
    ax1.set_title(f"Clean Dataset - {column}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Value", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Dirty dataset histogram
    ax2.hist(dirty_data, bins=30, color=COLORS["dirty"], alpha=0.7, edgecolor="black")
    ax2.set_title(f"Dirty Dataset - {column}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Value", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Distribution Comparison: {column}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
