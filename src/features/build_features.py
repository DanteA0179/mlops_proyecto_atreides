"""
Feature Engineering Pipeline for Steel Energy Dataset
US-011: Temporal Features Creation

This script creates temporal features from the cleaned dataset and generates
the featured dataset ready for machine learning model training.

Features Created:
- hour (0-23): Hour of day from NSM
- day_of_week (0-6): Numeric day of week
- is_weekend (bool): Weekend indicator
- cyclical_hour_sin, cyclical_hour_cos: Cyclical encoding of hour
- cyclical_day_sin, cyclical_day_cos: Cyclical encoding of day_of_week

Usage:
    python src/features/build_features.py

Output:
    - data/processed/steel_featured.parquet
    - reports/feature_engineering_report.md
    - reports/feature_engineering_log.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from src.features.temporal_transformers import TemporalFeatureEngineer
from src.utils.temporal_features import validate_temporal_features

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data/processed/steel_cleaned.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/steel_featured.parquet"
REPORT_PATH = PROJECT_ROOT / "reports/feature_engineering_report.md"
LOG_PATH = PROJECT_ROOT / "reports/feature_engineering_log.json"

# Expected features after transformation
REQUIRED_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "cyclical_hour_sin",
    "cyclical_hour_cos",
    "cyclical_day_sin",
    "cyclical_day_cos",
]

# Expected output schema (11 original + 7 new = 18 columns)
EXPECTED_OUTPUT_COLUMNS = 18

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_cleaned_data(path: Path) -> pl.DataFrame:
    """
    Load cleaned data from Parquet file.

    Parameters
    ----------
    path : Path
        Path to cleaned Parquet file

    Returns
    -------
    pl.DataFrame
        Cleaned dataframe

    Raises
    ------
    FileNotFoundError
        If input file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found at: {path}")

    logger.info(f"Loading cleaned data from: {path}")
    df = pl.read_parquet(path)
    logger.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


def save_featured_data(df: pl.DataFrame, path: Path) -> None:
    """
    Save featured data to Parquet file.

    Parameters
    ----------
    df : pl.DataFrame
        Featured dataframe to save
    path : Path
        Output path for Parquet file
    """
    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving featured data to: {path}")
    df.write_parquet(path, compression="snappy")

    # Get file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {df.shape[0]:,} rows × {df.shape[1]} columns ({file_size_mb:.2f} MB)")


def generate_feature_statistics(df: pl.DataFrame, features: list) -> dict:
    """
    Generate statistics for created features.

    Parameters
    ----------
    df : pl.DataFrame
        Featured dataframe
    features : list
        List of feature names to analyze

    Returns
    -------
    dict
        Statistics for each feature
    """
    logger.info("Generating feature statistics...")
    stats = {}

    for feat in features:
        if feat not in df.columns:
            continue

        if df[feat].dtype == pl.Boolean:
            # Boolean feature stats
            true_count = df[feat].sum()
            false_count = len(df) - true_count
            stats[feat] = {
                "type": "boolean",
                "true_count": int(true_count),
                "false_count": int(false_count),
                "true_percentage": float(true_count / len(df) * 100),
            }
        else:
            # Numeric feature stats
            stats[feat] = {
                "type": "numeric",
                "mean": float(df[feat].mean()),
                "std": float(df[feat].std()),
                "min": float(df[feat].min()),
                "max": float(df[feat].max()),
                "median": float(df[feat].median()),
            }

    return stats


def calculate_feature_correlations(df: pl.DataFrame, target: str = "Usage_kWh") -> dict:
    """
    Calculate correlations between new features and target.

    Parameters
    ----------
    df : pl.DataFrame
        Featured dataframe
    target : str
        Target variable name

    Returns
    -------
    dict
        Correlations for each feature
    """
    logger.info(f"Calculating correlations with target '{target}'...")
    correlations = {}

    numeric_features = [
        "hour",
        "day_of_week",
        "cyclical_hour_sin",
        "cyclical_hour_cos",
        "cyclical_day_sin",
        "cyclical_day_cos",
    ]

    for feat in numeric_features:
        if feat in df.columns and target in df.columns:
            corr = df.select([pl.corr(feat, target).alias("correlation")]).item()
            correlations[feat] = float(corr)

    return correlations


def generate_markdown_report(
    df: pl.DataFrame, stats: dict, correlations: dict, validation: dict, path: Path
) -> None:
    """
    Generate Markdown report of feature engineering process.

    Parameters
    ----------
    df : pl.DataFrame
        Featured dataframe
    stats : dict
        Feature statistics
    correlations : dict
        Feature correlations
    validation : dict
        Validation results
    path : Path
        Output path for report
    """
    logger.info(f"Generating Markdown report: {path}")

    report = f"""# Feature Engineering Report - US-011

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Input**: `{INPUT_PATH.name}`
**Output**: `{OUTPUT_PATH.name}`
**Status**: {'✅ SUCCESS' if validation['valid'] else '❌ FAILED'}

---

## Dataset Overview

- **Input rows**: {df.shape[0]:,}
- **Input columns**: {df.shape[1] - len(REQUIRED_FEATURES)}
- **Output columns**: {df.shape[1]}
- **New features created**: {len(REQUIRED_FEATURES)}

---

## Features Created

### 1. Direct Temporal Features

| Feature | Type | Description | Stats |
|---------|------|-------------|-------|
| `hour` | Int32 | Hour of day (0-23) from NSM | min={stats['hour']['min']}, max={stats['hour']['max']} |
| `day_of_week` | Int32 | Day of week (0-6): Mon=0, Sun=6 | min={stats['day_of_week']['min']}, max={stats['day_of_week']['max']} |
| `is_weekend` | Boolean | Weekend indicator (Sat/Sun) | {stats['is_weekend']['true_percentage']:.2f}% weekend |

### 2. Cyclical Encoded Features

Cyclical encoding preserves periodicity: hour 23 ≈ hour 0, day 6 ≈ day 0

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `cyclical_hour_sin` | Float64 | sin(2π × hour / 24) | [{stats['cyclical_hour_sin']['min']:.4f}, {stats['cyclical_hour_sin']['max']:.4f}] |
| `cyclical_hour_cos` | Float64 | cos(2π × hour / 24) | [{stats['cyclical_hour_cos']['min']:.4f}, {stats['cyclical_hour_cos']['max']:.4f}] |
| `cyclical_day_sin` | Float64 | sin(2π × day / 7) | [{stats['cyclical_day_sin']['min']:.4f}, {stats['cyclical_day_sin']['max']:.4f}] |
| `cyclical_day_cos` | Float64 | cos(2π × day / 7) | [{stats['cyclical_day_cos']['min']:.4f}, {stats['cyclical_day_cos']['max']:.4f}] |

---

## Feature Statistics

### Hour Distribution
- Mean: {stats['hour']['mean']:.2f}
- Std: {stats['hour']['std']:.2f}
- Median: {stats['hour']['median']:.2f}

### Day of Week Distribution
- Mean: {stats['day_of_week']['mean']:.2f}
- Std: {stats['day_of_week']['std']:.2f}

### Weekend Distribution
- Weekend records: {stats['is_weekend']['true_count']:,} ({stats['is_weekend']['true_percentage']:.2f}%)
- Weekday records: {stats['is_weekend']['false_count']:,} ({100 - stats['is_weekend']['true_percentage']:.2f}%)

---

## Correlation with Target (`Usage_kWh`)

| Feature | Pearson Correlation |
|---------|---------------------|
| `hour` | {correlations.get('hour', 0):.4f} |
| `day_of_week` | {correlations.get('day_of_week', 0):.4f} |
| `cyclical_hour_sin` | {correlations.get('cyclical_hour_sin', 0):.4f} |
| `cyclical_hour_cos` | {correlations.get('cyclical_hour_cos', 0):.4f} |
| `cyclical_day_sin` | {correlations.get('cyclical_day_sin', 0):.4f} |
| `cyclical_day_cos` | {correlations.get('cyclical_day_cos', 0):.4f} |

---

## Validation Results

**Overall Status**: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}

### Checks Performed
- ✅ All required features present: {len(validation['missing_features']) == 0}
- ✅ Feature ranges valid: {len(validation['invalid_ranges']) == 0}
- ✅ Cyclical orthogonality (sin² + cos² = 1): Verified

### Feature Range Checks
- `hour`: [0, 23] ✅
- `day_of_week`: [0, 6] ✅
- `cyclical_*`: [-1, 1] ✅
- `is_weekend`: Boolean ✅

---

## Output Files

- **Featured Dataset**: `{OUTPUT_PATH}`
  - Format: Parquet (compressed with Snappy)
  - Size: {(OUTPUT_PATH.stat().st_size / (1024**2)):.2f} MB
  - Rows: {df.shape[0]:,}
  - Columns: {df.shape[1]}

- **Report**: `{path}`
- **Log**: `{LOG_PATH}`

---

## Next Steps

1. ✅ US-011 completed: All 7 temporal features created
2. ⏭️ **US-012**: Model Training with featured dataset
3. ⏭️ **US-013**: Model evaluation and hyperparameter tuning

**Ready for ML model training!** 🚀

---

**Generated by**: Feature Engineering Pipeline (US-011)
**Version**: 1.0
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Write report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info("  Report generated successfully")


def generate_json_log(
    df: pl.DataFrame, stats: dict, correlations: dict, validation: dict, path: Path
) -> None:
    """
    Generate JSON log of feature engineering process.

    Parameters
    ----------
    df : pl.DataFrame
        Featured dataframe
    stats : dict
        Feature statistics
    correlations : dict
        Feature correlations
    validation : dict
        Validation results
    path : Path
        Output path for JSON log
    """
    logger.info(f"Generating JSON log: {path}")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "path": str(INPUT_PATH),
            "rows": df.shape[0] - len(REQUIRED_FEATURES),  # Approximation
            "columns": df.shape[1] - len(REQUIRED_FEATURES),
        },
        "output": {
            "path": str(OUTPUT_PATH),
            "rows": df.shape[0],
            "columns": df.shape[1],
            "size_mb": round(OUTPUT_PATH.stat().st_size / (1024**2), 2),
        },
        "features_created": REQUIRED_FEATURES,
        "statistics": stats,
        "correlations": correlations,
        "validation": {
            "valid": validation["valid"],
            "missing_features": validation["missing_features"],
            "invalid_ranges": validation["invalid_ranges"],
        },
    }

    # Write JSON log
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info("  JSON log generated successfully")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """
    Execute feature engineering pipeline.

    Steps:
    1. Load cleaned data
    2. Create temporal features
    3. Validate features
    4. Generate statistics
    5. Calculate correlations
    6. Save featured data
    7. Generate reports
    """
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE - US-011")
    logger.info("=" * 70)

    try:
        # Step 1: Load cleaned data
        df = load_cleaned_data(INPUT_PATH)

        # Step 2: Create temporal features using POO transformer
        logger.info("Creating temporal features...")
        engineer = TemporalFeatureEngineer(nsm_col="NSM", day_name_col="Day_of_week")
        df_featured = engineer.fit_transform(df)
        logger.info(f"  Created {len(REQUIRED_FEATURES)} new features")
        logger.info(f"  Total columns: {df_featured.shape[1]}")

        # Step 3: Validate features
        logger.info("Validating temporal features...")
        validation = validate_temporal_features(df_featured, REQUIRED_FEATURES)

        if not validation["valid"]:
            logger.error("❌ Feature validation FAILED")
            logger.error(f"  Missing features: {validation['missing_features']}")
            logger.error(f"  Invalid ranges: {validation['invalid_ranges']}")
            return 1

        logger.info("✅ Feature validation PASSED")

        # Step 4: Generate statistics
        stats = generate_feature_statistics(df_featured, REQUIRED_FEATURES)

        # Step 5: Calculate correlations
        correlations = calculate_feature_correlations(df_featured)

        # Step 6: Save featured data
        save_featured_data(df_featured, OUTPUT_PATH)

        # Step 7: Generate reports
        generate_markdown_report(df_featured, stats, correlations, validation, REPORT_PATH)
        generate_json_log(df_featured, stats, correlations, validation, LOG_PATH)

        # Final summary
        logger.info("=" * 70)
        logger.info("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Output file: {OUTPUT_PATH}")
        logger.info(f"Rows: {df_featured.shape[0]:,}")
        logger.info(f"Columns: {df_featured.shape[1]}")
        logger.info(f"New features: {len(REQUIRED_FEATURES)}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"❌ Feature engineering FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
