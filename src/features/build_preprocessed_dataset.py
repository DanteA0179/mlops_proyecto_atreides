"""
Build Preprocessed Dataset - US-012.

This script performs the complete preprocessing pipeline:
1. Load featured dataset
2. Split into train/val/test
3. Fit preprocessing pipeline on train
4. Transform all splits
5. Save preprocessed datasets
6. Save pipeline for inference
7. Generate reports

Usage:
    python src/features/build_preprocessed_dataset.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from src.features.preprocessing import PreprocessingPipeline
from src.utils.preprocessing_utils import (
    analyze_categorical_cardinality,
    calculate_scaling_statistics,
    identify_feature_types,
    validate_preprocessing_config,
)
from src.utils.split_data import (
    get_split_statistics,
    stratified_train_val_test_split,
    validate_splits,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Configuration
CONFIG = {
    "input_data": "data/processed/steel_featured.parquet",
    "output_dir": "data/processed",
    "models_dir": "models/preprocessing",
    "reports_dir": "reports",
    "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
    "target_col": "Usage_kWh",
    "numeric_features": [
        "NSM",
        "CO2(tCO2)",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
    ],
    "categorical_features": ["Load_Type"],
    "exclude_from_scaling": [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
    ],
    "binary_features": {"WeekStatus": {"Weekday": 0, "Weekend": 1}},
    "stratification_bins": 10,
    "random_state": 42,
}


def setup_directories():
    """Create necessary directories."""
    dirs = [
        Path(CONFIG["output_dir"]),
        Path(CONFIG["models_dir"]),
        Path(CONFIG["reports_dir"]),
        Path("logs"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directories setup complete")


def load_data() -> pl.DataFrame:
    """
    Load featured dataset.

    Returns
    -------
    pl.DataFrame
        Featured dataset
    """
    logger.info(f"Loading data from {CONFIG['input_data']}")
    df = pl.read_parquet(CONFIG["input_data"])
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def perform_eda(df: pl.DataFrame) -> dict:
    """
    Perform exploratory data analysis before preprocessing.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe

    Returns
    -------
    dict
        EDA results
    """
    logger.info("Performing EDA...")

    eda_results = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns,
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=True)},
    }

    # Analyze feature types
    feature_types = identify_feature_types(
        df, exclude_cols=[CONFIG["target_col"], "date", "day_of_week"]
    )
    eda_results["feature_types"] = feature_types
    logger.info(f"Numeric features: {len(feature_types['numeric'])}")
    logger.info(f"Categorical features: {len(feature_types['categorical'])}")

    # Analyze categorical cardinality
    categorical_analysis = {}
    for cat_feat in CONFIG["categorical_features"]:
        analysis = analyze_categorical_cardinality(df, cat_feat)
        categorical_analysis[cat_feat] = analysis
        logger.info(
            f"{cat_feat}: {analysis['n_categories']} categories, "
            f"OHE dims: {analysis['encoding_size']}"
        )
    eda_results["categorical_analysis"] = categorical_analysis

    # Calculate statistics for numeric features
    numeric_stats = calculate_scaling_statistics(df, CONFIG["numeric_features"])
    eda_results["numeric_statistics"] = numeric_stats

    return eda_results


def split_data(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data into train/val/test.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe

    Returns
    -------
    tuple of pl.DataFrame
        (df_train, df_val, df_test)
    """
    logger.info("Splitting data...")
    logger.info(
        f"Ratios: train={CONFIG['split_ratio']['train']}, "
        f"val={CONFIG['split_ratio']['val']}, "
        f"test={CONFIG['split_ratio']['test']}"
    )

    df_train, df_val, df_test = stratified_train_val_test_split(
        df=df,
        target_col=CONFIG["target_col"],
        train_size=CONFIG["split_ratio"]["train"],
        val_size=CONFIG["split_ratio"]["val"],
        test_size=CONFIG["split_ratio"]["test"],
        n_bins=CONFIG["stratification_bins"],
        random_state=CONFIG["random_state"],
    )

    logger.info(f"Train: {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
    logger.info(f"Val: {len(df_val):,} rows ({len(df_val)/len(df)*100:.1f}%)")
    logger.info(f"Test: {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%)")

    return df_train, df_val, df_test


def validate_split_quality(
    df_train: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame
) -> dict:
    """
    Validate split quality.

    Parameters
    ----------
    df_train : pl.DataFrame
        Training set
    df_val : pl.DataFrame
        Validation set
    df_test : pl.DataFrame
        Test set

    Returns
    -------
    dict
        Validation results
    """
    logger.info("Validating split quality...")

    validation = validate_splits(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        target_col=CONFIG["target_col"],
    )

    if validation["valid"]:
        logger.info("Split validation PASSED")
    else:
        logger.warning("Split validation issues detected:")
        for issue in validation["issues"]:
            logger.warning(f"  - {issue}")

    # Get statistics
    statistics = get_split_statistics(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        numeric_cols=[CONFIG["target_col"]],
    )

    return {"validation": validation, "statistics": statistics}


def fit_preprocessing_pipeline(df_train: pl.DataFrame) -> PreprocessingPipeline:
    """
    Fit preprocessing pipeline on training data.

    Parameters
    ----------
    df_train : pl.DataFrame
        Training data

    Returns
    -------
    PreprocessingPipeline
        Fitted pipeline
    """
    logger.info("Fitting preprocessing pipeline...")

    # Validate configuration
    validation = validate_preprocessing_config(
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        df=df_train,
    )

    if not validation["valid"]:
        logger.error("Invalid preprocessing configuration:")
        for key, value in validation.items():
            if value and key != "valid":
                logger.error(f"  {key}: {value}")
        raise ValueError("Invalid preprocessing configuration")

    logger.info("Configuration validation passed")

    # Create and fit pipeline
    pipeline = PreprocessingPipeline(
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        exclude_from_scaling=CONFIG["exclude_from_scaling"],
        binary_features=CONFIG["binary_features"],
        target_col=CONFIG["target_col"],
        drop_ohe="first",
    )

    pipeline.fit(df_train)

    logger.info("Pipeline fitted successfully")
    logger.info(f"Input features: {len(pipeline.feature_names_in_)}")
    logger.info(f"Output features: {len(pipeline.feature_names_out_)}")

    return pipeline


def transform_splits(
    pipeline: PreprocessingPipeline,
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Transform all splits with fitted pipeline.

    Parameters
    ----------
    pipeline : PreprocessingPipeline
        Fitted pipeline
    df_train : pl.DataFrame
        Training data
    df_val : pl.DataFrame
        Validation data
    df_test : pl.DataFrame
        Test data

    Returns
    -------
    tuple of pl.DataFrame
        (X_train, X_val, X_test) - preprocessed features
    """
    logger.info("Transforming splits...")

    X_train = pipeline.transform(df_train)
    logger.info(f"Train transformed: {X_train.shape}")

    X_val = pipeline.transform(df_val)
    logger.info(f"Val transformed: {X_val.shape}")

    X_test = pipeline.transform(df_test)
    logger.info(f"Test transformed: {X_test.shape}")

    return X_train, X_val, X_test


def extract_targets(
    df_train: pl.DataFrame, df_val: pl.DataFrame, df_test: pl.DataFrame
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """
    Extract target variable from each split.

    Parameters
    ----------
    df_train : pl.DataFrame
        Training data
    df_val : pl.DataFrame
        Validation data
    df_test : pl.DataFrame
        Test data

    Returns
    -------
    tuple of pl.Series
        (y_train, y_val, y_test)
    """
    target_col = CONFIG["target_col"]
    return df_train[target_col], df_val[target_col], df_test[target_col]


def save_preprocessed_data(
    X_train: pl.DataFrame,
    X_val: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.Series,
    y_val: pl.Series,
    y_test: pl.Series,
):
    """
    Save preprocessed datasets to disk.

    Parameters
    ----------
    X_train : pl.DataFrame
        Training features
    X_val : pl.DataFrame
        Validation features
    X_test : pl.DataFrame
        Test features
    y_train : pl.Series
        Training targets
    y_val : pl.Series
        Validation targets
    y_test : pl.Series
        Test targets
    """
    logger.info("Saving preprocessed datasets...")

    output_dir = Path(CONFIG["output_dir"])

    # Save features + target together
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        df_combined = X.with_columns(y.alias(CONFIG["target_col"]))
        filepath = output_dir / f"steel_preprocessed_{split_name}.parquet"
        df_combined.write_parquet(filepath)
        logger.info(f"Saved {split_name}: {filepath}")


def save_pipeline(pipeline: PreprocessingPipeline):
    """
    Save preprocessing pipeline to disk.

    Parameters
    ----------
    pipeline : PreprocessingPipeline
        Fitted pipeline
    """
    logger.info("Saving preprocessing pipeline...")

    models_dir = Path(CONFIG["models_dir"])
    filepath = models_dir / "preprocessing_pipeline.pkl"
    pipeline.save(filepath)

    logger.info(f"Pipeline saved: {filepath}")


def generate_reports(
    eda_results: dict,
    split_validation: dict,
    pipeline: PreprocessingPipeline,
    X_train: pl.DataFrame,
    X_val: pl.DataFrame,
    X_test: pl.DataFrame,
):
    """
    Generate preprocessing reports.

    Parameters
    ----------
    eda_results : dict
        EDA results
    split_validation : dict
        Split validation results
    pipeline : PreprocessingPipeline
        Fitted pipeline
    X_train : pl.DataFrame
        Preprocessed training features
    X_val : pl.DataFrame
        Preprocessed validation features
    X_test : pl.DataFrame
        Preprocessed test features
    """
    logger.info("Generating reports...")

    reports_dir = Path(CONFIG["reports_dir"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # JSON Report
    json_report = {
        "timestamp": timestamp,
        "config": CONFIG,
        "eda": eda_results,
        "split_validation": split_validation,
        "pipeline_summary": pipeline.get_preprocessing_summary(),
        "output_shapes": {
            "train": {"rows": len(X_train), "features": len(X_train.columns)},
            "val": {"rows": len(X_val), "features": len(X_val.columns)},
            "test": {"rows": len(X_test), "features": len(X_test.columns)},
        },
    }

    json_path = reports_dir / "preprocessing_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    logger.info(f"JSON report saved: {json_path}")

    # Markdown Report
    md_report = generate_markdown_report(json_report)
    md_path = reports_dir / "preprocessing_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"Markdown report saved: {md_path}")


def generate_markdown_report(report: dict) -> str:
    """
    Generate markdown report from JSON report.

    Parameters
    ----------
    report : dict
        JSON report data

    Returns
    -------
    str
        Markdown formatted report
    """
    md = f"""# Preprocessing Report - US-012

**Timestamp**: {report['timestamp']}

---

## 1. Configuration

- **Input Data**: `{report['config']['input_data']}`
- **Output Directory**: `{report['config']['output_dir']}`
- **Models Directory**: `{report['config']['models_dir']}`
- **Split Ratio**: Train {report['config']['split_ratio']['train']*100:.0f}% / Val {report['config']['split_ratio']['val']*100:.0f}% / Test {report['config']['split_ratio']['test']*100:.0f}%
- **Random State**: {report['config']['random_state']}

---

## 2. Input Data Summary

- **Total Rows**: {report['eda']['n_rows']:,}
- **Total Columns**: {report['eda']['n_columns']}

### Feature Types

- **Numeric**: {len(report['eda']['feature_types']['numeric'])} features
- **Categorical**: {len(report['eda']['feature_types']['categorical'])} features
- **Boolean**: {len(report['eda']['feature_types']['boolean'])} features

---

## 3. Categorical Analysis

"""

    for feat, analysis in report["eda"]["categorical_analysis"].items():
        md += f"""### {feat}

- **Categories**: {analysis['n_categories']}
- **Encoding Size** (OHE with drop='first'): {analysis['encoding_size']} features
- **Most Common**: {analysis['most_common']} ({analysis['distribution'][analysis['most_common']]*100:.1f}%)
- **Least Common**: {analysis['least_common']} ({analysis['distribution'][analysis['least_common']]*100:.1f}%)

"""

    md += """---

## 4. Data Splitting

"""

    validation = report["split_validation"]["validation"]
    md += f"""### Validation Results

- **Status**: {"✅ PASSED" if validation['valid'] else "⚠️ FAILED"}

"""

    if validation["issues"]:
        md += "**Issues Detected:**\n\n"
        for issue in validation["issues"]:
            md += f"- {issue}\n"
    else:
        md += "No issues detected.\n"

    md += "\n### Split Checks\n\n"
    for check_name, check_result in validation["checks"].items():
        status = "✅" if check_result else "❌"
        md += f"- {status} {check_name}\n"

    md += """

---

## 5. Preprocessing Pipeline

"""

    summary = report["pipeline_summary"]
    md += f"""### Pipeline Configuration

- **Input Features**: {summary['n_features_in']}
- **Output Features**: {summary['n_features_out']}

### Numeric Features ({len(summary['numeric_features'])})

"""
    for feat in summary["numeric_features"]:
        md += f"- `{feat}`\n"

    md += f"""
### Scaled Features ({len(summary['scaled_features'])})

"""
    for feat in summary["scaled_features"]:
        stats = summary["scaling_statistics"][feat]
        md += f"- `{feat}`: μ={stats['mean']:.2f}, σ={stats['std']:.2f}\n"

    md += f"""
### Excluded from Scaling ({len(summary['excluded_from_scaling'])})

"""
    for feat in summary["excluded_from_scaling"]:
        md += f"- `{feat}` (cyclical feature, already normalized)\n"

    md += f"""
### Categorical Features ({len(summary['categorical_features'])})

"""
    for feat in summary["categorical_features"]:
        categories = summary["ohe_categories"][feat]
        md += f"- `{feat}`: {len(categories)} categories → {len(categories)-1} OHE features\n"

    md += """
### Binary Features

"""
    for feat, mapping in summary["binary_features"].items():
        md += f"- `{feat}`: {mapping}\n"

    md += """

---

## 6. Output Datasets

"""

    shapes = report["output_shapes"]
    md += f"""### Train Set

- **Rows**: {shapes['train']['rows']:,}
- **Features**: {shapes['train']['features']}
- **File**: `{report['config']['output_dir']}/steel_preprocessed_train.parquet`

### Validation Set

- **Rows**: {shapes['val']['rows']:,}
- **Features**: {shapes['val']['features']}
- **File**: `{report['config']['output_dir']}/steel_preprocessed_val.parquet`

### Test Set

- **Rows**: {shapes['test']['rows']:,}
- **Features**: {shapes['test']['features']}
- **File**: `{report['config']['output_dir']}/steel_preprocessed_test.parquet`

---

## 7. Serialized Assets

- **Pipeline**: `{report['config']['models_dir']}/preprocessing_pipeline.pkl`
- **JSON Report**: `{report['config']['reports_dir']}/preprocessing_report.json`
- **Markdown Report**: `{report['config']['reports_dir']}/preprocessing_report.md`

---

## 8. Usage Example

```python
from src.features.preprocessing import PreprocessingPipeline
import polars as pl

# Load pipeline
pipeline = PreprocessingPipeline.load('models/preprocessing/preprocessing_pipeline.pkl')

# Load new data
df_new = pl.read_parquet('data/raw/new_data.parquet')

# Transform
X_new = pipeline.transform(df_new)
```

---

**Report Generated**: {report['timestamp']}
"""

    return md


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE - US-012")
    logger.info("=" * 80)

    try:
        # Step 1: Setup
        setup_directories()

        # Step 2: Load data
        df = load_data()

        # Step 3: EDA
        eda_results = perform_eda(df)

        # Step 4: Split data
        df_train, df_val, df_test = split_data(df)

        # Step 5: Validate splits
        split_validation = validate_split_quality(df_train, df_val, df_test)

        # Step 6: Fit preprocessing pipeline
        pipeline = fit_preprocessing_pipeline(df_train)

        # Step 7: Transform all splits
        X_train, X_val, X_test = transform_splits(pipeline, df_train, df_val, df_test)

        # Step 8: Extract targets
        y_train, y_val, y_test = extract_targets(df_train, df_val, df_test)

        # Step 9: Save preprocessed data
        save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)

        # Step 10: Save pipeline
        save_pipeline(pipeline)

        # Step 11: Generate reports
        generate_reports(
            eda_results, split_validation, pipeline, X_train, X_val, X_test
        )

        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
