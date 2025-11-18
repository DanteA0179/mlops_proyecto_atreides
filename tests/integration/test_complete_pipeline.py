"""
Test complete monitoring pipeline with the updated Evidently 0.7.15 API.
This tests the full flow: DriftAnalyzer -> ReportGenerator
"""

import sys

import numpy as np
import pandas as pd

sys.path.append(".")

import logging

from src.monitoring.config import (
    AlertsConfig,
    DriftDetectionConfig,
    DriftThresholds,
    MonitoringConfig,
    PerformanceThresholds,
    ProductionDataConfig,
    ReferenceDataConfig,
    ReportingConfig,
    StatisticalTestThresholds,
)
from src.monitoring.drift_analyzer import DriftAnalyzer
from src.monitoring.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)

print("=== Testing Complete Monitoring Pipeline ===\n")

# Create test configuration using dataclasses
config = MonitoringConfig(
    reference_data=ReferenceDataConfig(
        path="reports/monitoring/reference_data/train_data_sample.parquet",
        size=10000,
        stratify_column="Load_Type",
    ),
    production_data=ProductionDataConfig(source="file", timeframe_days=7, min_samples=1000),
    drift_detection=DriftDetectionConfig(
        thresholds=DriftThresholds(
            drift_score=0.7,
            share_of_drifted_features=0.5,
            feature_drift_score=0.5,
            target_drift_score=0.6,
            prediction_drift_score=0.6,
        ),
        statistical_tests=StatisticalTestThresholds(
            psi_threshold=0.2, ks_pvalue_threshold=0.05, wasserstein_threshold=0.1
        ),
    ),
    performance=PerformanceThresholds(
        rmse_degradation=0.15, mae_degradation=0.15, r2_degradation=0.10
    ),
    reporting=ReportingConfig(
        output_dir="reports/monitoring_test",
        retention_days=90,
        compress_old_reports=True,
        save_json_metrics=True,
        save_csv_history=True,
    ),
    alerts=AlertsConfig(enabled=False),
)

print("1. Creating test data with drift...")
np.random.seed(42)

# Reference data (training-like)
reference_data = pd.DataFrame(
    {
        "NSM": np.random.randint(0, 86400, 200),
        "CO2(tCO2)": np.random.normal(0.5, 0.1, 200),
        "Lagging_Current_Reactive.Power_kVarh": np.random.normal(100, 20, 200),
        "Leading_Current_Reactive_Power_kVarh": np.random.normal(50, 10, 200),
        "Lagging_Current_Power_Factor": np.random.uniform(0.8, 1.0, 200),
        "Leading_Current_Power_Factor": np.random.uniform(0.8, 1.0, 200),
        "WeekStatus": np.random.choice(["Weekday", "Weekend"], 200),
        "Load_Type_Maximum_Load": np.random.choice([0, 1], 200),
        "Load_Type_Medium_Load": np.random.choice([0, 1], 200),
        "Usage_kWh": np.random.normal(100, 20, 200),
        "predictions": np.random.normal(100, 20, 200),
    }
)

# Current data (with drift in some features)
current_data = pd.DataFrame(
    {
        "NSM": np.random.randint(0, 86400, 150),
        "CO2(tCO2)": np.random.normal(0.7, 0.1, 150),  # DRIFTED (mean shifted)
        "Lagging_Current_Reactive.Power_kVarh": np.random.normal(120, 20, 150),  # DRIFTED
        "Leading_Current_Reactive_Power_kVarh": np.random.normal(50, 10, 150),  # No drift
        "Lagging_Current_Power_Factor": np.random.uniform(0.8, 1.0, 150),
        "Leading_Current_Power_Factor": np.random.uniform(0.8, 1.0, 150),
        "WeekStatus": np.random.choice(["Weekday", "Weekend"], 150),
        "Load_Type_Maximum_Load": np.random.choice([0, 1], 150),
        "Load_Type_Medium_Load": np.random.choice([0, 1], 150),
        "Usage_kWh": np.random.normal(100, 20, 150),
        "predictions": np.random.normal(102, 20, 150),  # Slightly different
    }
)

print(f"  Reference data: {len(reference_data)} samples")
print(f"  Current data: {len(current_data)} samples\n")

print("2. Initializing DriftAnalyzer...")
analyzer = DriftAnalyzer(config)

print("\n3. Running drift analysis...")
try:
    snapshot, metrics = analyzer.analyze_drift(reference_data, current_data)
    print("  SUCCESS: Analysis completed")
    print(f"  Snapshot type: {type(snapshot)}")
    print(f"  Metrics extracted: {len(metrics)} keys")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n4. Extracted metrics:")
for key, value in metrics.items():
    if key == "drifted_features":
        print(f"  {key}: {len(value)} features")
        for feat, score in list(value.items())[:3]:
            print(f"    - {feat}: {score:.3f}")
    elif isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\n5. Initializing ReportGenerator...")
generator = ReportGenerator(config)

print("\n6. Saving reports...")
try:
    from datetime import datetime

    saved_paths = generator.save_all(snapshot, metrics, timestamp=datetime.now())
    print("  SUCCESS: Reports saved")
    for report_type, path in saved_paths.items():
        if path:
            print(f"  {report_type}: {path}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n=== Pipeline Test SUCCESSFUL ===")
print(f"\nHTML report available at: {saved_paths.get('html')}")
print("\nYou can open the HTML report in your browser to see the visualizations!")
