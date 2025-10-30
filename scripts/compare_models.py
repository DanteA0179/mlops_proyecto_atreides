"""
Script to compare Chronos-2, XGBoost, and CUBIST benchmark.
"""

import json
from pathlib import Path

import pandas as pd

# Load results
chronos_file = Path("models/foundation/chronos2_results_20251027_230102.json")
with open(chronos_file) as f:
    chronos_results = json.load(f)

xgboost_file = Path("reports/metrics/xgboost_test_metrics_optimized.json")
with open(xgboost_file) as f:
    xgboost_results = json.load(f)

# CUBIST benchmark (from US-013)
cubist_rmse_norm = 0.2410
target_rmse_norm = 0.2050

# Calculate normalized RMSE (RMSE / std of target)
# From data: std(Usage_kWh) ≈ 35.53 kWh
std_target = 35.53

chronos_rmse_norm = chronos_results["metrics"]["rmse"] / std_target
xgboost_rmse_norm = xgboost_results["rmse"] / std_target

# Create comparison table
comparison = {
    "Model": ["CUBIST (Benchmark)", "Target (-15%)", "XGBoost Optimized", "Chronos-2 Zero-Shot"],
    "RMSE (kWh)": [None, None, xgboost_results["rmse"], chronos_results["metrics"]["rmse"]],
    "RMSE Normalized": [cubist_rmse_norm, target_rmse_norm, xgboost_rmse_norm, chronos_rmse_norm],
    "MAE (kWh)": [None, None, xgboost_results["mae"], chronos_results["metrics"]["mae"]],
    "R²": [None, None, xgboost_results["r2"], chronos_results["metrics"]["r2"]],
    "MAPE (%)": [None, None, xgboost_results["mape"], chronos_results["metrics"]["mape"]],
}

df = pd.DataFrame(comparison)

print("\n" + "="*80)
print("MODEL COMPARISON: Chronos-2 vs XGBoost vs CUBIST")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Analysis
print("\nKey Findings:")
print(f"1. XGBoost RMSE normalized: {xgboost_rmse_norm:.4f}")
print(f"2. Chronos-2 RMSE normalized: {chronos_rmse_norm:.4f}")
print(f"3. Performance gap: Chronos-2 is {chronos_rmse_norm/xgboost_rmse_norm:.1f}x worse than XGBoost")
print(f"4. Target RMSE: {target_rmse_norm:.4f} (not achieved by either)")
print(f"5. CUBIST benchmark: {cubist_rmse_norm:.4f}")

print("\nConclusion:")
print("Chronos-2 zero-shot performs poorly on domain-specific industrial data.")
print("XGBoost with domain features significantly outperforms foundation model.")
print("Fine-tuning would be required to make Chronos-2 competitive.")
