# XGBoost Baseline optimized - Evaluation Report

## Test Set Performance

| Metric | Value |
|--------|-------|
| RMSE | 12.8425 |
| MAE | 3.5331 |
| R² | 0.8693 |
| MAPE | 31.46% |
| Max Error | 146.1853 |

## Cross-Validation Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| RMSE | 13.2775 | 0.6654 |
| MAE | 3.6868 | 0.0726 |
| R2 | 0.8598 | 0.0133 |

## Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | CO2(tCO2) | 0.7031 |
| 2 | Lagging_Current_Reactive.Power_kVarh | 0.1424 |
| 3 | Leading_Current_Power_Factor | 0.0440 |
| 4 | Lagging_Current_Power_Factor | 0.0394 |
| 5 | Load_Type_Maximum_Load | 0.0388 |
| 6 | Leading_Current_Reactive_Power_kVarh | 0.0112 |
| 7 | NSM | 0.0099 |
| 8 | Load_Type_Medium_Load | 0.0071 |
| 9 | WeekStatus | 0.0041 |

## Model Performance Assessment

**Status**: Below Target

Model RMSE (12.8425) is 6164.64% above target (0.2050). Further optimization recommended.
