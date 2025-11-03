# XGBoost Baseline ejemplo - Evaluation Report

## Test Set Performance

| Metric | Value |
|--------|-------|
| RMSE | 12.8311 |
| MAE | 3.4130 |
| R² | 0.8695 |
| MAPE | 30.96% |
| Max Error | 146.1499 |

## Cross-Validation Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| RMSE | 13.2535 | 0.6661 |
| MAE | 3.5716 | 0.0778 |
| R2 | 0.8603 | 0.0133 |

## Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | CO2(tCO2) | 0.7689 |
| 2 | Lagging_Current_Reactive.Power_kVarh | 0.0922 |
| 3 | Load_Type_Maximum_Load | 0.0897 |
| 4 | Leading_Current_Power_Factor | 0.0142 |
| 5 | Lagging_Current_Power_Factor | 0.0108 |
| 6 | Load_Type_Medium_Load | 0.0094 |
| 7 | NSM | 0.0063 |
| 8 | Leading_Current_Reactive_Power_kVarh | 0.0059 |
| 9 | WeekStatus | 0.0027 |

## Model Performance Assessment

**Status**: Below Target

Model RMSE (12.8311) is 6159.06% above target (0.2050). Further optimization recommended.
