# CatBoost Baseline test_v1 - Evaluation Report

## Test Set Performance

| Metric | Value |
|--------|-------|
| RMSE | 12.9211 |
| MAE | 3.6660 |
| R² | 0.8677 |
| MAPE | 31.83% |
| Max Error | 146.7422 |

## Cross-Validation Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| RMSE | 13.3866 | 0.6192 |
| MAE | 3.8530 | 0.0374 |
| R2 | 0.8575 | 0.0127 |

## Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | CO2(tCO2) | 42.9842 |
| 2 | Lagging_Current_Reactive.Power_kVarh | 19.2786 |
| 3 | Lagging_Current_Power_Factor | 12.9516 |
| 4 | NSM | 8.8566 |
| 5 | Leading_Current_Reactive_Power_kVarh | 5.8281 |
| 6 | Leading_Current_Power_Factor | 5.0670 |
| 7 | WeekStatus | 2.0554 |
| 8 | Load_Type_Medium_Load | 1.9736 |
| 9 | Load_Type_Maximum_Load | 1.0049 |

## Model Performance Assessment

**Status**: Below Target

Model RMSE (12.9211) is 6203.00% above target (0.2050). Further optimization recommended.
