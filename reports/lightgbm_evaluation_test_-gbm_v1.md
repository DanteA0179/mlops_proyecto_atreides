# LightGBM Baseline test_-gbm_v1 - Evaluation Report

## Test Set Performance

| Metric | Value |
|--------|-------|
| RMSE | 12.9521 |
| MAE | 3.5672 |
| R² | 0.8671 |
| MAPE | 34.96% |
| Max Error | 147.5008 |

## Cross-Validation Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| RMSE | 13.4670 | 0.5917 |
| MAE | 3.9324 | 0.0309 |
| R2 | 0.8558 | 0.0122 |

## Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Lagging_Current_Power_Factor | 2361.0000 |
| 2 | Lagging_Current_Reactive.Power_kVarh | 2256.0000 |
| 3 | NSM | 1776.0000 |
| 4 | Leading_Current_Reactive_Power_kVarh | 1209.0000 |
| 5 | Leading_Current_Power_Factor | 844.0000 |
| 6 | CO2(tCO2) | 543.0000 |
| 7 | WeekStatus | 132.0000 |
| 8 | Load_Type_Medium_Load | 110.0000 |
| 9 | Load_Type_Maximum_Load | 47.0000 |

## Model Performance Assessment

**Status**: Below Target

Model RMSE (12.9521) is 6218.09% above target (0.2050). Further optimization recommended.
