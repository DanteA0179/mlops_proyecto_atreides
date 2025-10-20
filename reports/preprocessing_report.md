# Preprocessing Report - US-012

**Timestamp**: 2025-10-20 15:44:51

---

## 1. Configuration

- **Input Data**: `data/processed/steel_featured.parquet`
- **Output Directory**: `data/processed`
- **Models Directory**: `models/preprocessing`
- **Split Ratio**: Train 70% / Val 15% / Test 15%
- **Random State**: 42

---

## 2. Input Data Summary

- **Total Rows**: 34,910
- **Total Columns**: 18

### Feature Types

- **Numeric**: 11 features
- **Categorical**: 3 features
- **Boolean**: 1 features

---

## 3. Categorical Analysis

### Load_Type

- **Categories**: 3
- **Encoding Size** (OHE with drop='first'): 2 features
- **Most Common**: Light_Load (52.0%)
- **Least Common**: Maximum_Load (20.5%)

---

## 4. Data Splitting

### Validation Results

- **Status**: ⚠️ FAILED

**Issues Detected:**

- Data leakage detected: train-val overlap=17, train-test overlap=16, val-test overlap=5

### Split Checks

- ✅ sizes
- ✅ no_leakage
- ✅ distributions


---

## 5. Preprocessing Pipeline

### Pipeline Configuration

- **Input Features**: 18
- **Output Features**: 9

### Numeric Features (6)

- `NSM`
- `CO2(tCO2)`
- `Lagging_Current_Reactive.Power_kVarh`
- `Leading_Current_Reactive_Power_kVarh`
- `Lagging_Current_Power_Factor`
- `Leading_Current_Power_Factor`

### Scaled Features (6)

- `NSM`: μ=43003.70, σ=25002.90
- `CO2(tCO2)`: μ=0.01, σ=0.02
- `Lagging_Current_Reactive.Power_kVarh`: μ=13.56, σ=17.29
- `Leading_Current_Reactive_Power_kVarh`: μ=4.08, σ=7.69
- `Lagging_Current_Power_Factor`: μ=80.81, σ=18.97
- `Leading_Current_Power_Factor`: μ=84.32, σ=30.47

### Excluded from Scaling (8)

- `hour_sin` (cyclical feature, already normalized)
- `hour_cos` (cyclical feature, already normalized)
- `day_sin` (cyclical feature, already normalized)
- `day_cos` (cyclical feature, already normalized)
- `week_sin` (cyclical feature, already normalized)
- `week_cos` (cyclical feature, already normalized)
- `month_sin` (cyclical feature, already normalized)
- `month_cos` (cyclical feature, already normalized)

### Categorical Features (1)

- `Load_Type`: 3 categories → 2 OHE features

### Binary Features

- `WeekStatus`: {'Weekday': 0, 'Weekend': 1}


---

## 6. Output Datasets

### Train Set

- **Rows**: 24,437
- **Features**: 9
- **File**: `data/processed/steel_preprocessed_train.parquet`

### Validation Set

- **Rows**: 5,236
- **Features**: 9
- **File**: `data/processed/steel_preprocessed_val.parquet`

### Test Set

- **Rows**: 5,237
- **Features**: 9
- **File**: `data/processed/steel_preprocessed_test.parquet`

---

## 7. Serialized Assets

- **Pipeline**: `models/preprocessing/preprocessing_pipeline.pkl`
- **JSON Report**: `reports/preprocessing_report.json`
- **Markdown Report**: `reports/preprocessing_report.md`

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

**Report Generated**: 2025-10-20 15:44:51
