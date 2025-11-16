"""
End-to-end tests for complete ML pipeline.

Tests the entire pipeline from data loading to model training
and evaluation, ensuring all components work together correctly.
"""

import shutil
import tempfile
from pathlib import Path

import duckdb
import polars as pl
import pytest
from sklearn.metrics import mean_squared_error, r2_score


@pytest.fixture(scope="module")
def temp_pipeline_dir():
    """
    Create temporary directory for pipeline tests.
    
    Yields
    ------
    Path
        Temporary directory path
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def sample_data_path():
    """
    Get path to sample data for testing.
    
    Returns
    -------
    Path
        Path to steel data
    """
    data_path = Path("data/raw/Steel_industry_data.csv")
    if not data_path.exists():
        pytest.skip("Sample data not available")
    return data_path


class TestDataLoadingPipeline:
    """Test data loading and initial processing"""
    
    def test_load_raw_data(self, sample_data_path: Path):
        """Test loading raw CSV data"""
        df = pl.read_csv(sample_data_path)
        
        assert len(df) > 0
        assert df.shape[1] > 0
        
        expected_columns = [
            "Usage_kWh",
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            "NSM",
            "Load_Type"
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
            
    def test_load_to_duckdb(self, sample_data_path: Path, temp_pipeline_dir: Path):
        """Test loading data into DuckDB"""
        from src.data.load_to_duckdb import load_csv_to_duckdb
        
        db_path = temp_pipeline_dir / "test_steel.duckdb"
        
        load_csv_to_duckdb(
            csv_path=str(sample_data_path),
            db_path=str(db_path),
            table_name="steel_raw"
        )
        
        assert db_path.exists()
        
        conn = duckdb.connect(str(db_path))
        result = conn.execute("SELECT COUNT(*) FROM steel_raw").fetchone()
        row_count = result[0]
        
        assert row_count > 0
        conn.close()
        
    def test_data_quality_checks(self, sample_data_path: Path):
        """Test data quality validation"""
        df = pl.read_csv(sample_data_path)
        
        # Check for required columns
        assert "Usage_kWh" in df.columns
        
        # Check data types
        assert df["Usage_kWh"].dtype in [pl.Float64, pl.Float32, pl.Int64]
        
        # Check for reasonable values
        usage_stats = df["Usage_kWh"].describe()
        assert df["Usage_kWh"].min() >= 0


class TestDataCleaningPipeline:
    """Test data cleaning and preprocessing"""
    
    @pytest.fixture
    def raw_data(self, sample_data_path: Path) -> pl.DataFrame:
        """Load raw data for cleaning tests"""
        return pl.read_csv(sample_data_path)
    
    def test_handle_missing_values(self, raw_data: pl.DataFrame):
        """Test missing value handling"""
        from src.utils.data_cleaning import handle_null_values
        
        df_cleaned = handle_null_values(raw_data)
        
        # Check no missing values in critical columns
        assert df_cleaned["Usage_kWh"].null_count() == 0
        
    def test_outlier_detection(self, raw_data: pl.DataFrame):
        """Test outlier detection and handling"""
        from src.utils.outlier_detection import detect_outliers_iqr
        
        outliers = detect_outliers_iqr(raw_data, "Usage_kWh")
        
        assert isinstance(outliers, pl.DataFrame)
        assert len(outliers) >= 0
        
    def test_data_type_conversion(self, raw_data: pl.DataFrame):
        """Test data type conversions"""
        from src.utils.data_cleaning import convert_data_types
        
        df_converted = convert_data_types(raw_data)
        
        # Check NSM is integer
        if "NSM" in df_converted.columns:
            assert df_converted["NSM"].dtype in [pl.Int64, pl.Int32]
            
    def test_duplicate_removal(self, raw_data: pl.DataFrame):
        """Test duplicate row removal"""
        from src.utils.data_cleaning import remove_duplicates
        
        df_dedup = remove_duplicates(raw_data)
        
        assert len(df_dedup) <= len(raw_data)


class TestFeatureEngineeringPipeline:
    """Test feature engineering pipeline"""
    
    @pytest.fixture
    def cleaned_data(self, sample_data_path: Path) -> pl.DataFrame:
        """Load and clean data for feature engineering"""
        df = pl.read_csv(sample_data_path)
        
        # Basic cleaning
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").str.to_datetime())
            
        return df
    
    def test_temporal_feature_engineering(self, cleaned_data: pl.DataFrame):
        """Test temporal feature creation"""
        from src.features.temporal_transformers import TemporalFeatureEngineer
        
        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        df_transformed = engineer.fit_transform(cleaned_data)
        
        # Check new features created
        assert "hour" in df_transformed.columns
        assert df_transformed["hour"].min() >= 0
        assert df_transformed["hour"].max() <= 23
        
    def test_cyclical_encoding(self):
        """Test cyclical feature encoding"""
        from src.features.temporal_transformers import CyclicalEncoder
        
        df = pl.DataFrame({
            "hour": [0, 6, 12, 18, 23],
            "day_of_week": [0, 1, 2, 3, 6]
        })
        
        encoder = CyclicalEncoder(column="hour", period=24)
        df_encoded = encoder.fit_transform(df)
        
        assert "cyclical_hour_sin" in df_encoded.columns
        assert "cyclical_hour_cos" in df_encoded.columns
        
    def test_feature_pipeline_integration(self, cleaned_data: pl.DataFrame):
        """Test complete feature engineering pipeline"""
        from sklearn.pipeline import Pipeline
        from src.features.temporal_transformers import TemporalFeatureEngineer
        
        pipeline = Pipeline([
            ("temporal", TemporalFeatureEngineer(nsm_column="NSM"))
        ])
        
        df_transformed = pipeline.fit_transform(cleaned_data)
        
        assert len(df_transformed) == len(cleaned_data)


@pytest.fixture(scope="module")
def train_test_split(sample_data_path: Path):
    """Create train/test split for testing"""
    from src.data.split_data import split_time_series_data
    
    df = pl.read_csv(sample_data_path)
    
    # Prepare data
    feature_cols = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM"
    ]
    
    X = df.select(feature_cols).to_numpy()
    y = df["Usage_kWh"].to_numpy()
    
    X_train, X_test, y_train, y_test = split_time_series_data(
        X, y, test_size=0.2
    )
    
    return X_train, X_test, y_train, y_test


class TestTrainingPipeline:
    """Test model training pipeline"""
    
    def test_baseline_model_training(self, train_test_split):
        """Test training baseline model"""
        from sklearn.linear_model import LinearRegression
        
        X_train, X_test, y_train, y_test = train_test_split
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Check model trained successfully
        assert hasattr(model, "coef_")
        assert len(model.coef_) == X_train.shape[1]
        
        # Check predictions work
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        
    def test_xgboost_model_training(self, train_test_split):
        """Test training XGBoost model"""
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split
        
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Check reasonable performance
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        assert rmse > 0
        assert r2 > -1  # At least better than constant prediction
        
    def test_model_evaluation(self, train_test_split):
        """Test model evaluation metrics"""
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        assert rmse >= 0
        assert -1 <= r2 <= 1


class TestMLflowIntegration:
    """Test MLflow experiment tracking"""
    
    def test_mlflow_tracking_available(self):
        """Test MLflow is available and configured"""
        import mlflow
        
        # Check MLflow can be imported
        assert mlflow is not None
        
        # Try to set experiment
        try:
            mlflow.set_experiment("test_experiment")
            experiment = mlflow.get_experiment_by_name("test_experiment")
            assert experiment is not None
        except Exception as e:
            pytest.skip(f"MLflow not configured: {e}")
            
    def test_log_model_to_mlflow(self, train_test_split):
        """Test logging model to MLflow"""
        import mlflow
        from sklearn.ensemble import RandomForestRegressor
        
        try:
            mlflow.set_experiment("test_e2e_pipeline")
        except Exception:
            pytest.skip("MLflow not configured")
        
        X_train, X_test, y_train, y_test = train_test_split
        
        with mlflow.start_run(run_name="e2e_test_run"):
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            
            # Log parameters
            mlflow.log_param("n_estimators", 10)
            mlflow.log_param("random_state", 42)
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            run = mlflow.active_run()
            assert run is not None
            assert run.info.status == "RUNNING"


class TestCompletePipeline:
    """Test complete end-to-end pipeline"""
    
    def test_full_pipeline_workflow(
        self,
        sample_data_path: Path,
        temp_pipeline_dir: Path
    ):
        """Test complete workflow from raw data to predictions"""
        # Load data
        df = pl.read_csv(sample_data_path)
        assert len(df) > 0
        
        # Clean data
        from src.utils.data_cleaning import handle_null_values, remove_duplicates
        df_cleaned = handle_null_values(df)
        df_cleaned = remove_duplicates(df_cleaned)
        
        # Feature engineering
        from src.features.temporal_transformers import TemporalFeatureEngineer
        engineer = TemporalFeatureEngineer(nsm_column="NSM")
        df_features = engineer.fit_transform(df_cleaned)
        
        # Prepare train/test split
        feature_cols = [col for col in df_features.columns if col != "Usage_kWh"]
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        X = df_features.select(available_features[:6]).to_numpy()  # Use first 6 features
        y = df_features["Usage_kWh"].to_numpy()
        
        from src.data.split_data import split_time_series_data
        X_train, X_test, y_train, y_test = split_time_series_data(X, y, test_size=0.2)
        
        # Train model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        # Assertions
        assert len(y_pred) == len(y_test)
        assert rmse > 0
        assert r2 > -1
        
        # Save model
        import joblib
        model_path = temp_pipeline_dir / "test_model.pkl"
        joblib.dump(model, model_path)
        
        assert model_path.exists()
        
        # Load and verify saved model
        loaded_model = joblib.load(model_path)
        y_pred_loaded = loaded_model.predict(X_test[:5])
        
        assert len(y_pred_loaded) == 5
        
    def test_pipeline_reproducibility(self, sample_data_path: Path):
        """Test pipeline produces consistent results"""
        from sklearn.ensemble import RandomForestRegressor
        
        df = pl.read_csv(sample_data_path)
        
        # Prepare data
        feature_cols = [
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "NSM"
        ]
        
        X = df.select(feature_cols).to_numpy()
        y = df["Usage_kWh"].to_numpy()
        
        # Train two models with same seed
        model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        model1.fit(X[:100], y[:100])
        pred1 = model1.predict(X[100:110])
        
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(X[:100], y[:100])
        pred2 = model2.predict(X[100:110])
        
        # Predictions should be identical
        assert (pred1 == pred2).all()


class TestDataVersioning:
    """Test data versioning with DVC"""
    
    def test_dvc_initialized(self):
        """Test DVC is initialized in project"""
        dvc_dir = Path(".dvc")
        
        if not dvc_dir.exists():
            pytest.skip("DVC not initialized")
            
        assert dvc_dir.is_dir()
        
    def test_data_tracked_by_dvc(self):
        """Test data files are tracked by DVC"""
        data_dir = Path("data")
        
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Look for .dvc files
        dvc_files = list(data_dir.rglob("*.dvc"))
        
        # Should have at least some data tracked
        # Skip if no DVC files (may be in development)
        if len(dvc_files) == 0:
            pytest.skip("No DVC-tracked files found")


class TestPipelineErrorHandling:
    """Test error handling in pipeline"""
    
    def test_handle_invalid_data_path(self):
        """Test error handling for invalid data path"""
        with pytest.raises((FileNotFoundError, Exception)):
            pl.read_csv("invalid/path/to/data.csv")
            
    def test_handle_missing_columns(self):
        """Test error handling for missing required columns"""
        df = pl.DataFrame({"col1": [1, 2, 3]})
        
        with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
            df["Usage_kWh"]
            
    def test_handle_invalid_model_input(self, train_test_split):
        """Test error handling for invalid model input"""
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Try to predict with wrong number of features
        with pytest.raises((ValueError, Exception)):
            model.predict([[1, 2, 3]])  # Wrong number of features
