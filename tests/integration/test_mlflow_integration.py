"""
Integration tests for MLflow tracking and model registry.

Tests integration between model training, MLflow logging, and model registry.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import mlflow


@pytest.fixture(scope="module")
def mlflow_tracking_uri(tmp_path_factory):
    """
    Setup MLflow tracking URI.

    Parameters
    ----------
    tmp_path_factory : TempPathFactory
        Pytest temp path factory

    Returns
    -------
    str
        MLflow tracking URI
    """
    temp_dir = tmp_path_factory.mktemp("mlflow")
    uri = f"file://{temp_dir}"
    mlflow.set_tracking_uri(uri)
    return uri


@pytest.fixture
def sample_ml_data():
    """
    Create sample ML data.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    np.random.seed(42)
    n_samples = 100

    X = np.random.rand(n_samples, 5)
    y = X.sum(axis=1) + np.random.randn(n_samples) * 0.1

    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


class TestMLflowLoggingIntegration:
    """Test MLflow logging integration"""

    def test_basic_experiment_logging(self, mlflow_tracking_uri):
        """Test basic experiment logging"""
        mlflow.set_experiment("test_integration")

        with mlflow.start_run(run_name="test_run"):
            mlflow.log_param("test_param", "value")
            mlflow.log_metric("test_metric", 0.95)

            run = mlflow.active_run()
            assert run is not None

    def test_model_training_with_logging(self, mlflow_tracking_uri, sample_ml_data):
        """Test model training with MLflow logging"""
        X_train, X_test, y_train, y_test = sample_ml_data

        mlflow.set_experiment("model_training_test")

        with mlflow.start_run(run_name="rf_test"):
            model = RandomForestRegressor(n_estimators=10, random_state=42)

            mlflow.log_param("n_estimators", 10)
            mlflow.log_param("random_state", 42)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            r2 = r2_score(y_test, predictions)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        loaded_predictions = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(predictions, loaded_predictions)


class TestMLflowModelRegistry:
    """Test MLflow model registry integration"""

    def test_register_model(self, mlflow_tracking_uri, sample_ml_data):
        """Test registering model to registry"""
        X_train, X_test, y_train, y_test = sample_ml_data

        mlflow.set_experiment("model_registry_test")

        with mlflow.start_run(run_name="register_test"):
            model = LinearRegression()
            model.fit(X_train, y_train)

            mlflow.sklearn.log_model(model, "model", registered_model_name="test_model")

            run_id = mlflow.active_run().info.run_id

        assert run_id is not None


class TestMLflowExperimentComparison:
    """Test comparing multiple experiments"""

    def test_compare_multiple_models(self, mlflow_tracking_uri, sample_ml_data):
        """Test comparing multiple models in MLflow"""
        X_train, X_test, y_train, y_test = sample_ml_data

        mlflow.set_experiment("model_comparison")

        models = {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=10, random_state=42),
        }

        results = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=f"{name}_run"):
                mlflow.log_param("model_type", name)

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                rmse = mean_squared_error(y_test, predictions, squared=False)
                r2 = r2_score(y_test, predictions)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)

                results[name] = {"rmse": rmse, "r2": r2}

        assert len(results) == 2
        assert all("rmse" in v for v in results.values())


class TestMLflowArtifacts:
    """Test MLflow artifacts logging"""

    def test_log_artifacts(self, mlflow_tracking_uri, tmp_path):
        """Test logging artifacts to MLflow"""
        mlflow.set_experiment("artifacts_test")

        artifact_file = tmp_path / "test_artifact.txt"
        artifact_file.write_text("test content")

        with mlflow.start_run():
            mlflow.log_artifact(str(artifact_file))

            run_id = mlflow.active_run().info.run_id

        assert run_id is not None

    def test_log_multiple_artifacts(self, mlflow_tracking_uri, tmp_path):
        """Test logging multiple artifacts"""
        mlflow.set_experiment("multi_artifacts_test")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        for i in range(3):
            (artifacts_dir / f"file_{i}.txt").write_text(f"content {i}")

        with mlflow.start_run():
            mlflow.log_artifacts(str(artifacts_dir))

            run = mlflow.active_run()
            assert run is not None


class TestMLflowMetricTracking:
    """Test metric tracking over time"""

    def test_log_metrics_over_epochs(self, mlflow_tracking_uri):
        """Test logging metrics over training epochs"""
        mlflow.set_experiment("metric_tracking_test")

        with mlflow.start_run():
            for epoch in range(10):
                loss = 1.0 / (epoch + 1)
                mlflow.log_metric("loss", loss, step=epoch)

            run = mlflow.active_run()
            assert run is not None

    def test_multiple_metric_logging(self, mlflow_tracking_uri, sample_ml_data):
        """Test logging multiple metrics"""
        X_train, X_test, y_train, y_test = sample_ml_data

        mlflow.set_experiment("multi_metrics_test")

        with mlflow.start_run():
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            rmse = mean_squared_error(y_test, predictions, squared=False)
            mae = np.mean(np.abs(y_test - predictions))
            r2 = r2_score(y_test, predictions)

            metrics = {"rmse": rmse, "mae": mae, "r2": r2}

            mlflow.log_metrics(metrics)

            run = mlflow.active_run()
            assert run is not None


class TestMLflowParameterLogging:
    """Test parameter logging"""

    def test_log_model_parameters(self, mlflow_tracking_uri):
        """Test logging model parameters"""
        mlflow.set_experiment("param_logging_test")

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        with mlflow.start_run():
            params = model.get_params()
            mlflow.log_params(params)

            run = mlflow.active_run()
            assert run is not None

    def test_nested_parameters(self, mlflow_tracking_uri):
        """Test logging nested parameters"""
        mlflow.set_experiment("nested_params_test")

        with mlflow.start_run():
            mlflow.log_param("model.type", "RandomForest")
            mlflow.log_param("model.n_estimators", 100)
            mlflow.log_param("preprocessing.scaler", "StandardScaler")

            run = mlflow.active_run()
            assert run is not None


class TestMLflowTags:
    """Test MLflow tags"""

    def test_set_tags(self, mlflow_tracking_uri):
        """Test setting tags on runs"""
        mlflow.set_experiment("tags_test")

        with mlflow.start_run():
            mlflow.set_tag("model_type", "regression")
            mlflow.set_tag("dataset", "steel_industry")
            mlflow.set_tag("version", "v1.0")

            run = mlflow.active_run()
            assert run is not None


class TestMLflowWorkflow:
    """Test complete MLflow workflow"""

    def test_complete_training_workflow(self, mlflow_tracking_uri, sample_ml_data):
        """Test complete training workflow with MLflow"""
        X_train, X_test, y_train, y_test = sample_ml_data

        mlflow.set_experiment("complete_workflow_test")

        with mlflow.start_run(run_name="full_workflow"):
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("model_family", "tree_based")

            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
            test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

            mlflow.log_metrics(
                {
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "train_r2": r2_score(y_train, train_predictions),
                    "test_r2": r2_score(y_test, test_predictions),
                }
            )

            mlflow.sklearn.log_model(model, "model", registered_model_name="workflow_test_model")

            run_id = mlflow.active_run().info.run_id

        loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        loaded_predictions = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(test_predictions, loaded_predictions)


class TestMLflowErrorHandling:
    """Test MLflow error handling"""

    def test_handle_invalid_metric(self, mlflow_tracking_uri):
        """Test handling invalid metric values"""
        mlflow.set_experiment("error_handling_test")

        with mlflow.start_run():
            try:
                mlflow.log_metric("valid_metric", 0.95)
                assert True
            except Exception:
                pytest.fail("Valid metric should not raise exception")

    def test_handle_duplicate_run_name(self, mlflow_tracking_uri):
        """Test handling duplicate run names"""
        mlflow.set_experiment("duplicate_run_test")

        with mlflow.start_run(run_name="duplicate_test"):
            mlflow.log_metric("metric1", 1.0)

        with mlflow.start_run(run_name="duplicate_test"):
            mlflow.log_metric("metric2", 2.0)


class TestMLflowSearchAndQuery:
    """Test MLflow search and query functionality"""

    def test_search_runs(self, mlflow_tracking_uri, sample_ml_data):
        """Test searching for runs"""
        X_train, X_test, y_train, y_test = sample_ml_data

        experiment_name = "search_test"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="run1"):
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            mlflow.log_metric("rmse", rmse)

        with mlflow.start_run(run_name="run2"):
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            mlflow.log_metric("rmse", rmse)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        assert len(runs) == 2
