"""
Test MLflow connection to docker-compose service.

This script verifies that MLflow is accessible and functional.
"""

import sys

import mlflow


def test_mlflow_connection():
    """Test connection to MLflow tracking server."""

    # MLflow tracking URI from docker-compose
    tracking_uri = "http://localhost:5000"

    print(f"Testing MLflow connection to: {tracking_uri}")

    try:
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Get or create test experiment
        experiment_name = "test_connection"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        # Start a test run
        with mlflow.start_run(experiment_id=experiment_id, run_name="connection_test"):
            # Log a test parameter
            mlflow.log_param("test_param", "test_value")

            # Log a test metric
            mlflow.log_metric("test_metric", 42.0)

            # Get run info
            run = mlflow.active_run()
            print(f"Successfully created run: {run.info.run_id}")
            print(f"Run URL: {tracking_uri}/#/experiments/{experiment_id}/runs/{run.info.run_id}")

        print("\nMLflow connection test PASSED")
        print(f"MLflow UI available at: {tracking_uri}")
        return True

    except Exception as e:
        print("\nMLflow connection test FAILED")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = test_mlflow_connection()
    sys.exit(0 if success else 1)
