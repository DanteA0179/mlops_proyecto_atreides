"""
Dagster Jobs for Training Pipeline.

Jobs are the equivalent of Prefect flows - they define the execution graph
by connecting ops together.
"""

from dagster import job

from .ops import (
    check_threshold_op,
    dvc_add_op,
    evaluate_model_op,
    load_config_op,
    load_data_op,
    log_mlflow_op,
    save_artifacts_op,
    send_notification_op,
    train_model_op,
    validate_data_op,
)


@job(
    name="training_pipeline",
    description="End-to-end ML training pipeline with MLflow and DVC",
    tags={"pipeline": "training", "model": "xgboost"},
)
def training_job():
    """
    Training pipeline job.

    This job orchestrates the complete training workflow:
    1. Load configuration
    2. Load preprocessed data
    3. Validate data quality
    4. Train model with GPU fallback
    5. Evaluate model
    6. Check performance threshold
    7. Log to MLflow
    8. Save artifacts
    9. Add to DVC
    10. Send notification

    All ops reuse existing utilities from previous User Stories.
    """
    # Step 1: Load config
    config = load_config_op()

    # Step 2: Load data
    data = load_data_op(config)

    # Step 3: Validate data
    validation = validate_data_op(data)

    # Step 4: Train model
    model = train_model_op(data, config)

    # Step 5: Evaluate model
    metrics = evaluate_model_op(model, data, config)

    # Step 6: Check threshold
    threshold_result = check_threshold_op(metrics, config)

    # Step 7: Log to MLflow
    run_id = log_mlflow_op(model, metrics, config)

    # Step 8: Save artifacts
    model_path = save_artifacts_op(model, metrics, config)

    # Step 9: DVC add
    dvc_file = dvc_add_op(model_path)

    # Step 10: Send notification
    send_notification_op(metrics, run_id, config)
