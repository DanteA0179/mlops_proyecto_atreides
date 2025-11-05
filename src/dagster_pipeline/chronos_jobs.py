"""
Dagster jobs for Chronos-2 training pipeline.

Three job definitions:
- chronos_zeroshot_job: Zero-shot inference (no training)
- chronos_finetuned_job: Fine-tuning without covariates
- chronos_covariates_job: Fine-tuning with past covariates
"""

from dagster import job

from .chronos_ops import (
    evaluate_chronos_model_op,
    load_chronos_config_op,
    load_chronos_data_op,
    load_chronos_pipeline_op,
    log_chronos_mlflow_op,
    prepare_chronos_data_op,
    save_chronos_model_op,
    train_chronos_model_op,
)


@job(
    name="chronos_zeroshot_job",
    description="Chronos-2 zero-shot inference (no training)",
    tags={"model": "chronos2_zeroshot", "approach": "zero-shot"},
)
def chronos_zeroshot_job():
    """
    Zero-shot Chronos-2 job.

    Pipeline:
    1. Load config
    2. Load data
    3. Load pre-trained pipeline
    4. Prepare temporal data
    5. Evaluate (no training)
    6. Log to MLflow
    """
    cfg = load_chronos_config_op()
    data = load_chronos_data_op(cfg)
    pipeline = load_chronos_pipeline_op(cfg)
    prepared_data = prepare_chronos_data_op(data, cfg)
    metrics = evaluate_chronos_model_op(pipeline, prepared_data, cfg)
    log_chronos_mlflow_op(metrics, cfg)


@job(
    name="chronos_finetuned_job",
    description="Chronos-2 fine-tuning without covariates",
    tags={"model": "chronos2_finetuned", "approach": "fine-tuning"},
)
def chronos_finetuned_job():
    """
    Fine-tuned Chronos-2 job (without covariates).

    Pipeline:
    1. Load config
    2. Load data
    3. Load pre-trained pipeline
    4. Prepare temporal data
    5. Fine-tune model
    6. Evaluate
    7. Save model
    8. Log to MLflow
    """
    cfg = load_chronos_config_op()
    data = load_chronos_data_op(cfg)
    pipeline = load_chronos_pipeline_op(cfg)
    prepared_data = prepare_chronos_data_op(data, cfg)
    finetuned_pipeline = train_chronos_model_op(pipeline, prepared_data, cfg)
    metrics = evaluate_chronos_model_op(finetuned_pipeline, prepared_data, cfg)
    model_path = save_chronos_model_op(finetuned_pipeline, cfg)
    log_chronos_mlflow_op(metrics, cfg, model_path)


@job(
    name="chronos_covariates_job",
    description="Chronos-2 fine-tuning with past covariates",
    tags={"model": "chronos2_covariates", "approach": "fine-tuning-covariates"},
)
def chronos_covariates_job():
    """
    Fine-tuned Chronos-2 job (with 9 past covariates).

    Pipeline:
    1. Load config
    2. Load data
    3. Load pre-trained pipeline
    4. Prepare temporal data (with covariates)
    5. Fine-tune model
    6. Evaluate
    7. Save model
    8. Log to MLflow
    """
    cfg = load_chronos_config_op()
    data = load_chronos_data_op(cfg)
    pipeline = load_chronos_pipeline_op(cfg)
    prepared_data = prepare_chronos_data_op(data, cfg)
    finetuned_pipeline = train_chronos_model_op(pipeline, prepared_data, cfg)
    metrics = evaluate_chronos_model_op(finetuned_pipeline, prepared_data, cfg)
    model_path = save_chronos_model_op(finetuned_pipeline, cfg)
    log_chronos_mlflow_op(metrics, cfg, model_path)
