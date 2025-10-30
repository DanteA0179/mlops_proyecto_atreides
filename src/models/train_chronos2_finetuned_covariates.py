"""
Chronos-2 Fine-Tuning Training Script with Covariates Support.

This script fine-tunes Chronos-2 with covariates, respecting the constraint
that future_covariates must be a subset of past_covariates.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import polars as pl
import torch
from chronos import Chronos2Pipeline

from src.models.chronos2_finetuning_covariates import (
    finetune_chronos2_with_covariates,
    save_finetuned_pipeline,
)
from src.models.train_chronos2 import (
    setup_gpu,
    load_data,
    evaluate_chronos2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 model with covariates")
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length for predictions",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of fine-tuning steps (default: 10 for quick testing)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for fine-tuning (use smaller for GPU memory)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="steel_energy_chronos2_finetuned_covariates",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--skip-zero-shot",
        action="store_true",
        help="Skip zero-shot evaluation (if already done)",
    )
    
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("="*70)
    logger.info("Chronos-2 Fine-Tuning Pipeline with Covariates")
    logger.info("="*70)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Context length: {args.context_length}")
    logger.info(f"Fine-tuning steps: {args.num_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info("="*70)
    
    # Setup
    device = setup_gpu()
    
    # Clear GPU cache at start
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    data_dir = Path("data/processed")
    output_dir = Path("models/foundation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n[1/6] Loading data")
    df_train, df_val, df_test = load_data(data_dir)
    
    # Load base model
    logger.info("\n[2/6] Loading Chronos-2 base model")
    model_name = "s3://autogluon/chronos-2"
    
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    
    logger.info(f"Model loaded: {model_name}")
    
    # Setup MLflow - use http://localhost:5000 to connect to Docker MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.experiment_name)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Zero-shot evaluation (optional)
    zero_shot_metrics = None
    if not args.skip_zero_shot:
        logger.info("\n[3/6] Zero-shot evaluation (baseline)")
        
        with mlflow.start_run(run_name=f"chronos2_zero_shot_{timestamp}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", "chronos2_zero_shot")
            mlflow.log_param("context_length", args.context_length)
            mlflow.log_param("device", device)
            
            zero_shot_results = evaluate_chronos2(
                pipeline=pipeline,
                df=df_test,
                target_col="Usage_kWh",
                context_length=args.context_length,
                prediction_length=1,
            )
            
            zero_shot_metrics = zero_shot_results["metrics"]
            
            for key, value in zero_shot_metrics.items():
                mlflow.log_metric(f"test_{key}", value)
            
            logger.info(f"Zero-shot RMSE: {zero_shot_metrics['rmse']:.2f} kWh")
    else:
        logger.info("\n[3/6] Skipping zero-shot evaluation")
    
    # Fine-tuning with covariates
    logger.info("\n[4/6] Fine-tuning Chronos-2 with Covariates")
    
    with mlflow.start_run(run_name=f"chronos2_finetuned_covariates_{timestamp}"):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", "chronos2_finetuned_covariates")
        mlflow.log_param("num_steps", args.num_steps)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("context_length", args.context_length)
        mlflow.log_param("device", device)
        
        # Define covariates for fine-tuning
        # IMPORTANT: During fine-tuning, we can only use past_covariates
        # future_covariates are only used during inference/prediction
        
        past_covariates = [
            "WeekStatus",
            "Load_Type_Maximum_Load",
            "Load_Type_Medium_Load",
            "NSM",
            "CO2(tCO2)",
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
        ]
        
        # No future covariates during training (only during inference)
        future_covariates = None
        
        logger.info(f"Past covariates ({len(past_covariates)}): {past_covariates}")
        logger.info("Future covariates: None (only used during inference)")
        
        mlflow.log_param("past_covariates", ",".join(past_covariates))
        mlflow.log_param("future_covariates", "none")
        
        # Fine-tune
        finetuned_pipeline = finetune_chronos2_with_covariates(
            pipeline=pipeline,
            df_train=df_train,
            df_val=df_val,
            target_col="Usage_kWh",
            prediction_length=1,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            logging_steps=10,
        )
        
        # Save fine-tuned model
        logger.info("\n[5/6] Saving fine-tuned model")
        save_path = output_dir / f"chronos2_finetuned_covariates_{timestamp}"
        save_finetuned_pipeline(finetuned_pipeline, save_path)
        
        # Evaluate fine-tuned model
        logger.info("\n[6/6] Evaluating fine-tuned model")
        finetuned_results = evaluate_chronos2(
            pipeline=finetuned_pipeline,
            df=df_test,
            target_col="Usage_kWh",
            context_length=args.context_length,
            prediction_length=1,
        )
        
        finetuned_metrics = finetuned_results["metrics"]
        
        # Log metrics
        for key, value in finetuned_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        
        # Calculate improvement
        if zero_shot_metrics:
            improvement = (
                (zero_shot_metrics["rmse"] - finetuned_metrics["rmse"])
                / zero_shot_metrics["rmse"]
                * 100
            )
            mlflow.log_metric("improvement_vs_zero_shot_pct", improvement)
            logger.info(f"\nImprovement vs zero-shot: {improvement:.1f}%")
        
        # Save results
        results_file = output_dir / f"chronos2_finetuned_covariates_results_{timestamp}.json"
        results_data = {
            "model": model_name,
            "finetuned": True,
            "with_covariates": True,
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "metrics": finetuned_metrics,
            "timestamp": timestamp,
        }
        
        if zero_shot_metrics:
            results_data["zero_shot_metrics"] = zero_shot_metrics
            results_data["improvement_pct"] = improvement
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Log only results file, NOT the full model (455MB)
        # Model is saved locally in models/foundation/
        mlflow.log_artifact(str(results_file))
        mlflow.log_param("model_path", str(save_path))
        mlflow.log_param("model_size_mb", 455.79)
        
        logger.info("\n" + "="*70)
        logger.info("RESULTS SUMMARY")
        logger.info("="*70)
        
        if zero_shot_metrics:
            logger.info(f"Zero-shot RMSE: {zero_shot_metrics['rmse']:.2f} kWh")
        logger.info(f"Fine-tuned RMSE: {finetuned_metrics['rmse']:.2f} kWh")
        logger.info(f"Fine-tuned MAE: {finetuned_metrics['mae']:.2f} kWh")
        logger.info(f"Fine-tuned R²: {finetuned_metrics['r2']:.4f}")
        
        if zero_shot_metrics:
            logger.info(f"Improvement: {improvement:.1f}%")
        
        logger.info("="*70)
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Model saved to: {save_path}")
        logger.info("="*70)
    
    return finetuned_results


if __name__ == "__main__":
    main()
