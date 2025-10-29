"""
Quick test script for Chronos-2 fine-tuning implementation.

This script runs a minimal fine-tuning test (10 steps) to verify
the implementation works before running the full training.
"""

import logging
from pathlib import Path

import polars as pl
import torch
from chronos import Chronos2Pipeline

from src.models.chronos2_finetuning import finetune_chronos2
from src.utils.chronos_data_prep import prepare_chronos_finetuning_data, validate_finetuning_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_data_preparation():
    """Test data preparation function."""
    logger.info("="*70)
    logger.info("TEST 1: Data Preparation")
    logger.info("="*70)
    
    # Load small sample
    df = pl.read_parquet("data/processed/steel_preprocessed_train.parquet")
    df_sample = df.head(1000)  # Use only 1000 samples for quick test
    
    logger.info(f"Sample size: {len(df_sample)}")
    
    # Prepare data
    train_inputs = prepare_chronos_finetuning_data(
        df=df_sample,
        target_col="Usage_kWh",
    )
    
    # Validate
    stats = validate_finetuning_data(train_inputs)
    
    logger.info(f"✓ Data preparation successful")
    logger.info(f"  Series: {stats['n_series']}")
    logger.info(f"  Length: {stats['avg_length']:.0f} timesteps")
    
    return train_inputs


def test_model_loading():
    """Test model loading."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Model Loading")
    logger.info("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    model_name = "s3://autogluon/chronos-2"
    
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    
    logger.info(f"✓ Model loaded: {model_name}")
    
    return pipeline


def test_quick_finetuning(pipeline, train_inputs):
    """Test quick fine-tuning (10 steps)."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Quick Fine-Tuning (10 steps)")
    logger.info("="*70)
    
    # Create minimal DataFrame for testing
    df_sample = pl.DataFrame({
        "Usage_kWh": train_inputs[0]["target"][:1000]
    })
    
    try:
        finetuned = finetune_chronos2(
            pipeline=pipeline,
            df_train=df_sample,
            df_val=None,
            target_col="Usage_kWh",
            prediction_length=1,
            num_steps=10,  # Only 10 steps for quick test
            learning_rate=1e-5,
            batch_size=32,
            logging_steps=5,
        )
        
        logger.info("✓ Fine-tuning successful")
        
        return finetuned
    
    except Exception as e:
        logger.error(f"✗ Fine-tuning failed: {e}")
        raise


def test_prediction(pipeline, train_inputs):
    """Test prediction with fine-tuned model."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Prediction")
    logger.info("="*70)
    
    # Get sample context
    context = train_inputs[0]["target"][:512]
    
    # Prepare input tensor
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Predict
    forecasts = pipeline.predict(
        inputs=context_tensor,
        prediction_length=1,
    )
    
    # Extract prediction
    median_idx = forecasts[0].shape[1] // 2
    pred = float(forecasts[0][0, median_idx, 0].item())
    
    logger.info(f"✓ Prediction successful: {pred:.2f}")
    
    return pred


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("Chronos-2 Fine-Tuning Implementation Tests")
    logger.info("="*70)
    
    try:
        # Test 1: Data preparation
        train_inputs = test_data_preparation()
        
        # Test 2: Model loading
        pipeline = test_model_loading()
        
        # Test 3: Quick fine-tuning
        finetuned = test_quick_finetuning(pipeline, train_inputs)
        
        # Test 4: Prediction
        pred = test_prediction(finetuned, train_inputs)
        
        logger.info("\n" + "="*70)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*70)
        logger.info("\nReady to run full fine-tuning:")
        logger.info("  poetry run python src/models/train_chronos2_finetuned.py")
        logger.info("="*70)
        
        return True
    
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error(f"TESTS FAILED: {e}")
        logger.error("="*70)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
