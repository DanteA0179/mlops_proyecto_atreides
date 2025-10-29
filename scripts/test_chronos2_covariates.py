"""
Quick test script for Chronos-2 fine-tuning with covariates.

This script validates that the covariates implementation works correctly
without running a full training.
"""

import logging
from pathlib import Path

import polars as pl

from src.utils.chronos_data_prep_covariates import (
    prepare_chronos_finetuning_data_with_covariates,
    validate_finetuning_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_covariates_constraint():
    """Test that covariates constraint validation works."""
    logger.info("="*70)
    logger.info("Testing Chronos-2 Covariates Constraint Validation")
    logger.info("="*70)
    
    # Load sample data
    data_dir = Path("data/processed")
    df_train = pl.read_parquet(data_dir / "steel_preprocessed_train.parquet")
    
    logger.info(f"Loaded training data: {len(df_train)} samples")
    logger.info(f"Columns: {df_train.columns}")
    
    # Test 1: Valid configuration (future is subset of past)
    logger.info("\n[Test 1] Valid configuration - future ⊆ past")
    
    past_covariates = [
        "WeekStatus",
        "Load_Type_Maximum_Load",
        "Load_Type_Medium_Load",
        "NSM",
        "CO2(tCO2)",
    ]
    
    future_covariates = [
        "WeekStatus",
        "Load_Type_Maximum_Load",
        "Load_Type_Medium_Load",
    ]
    
    try:
        train_inputs = prepare_chronos_finetuning_data_with_covariates(
            df=df_train,
            target_col="Usage_kWh",
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        
        stats = validate_finetuning_data(train_inputs)
        
        logger.info("✅ Test 1 PASSED")
        logger.info(f"   Created {stats['n_series']} series")
        logger.info(f"   Past covariates: {stats.get('past_covariate_names', [])}")
        logger.info(f"   Future covariates: {stats.get('future_covariate_names', [])}")
        
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {e}")
        return False
    
    # Test 2: Invalid configuration (future NOT subset of past)
    logger.info("\n[Test 2] Invalid configuration - future ⊄ past (should fail)")
    
    past_covariates_invalid = [
        "NSM",
        "CO2(tCO2)",
    ]
    
    future_covariates_invalid = [
        "WeekStatus",  # Not in past!
        "Load_Type_Maximum_Load",  # Not in past!
    ]
    
    try:
        train_inputs = prepare_chronos_finetuning_data_with_covariates(
            df=df_train,
            target_col="Usage_kWh",
            past_covariates=past_covariates_invalid,
            future_covariates=future_covariates_invalid,
        )
        
        logger.error("❌ Test 2 FAILED: Should have raised ValueError")
        return False
        
    except ValueError as e:
        logger.info("✅ Test 2 PASSED")
        logger.info(f"   Correctly caught constraint violation: {str(e)[:100]}...")
    
    # Test 3: No covariates (should work)
    logger.info("\n[Test 3] No covariates")
    
    try:
        train_inputs = prepare_chronos_finetuning_data_with_covariates(
            df=df_train,
            target_col="Usage_kWh",
            past_covariates=None,
            future_covariates=None,
        )
        
        stats = validate_finetuning_data(train_inputs)
        
        logger.info("✅ Test 3 PASSED")
        logger.info(f"   Created {stats['n_series']} series")
        logger.info(f"   Has past covariates: {stats['has_past_covariates']}")
        logger.info(f"   Has future covariates: {stats['has_future_covariates']}")
        
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        return False
    
    # Test 4: Only past covariates (should work)
    logger.info("\n[Test 4] Only past covariates")
    
    try:
        train_inputs = prepare_chronos_finetuning_data_with_covariates(
            df=df_train,
            target_col="Usage_kWh",
            past_covariates=["NSM", "CO2(tCO2)"],
            future_covariates=None,
        )
        
        stats = validate_finetuning_data(train_inputs)
        
        logger.info("✅ Test 4 PASSED")
        logger.info(f"   Has past covariates: {stats['has_past_covariates']}")
        logger.info(f"   Has future covariates: {stats['has_future_covariates']}")
        
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {e}")
        return False
    
    logger.info("\n" + "="*70)
    logger.info("ALL TESTS PASSED ✅")
    logger.info("="*70)
    
    return True


if __name__ == "__main__":
    success = test_covariates_constraint()
    exit(0 if success else 1)
