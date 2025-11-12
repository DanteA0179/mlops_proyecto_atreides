"""
Prepare base models for ensemble training.

This script copies or links the trained base models to a common directory
structure expected by the ensemble training script.

The ensemble expects models in:
    models/gradient_boosting/
        - xgboost_model.pkl
        - lightgbm_model.pkl
        - catboost_model.pkl

Usage:
    poetry run python scripts/prepare_base_models.py
"""

import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_model(model_dir: Path, pattern: str) -> Path | None:
    """
    Find the latest model file matching pattern.

    Parameters
    ----------
    model_dir : Path
        Directory to search
    pattern : str
        Glob pattern for model files

    Returns
    -------
    Path or None
        Path to latest model file
    """
    model_files = list(model_dir.glob(pattern))

    if not model_files:
        return None

    # Sort by modification time (newest first)
    latest = max(model_files, key=lambda p: p.stat().st_mtime)

    return latest


def prepare_base_models() -> None:
    """
    Prepare base models for ensemble training.

    Copies the latest version of each base model to the
    expected directory structure.
    """
    logger.info("Preparing base models for ensemble training")

    # Define source directory
    source_dir = Path("models/baselines")
    
    # Define target directory
    target_dir = Path("models/gradient_boosting")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Model mapping: (source_pattern, target_name)
    model_mappings = {
        'xgboost': ('xgboost_*.pkl', 'xgboost_model.pkl'),
        'lightgbm': ('lightgbm_*.pkl', 'lightgbm_model.pkl'),
        'catboost': ('catboost_*.pkl', 'catboost_model.pkl'),
    }

    copied_models = []

    for model_type, (pattern, target_name) in model_mappings.items():
        logger.info(f"Processing {model_type}")

        # Find latest model
        source_file = find_latest_model(source_dir, pattern)

        if source_file is None:
            logger.warning(f"No {model_type} model found with pattern '{pattern}'")
            logger.warning(f"Skipping {model_type}")
            continue

        # Copy to target
        target_file = target_dir / target_name

        logger.info(f"Copying {source_file.name} -> {target_file}")
        shutil.copy2(source_file, target_file)

        copied_models.append(model_type)
        logger.info(f"Successfully copied {model_type}")

    # Summary
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Models copied: {len(copied_models)}/{len(model_mappings)}")

    if copied_models:
        logger.info("\nCopied models:")
        for model_type in copied_models:
            target_file = target_dir / model_mappings[model_type][1]
            size_mb = target_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {model_type:12s} - {target_file.name:25s} ({size_mb:.2f} MB)")

    missing_models = set(model_mappings.keys()) - set(copied_models)
    if missing_models:
        logger.warning("\nMissing models:")
        for model_type in missing_models:
            logger.warning(f"  {model_type}")
        logger.warning("\nTrain missing models before running ensemble training")

    logger.info("="*70)

    if len(copied_models) >= 2:
        logger.info("✓ Ready for ensemble training (minimum 2 base models)")
    else:
        logger.error("✗ Not ready - need at least 2 base models")
        sys.exit(1)


if __name__ == "__main__":
    prepare_base_models()
