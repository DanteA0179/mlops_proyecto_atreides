"""
Script to export all available models to ONNX format.

This script exports gradient boosting models, ensembles, and foundation models
to ONNX format for optimized inference.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.export_onnx import ONNXExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Export all available models to ONNX format."""
    logger.info("Starting ONNX model export...")

    exporter = ONNXExporter()

    logger.info("Available models for export:")
    for model_name in exporter.list_available_models():
        info = exporter.get_model_info(model_name)
        status = "✓" if info["exists"] else "✗"
        logger.info(f"  {status} {model_name} ({info['type']})")

    models_to_export = ["lightgbm", "lightgbm_ensemble"]

    logger.info(f"\nExporting {len(models_to_export)} models...")
    exported = exporter.export_all_models(models=models_to_export)

    logger.info("\nExport complete!")
    logger.info(f"Successfully exported {len(exported)} models:")
    for name, path in exported.items():
        if isinstance(path, dict):
            logger.info(f"  ✓ {name} (ensemble):")
            for sub_name, sub_path in path.items():
                logger.info(f"    - {sub_name}: {sub_path}")
        else:
            logger.info(f"  ✓ {name}: {path}")

    logger.info("\nNext steps:")
    logger.info("1. Run validation: python scripts/validate_onnx_models.py")
    logger.info("2. Run benchmark: python scripts/benchmark_onnx_models.py")
    logger.info("3. Test API: python -m src.api.main")


if __name__ == "__main__":
    main()
