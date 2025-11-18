"""
Script to validate ONNX model exports against original models.

This script validates that ONNX models produce identical predictions
to original models within numerical tolerance.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.onnx_validator import ONNXValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Validate ONNX models against original models."""
    logger.info("Starting ONNX model validation...")

    validator = ONNXValidator(tolerance=1e-4, num_samples=100)

    models_to_validate = {
        "lightgbm_ensemble": {
            "original": "models/ensembles/ensemble_lightgbm_v3.pkl",
            "onnx": "models/onnx/lightgbm_ensemble",
        },
    }

    logger.info(f"Validating {len(models_to_validate)} models...")
    results = validator.validate_all_models(models_to_validate)

    output_path = "models/benchmarks/validation_report.json"
    validator.generate_report(results, output_path)

    passed = sum(1 for r in results if r.get("status") == "PASSED")
    failed = sum(1 for r in results if r.get("status") == "FAILED")
    errors = sum(1 for r in results if r.get("status") == "ERROR")

    logger.info("\nValidation Summary:")
    logger.info(f"  ✓ Passed: {passed}")
    logger.info(f"  ✗ Failed: {failed}")
    logger.info(f"  ⚠ Errors: {errors}")
    logger.info(f"\nDetailed report saved to: {output_path}")

    if failed > 0 or errors > 0:
        logger.error("Some models failed validation!")
        sys.exit(1)
    else:
        logger.info("All models passed validation!")


if __name__ == "__main__":
    main()
