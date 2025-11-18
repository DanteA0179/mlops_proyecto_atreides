"""
CLI Script to Run Training Pipeline.

This script provides a command-line interface for executing the Prefect training pipeline
with various configuration options.

Usage Examples
--------------
# Run with default XGBoost config
python scripts/run_training_pipeline.py --model-type xgboost

# Run with custom config
python scripts/run_training_pipeline.py --config config/training/custom.yaml

# Run LightGBM
python scripts/run_training_pipeline.py --model-type lightgbm

# Run without DVC push (for testing)
python scripts/run_training_pipeline.py --model-type xgboost --skip-dvc-push

# Run with verbose output
python scripts/run_training_pipeline.py --model-type xgboost --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flows.training_pipeline import training_flow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure Prefect logging to show task execution
try:
    from prefect.logging import get_logger as get_prefect_logger

    # Enable Prefect logging
    prefect_logger = get_prefect_logger()
    prefect_logger.setLevel(logging.INFO)

    # Also configure the prefect.flow_runs logger
    flow_runs_logger = logging.getLogger("prefect.flow_runs")
    flow_runs_logger.setLevel(logging.INFO)

    task_runs_logger = logging.getLogger("prefect.task_runs")
    task_runs_logger.setLevel(logging.INFO)

except ImportError:
    logger.warning("Prefect not installed - task logging will be limited")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run ML training pipeline with Prefect orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run XGBoost training
  python scripts/run_training_pipeline.py --model-type xgboost

  # Run with custom config
  python scripts/run_training_pipeline.py --config config/training/custom.yaml

  # Run without DVC push (testing)
  python scripts/run_training_pipeline.py --model-type xgboost --skip-dvc-push

  # Run all models sequentially
  python scripts/run_training_pipeline.py --model-type all
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (overrides --model-type)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=[
            "xgboost",
            "lightgbm",
            "catboost",
            "ensemble_ridge",
            "ensemble_lightgbm",
            "all",
        ],
        help="Model type (uses default config from config/training/)",
    )

    parser.add_argument(
        "--skip-dvc-push",
        action="store_true",
        help="Skip DVC push to remote storage",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )

    return parser.parse_args()


def run_single_model(config_path: str, skip_dvc_push: bool) -> dict:
    """
    Run training pipeline for a single model.

    Parameters
    ----------
    config_path : str
        Path to configuration file
    skip_dvc_push : bool
        Whether to skip DVC push

    Returns
    -------
    dict
        Pipeline results
    """
    logger.info("=" * 70)
    logger.info(f"Running training pipeline: {config_path}")
    logger.info("=" * 70)

    # Run flow
    result = training_flow(config_path=config_path, skip_dvc_push=skip_dvc_push)

    return result


def run_all_models(skip_dvc_push: bool) -> dict[str, dict]:
    """
    Run training pipeline for all model types sequentially.

    Parameters
    ----------
    skip_dvc_push : bool
        Whether to skip DVC push

    Returns
    -------
    dict[str, dict]
        Results for each model type
    """
    model_types = [
        "xgboost",
        "lightgbm",
        "catboost",
        "ensemble_ridge",
        "ensemble_lightgbm",
    ]

    results = {}

    for model_type in model_types:
        config_path = f"config/training/{model_type}_config.yaml"

        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING MODEL {len(results) + 1}/{len(model_types)}: {model_type.upper()}")
        logger.info("=" * 70)

        try:
            result = run_single_model(config_path, skip_dvc_push)
            results[model_type] = result
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = {"status": "error", "error": str(e)}

    return results


def print_results(results: dict) -> None:
    """
    Print pipeline results in a formatted way.

    Parameters
    ----------
    results : dict
        Pipeline results
    """
    print("\n" + "=" * 70)
    print("📊 PIPELINE RESULTS")
    print("=" * 70)

    print(f"Status: {results['status']}")

    if results["status"] == "success":
        metrics = results["metrics"]
        print("\nValidation Metrics:")
        print(f"  RMSE: {metrics['val']['rmse']:.4f} kWh")
        print(f"  R²:   {metrics['val']['r2']:.4f}")
        print(f"  MAE:  {metrics['val']['mae']:.4f} kWh")

        print("\nTest Metrics:")
        print(f"  RMSE: {metrics['test']['rmse']:.4f} kWh")
        print(f"  R²:   {metrics['test']['r2']:.4f}")
        print(f"  MAE:  {metrics['test']['mae']:.4f} kWh")

        print("\nArtifacts:")
        print(f"  Model path: {results['model_path']}")
        print(f"  MLflow run: {results['run_id']}")
        print(f"  DVC file:   {results['dvc_file']}")
        print(f"  DVC pushed: {results['dvc_pushed']}")

    elif results["status"] == "failed_threshold":
        threshold_result = results.get("threshold_result", {})
        print("\nModel failed performance threshold:")
        print(
            f"  Val RMSE: {threshold_result.get('val_rmse', 0):.4f} kWh "
            f"(threshold: {threshold_result.get('threshold_rmse', 0):.4f})"
        )
        print(
            f"  Val R²:   {threshold_result.get('val_r2', 0):.4f} "
            f"(threshold: {threshold_result.get('threshold_r2', 0):.4f})"
        )

    print("=" * 70)


def print_all_results(all_results: dict[str, dict]) -> None:
    """
    Print results for all models.

    Parameters
    ----------
    all_results : dict[str, dict]
        Results for each model type
    """
    print("\n" + "=" * 70)
    print("📊 ALL MODELS RESULTS")
    print("=" * 70)

    for model_type, results in all_results.items():
        status = results.get("status", "unknown")
        print(f"\n{model_type.upper()}:")
        print(f"  Status: {status}")

        if status == "success":
            test_rmse = results["metrics"]["test"]["rmse"]
            test_r2 = results["metrics"]["test"]["r2"]
            print(f"  Test RMSE: {test_rmse:.4f} kWh")
            print(f"  Test R²:   {test_r2:.4f}")
        elif status == "failed_threshold":
            val_rmse = results["threshold_result"]["val_rmse"]
            threshold_rmse = results["threshold_result"]["threshold_rmse"]
            print(f"  Val RMSE:  {val_rmse:.4f} kWh (threshold: {threshold_rmse:.4f})")

    print("=" * 70)


def main() -> None:
    """Main entry point for CLI script."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    # Validate arguments
    if not args.config and not args.model_type:
        logger.error("Error: Must specify either --config or --model-type")
        sys.exit(1)

    # Determine config path(s)
    if args.model_type == "all":
        # Run all models
        logger.info("Running training for all model types...")
        all_results = run_all_models(args.skip_dvc_push)
        print_all_results(all_results)
        sys.exit(0)

    # Single model
    if args.config:
        config_path = args.config
    else:
        config_path = f"config/training/{args.model_type}_config.yaml"

    # Check config exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_single_model(config_path, args.skip_dvc_push)
        print_results(results)

        # Exit with appropriate code
        if results["status"] == "success":
            sys.exit(0)
        elif results["status"] == "failed_threshold":
            logger.warning("Model failed threshold - exiting with code 2")
            sys.exit(2)
        else:
            logger.error("Unknown status - exiting with code 1")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
