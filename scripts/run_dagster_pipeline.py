"""
Run Dagster Training Pipeline from CLI.

Usage:
    python scripts/run_dagster_pipeline.py --config config/training/xgboost_config.yaml
    python scripts/run_dagster_pipeline.py --model-type xgboost
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dagster import DagsterInstance, execute_job

from src.dagster_pipeline.definitions import defs


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dagster training pipeline")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "lightgbm", "catboost", "ensemble_ridge", "ensemble_lightgbm"],
        help="Model type (uses default config)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
    elif args.model_type:
        config_path = f"config/training/{args.model_type}_config.yaml"
    else:
        print("Error: Must specify --config or --model-type")
        sys.exit(1)

    # Check config exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("DAGSTER TRAINING PIPELINE")
    print("=" * 70)
    print(f"Config: {config_path}")
    print("=" * 70)
    print("")

    # Execute job
    try:
        result = execute_job(
            defs.get_job_def("training_pipeline"),
            run_config={
                "ops": {
                    "load_config": {
                        "config": {"config_path": config_path}
                    }
                }
            },
            instance=DagsterInstance.get(),
        )

        print("")
        print("=" * 70)
        print("PIPELINE RESULTS")
        print("=" * 70)
        print(f"Success: {result.success}")
        print(f"Run ID: {result.run_id}")
        print("=" * 70)

        if result.success:
            print("")
            print("View run in Dagster UI:")
            print("  http://localhost:3000/runs")
            print("")
            sys.exit(0)
        else:
            print("")
            print("Pipeline failed. Check logs above.")
            print("")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
