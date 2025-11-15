"""
Script to benchmark ONNX models against original models.

This script measures latency, throughput, and memory usage
for both original and ONNX models.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.onnx_benchmark import ONNXBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Benchmark ONNX models against original models."""
    logger.info("Starting ONNX model benchmark...")

    benchmark = ONNXBenchmark(num_runs=1000, warmup_runs=10)

    models_to_benchmark = {
        "xgboost": {
            "original": "models/gradient_boosting/xgboost_model.pkl",
            "onnx": "models/onnx/xgboost.onnx",
        },
        "lightgbm": {
            "original": "models/gradient_boosting/lightgbm_model.pkl",
            "onnx": "models/onnx/lightgbm.onnx",
        },
        "catboost": {
            "original": "models/gradient_boosting/catboost_model.pkl",
            "onnx": "models/onnx/catboost.onnx",
        },
    }

    logger.info(f"Benchmarking {len(models_to_benchmark)} models...")
    logger.info("This may take several minutes...")

    results = benchmark.benchmark_all_models(models_to_benchmark)

    output_path = "models/benchmarks/onnx_comparison.json"
    benchmark.generate_report(results, output_path)

    logger.info(f"\nBenchmark complete!")
    logger.info(f"Results saved to: {output_path}")

    for model_name, result in results.items():
        if "improvement" in result:
            logger.info(f"\n{model_name}:")
            logger.info(f"  Latency improvement: {result['improvement']['latency_reduction_pct']:.1f}%")
            logger.info(f"  Throughput improvement: {result['improvement']['throughput_increase_pct']:.1f}%")


if __name__ == "__main__":
    main()
