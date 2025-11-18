"""
ONNX Model Benchmark for Energy Optimization Copilot.

This module provides functionality to benchmark ONNX models against
original models, measuring latency, throughput, and memory usage.

Supports benchmarking for:
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Ensembles: Ridge Stacking, LightGBM Stacking
- Foundation Models: Chronos-2 (zero-shot, fine-tuned, covariates)
"""

import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import psutil

logger = logging.getLogger(__name__)


class ONNXBenchmark:
    """
    ONNX model benchmark with support for multiple model types.

    Measures performance metrics including latency, throughput, and memory usage
    for both original and ONNX models.

    Attributes
    ----------
    num_runs : int
        Number of benchmark runs for latency measurement
    warmup_runs : int
        Number of warmup runs before benchmarking
    """

    def __init__(self, num_runs: int = 1000, warmup_runs: int = 10):
        """
        Initialize ONNX benchmark.

        Parameters
        ----------
        num_runs : int, default=1000
            Number of runs for latency measurement
        warmup_runs : int, default=10
            Number of warmup runs
        """
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        logger.info(f"ONNXBenchmark initialized with {num_runs} runs, {warmup_runs} warmup")

    def benchmark_latency(
        self,
        model,
        test_data: np.ndarray,
        model_type: str = "original",
    ) -> dict[str, float]:
        """
        Benchmark model latency.

        Parameters
        ----------
        model : Any
            Model to benchmark (original or ONNX session)
        test_data : np.ndarray
            Test data for inference
        model_type : str, default='original'
            Type of model ('original' or 'onnx')

        Returns
        -------
        dict[str, float]
            Latency metrics (p50, p95, p99, mean, std)
        """
        logger.info(f"Benchmarking latency for {model_type} model...")

        for _ in range(self.warmup_runs):
            if model_type == "onnx":
                input_name = model.get_inputs()[0].name
                model.run(None, {input_name: test_data})
            else:
                model.predict(test_data)

        latencies = []
        for _ in range(self.num_runs):
            start = time.perf_counter()

            if model_type == "onnx":
                input_name = model.get_inputs()[0].name
                model.run(None, {input_name: test_data})
            else:
                model.predict(test_data)

            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies_array = np.array(latencies)

        metrics = {
            "p50_ms": float(np.percentile(latencies_array, 50)),
            "p95_ms": float(np.percentile(latencies_array, 95)),
            "p99_ms": float(np.percentile(latencies_array, 99)),
            "mean_ms": float(np.mean(latencies_array)),
            "std_ms": float(np.std(latencies_array)),
            "min_ms": float(np.min(latencies_array)),
            "max_ms": float(np.max(latencies_array)),
        }

        logger.info(f"Latency p50: {metrics['p50_ms']:.2f}ms, p95: {metrics['p95_ms']:.2f}ms")

        return metrics

    def benchmark_throughput(
        self,
        model,
        test_data: np.ndarray,
        model_type: str = "original",
        batch_sizes: list[int] | None = None,
    ) -> dict[int, float]:
        """
        Benchmark model throughput for different batch sizes.

        Parameters
        ----------
        model : Any
            Model to benchmark
        test_data : np.ndarray
            Test data template
        model_type : str, default='original'
            Type of model
        batch_sizes : list[int], optional
            Batch sizes to test. Default: [1, 10, 50, 100]

        Returns
        -------
        dict[int, float]
            Throughput (predictions/second) for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100]

        logger.info(f"Benchmarking throughput for {model_type} model...")

        throughput_results = {}

        for batch_size in batch_sizes:
            batch_data = np.repeat(test_data[:1], batch_size, axis=0)

            for _ in range(self.warmup_runs):
                if model_type == "onnx":
                    input_name = model.get_inputs()[0].name
                    model.run(None, {input_name: batch_data})
                else:
                    model.predict(batch_data)

            start = time.perf_counter()
            num_iterations = max(10, 100 // batch_size)

            for _ in range(num_iterations):
                if model_type == "onnx":
                    input_name = model.get_inputs()[0].name
                    model.run(None, {input_name: batch_data})
                else:
                    model.predict(batch_data)

            end = time.perf_counter()
            elapsed = end - start

            throughput = (batch_size * num_iterations) / elapsed
            throughput_results[batch_size] = float(throughput)

            logger.info(f"Batch {batch_size}: {throughput:.2f} pred/sec")

        return throughput_results

    def benchmark_memory(
        self,
        model,
        test_data: np.ndarray,
        model_type: str = "original",
    ) -> dict[str, float]:
        """
        Benchmark model memory usage.

        Parameters
        ----------
        model : Any
            Model to benchmark
        test_data : np.ndarray
            Test data for inference
        model_type : str, default='original'
            Type of model

        Returns
        -------
        dict[str, float]
            Memory usage metrics (RAM in MB)
        """
        logger.info(f"Benchmarking memory for {model_type} model...")

        process = psutil.Process()

        mem_before = process.memory_info().rss / (1024 * 1024)

        for _ in range(100):
            if model_type == "onnx":
                input_name = model.get_inputs()[0].name
                model.run(None, {input_name: test_data})
            else:
                model.predict(test_data)

        mem_after = process.memory_info().rss / (1024 * 1024)

        memory_usage = mem_after - mem_before

        metrics = {
            "memory_before_mb": float(mem_before),
            "memory_after_mb": float(mem_after),
            "memory_used_mb": float(memory_usage),
        }

        logger.info(f"Memory used: {memory_usage:.2f} MB")

        return metrics

    def benchmark_model_comparison(
        self,
        original_path: str,
        onnx_path: str,
        model_name: str,
        test_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Benchmark and compare original vs ONNX model.

        Parameters
        ----------
        original_path : str
            Path to original model
        onnx_path : str
            Path to ONNX model
        model_name : str
            Name of the model
        test_data : np.ndarray, optional
            Test data. If None, generates random data.

        Returns
        -------
        dict[str, Any]
            Comparison report with metrics for both models
        """
        logger.info(f"Benchmarking {model_name}...")

        original_model = joblib.load(original_path)
        onnx_session = ort.InferenceSession(onnx_path)

        if test_data is None:
            test_data = np.random.randn(1, 18).astype(np.float32)

        original_latency = self.benchmark_latency(original_model, test_data, "original")
        onnx_latency = self.benchmark_latency(onnx_session, test_data, "onnx")

        original_throughput = self.benchmark_throughput(original_model, test_data, "original")
        onnx_throughput = self.benchmark_throughput(onnx_session, test_data, "onnx")

        original_memory = self.benchmark_memory(original_model, test_data, "original")
        onnx_memory = self.benchmark_memory(onnx_session, test_data, "onnx")

        latency_improvement = (
            (original_latency["p50_ms"] - onnx_latency["p50_ms"]) / original_latency["p50_ms"] * 100
        )

        throughput_improvement = (
            (onnx_throughput[1] - original_throughput[1]) / original_throughput[1] * 100
        )

        memory_improvement = (
            (original_memory["memory_used_mb"] - onnx_memory["memory_used_mb"])
            / original_memory["memory_used_mb"]
            * 100
        )

        report = {
            "model_name": model_name,
            "benchmark_date": datetime.now().isoformat(),
            "original": {
                "latency": original_latency,
                "throughput": original_throughput,
                "memory": original_memory,
            },
            "onnx": {
                "latency": onnx_latency,
                "throughput": onnx_throughput,
                "memory": onnx_memory,
            },
            "improvement": {
                "latency_reduction_pct": float(latency_improvement),
                "throughput_increase_pct": float(throughput_improvement),
                "memory_reduction_pct": float(memory_improvement),
            },
        }

        logger.info(f"Latency improvement: {latency_improvement:.1f}%")
        logger.info(f"Throughput improvement: {throughput_improvement:.1f}%")

        return report

    def benchmark_all_models(
        self,
        models_config: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """
        Benchmark multiple models.

        Parameters
        ----------
        models_config : dict[str, dict[str, str]]
            Dictionary mapping model names to paths
            Format: {"model_name": {"original": "path", "onnx": "path"}}

        Returns
        -------
        dict[str, Any]
            Consolidated benchmark report
        """
        results = {}

        for model_name, paths in models_config.items():
            try:
                logger.info(f"Benchmarking {model_name}...")
                result = self.benchmark_model_comparison(
                    paths["original"],
                    paths["onnx"],
                    model_name,
                )
                results[model_name] = result

            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {
                    "model_name": model_name,
                    "status": "ERROR",
                    "error": str(e),
                }

        return results

    def generate_report(
        self,
        benchmark_results: dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Generate consolidated benchmark report.

        Parameters
        ----------
        benchmark_results : dict[str, Any]
            Benchmark results from multiple models
        output_path : str
            Path to save report JSON
        """
        hardware_info = {
            "cpu": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "platform": platform.platform(),
        }

        try:
            import torch

            if torch.cuda.is_available():
                hardware_info["gpu"] = torch.cuda.get_device_name(0)
                hardware_info["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
            else:
                hardware_info["gpu"] = "None"
        except ImportError:
            hardware_info["gpu"] = "Unknown"

        avg_latency_improvement = np.mean(
            [
                r["improvement"]["latency_reduction_pct"]
                for r in benchmark_results.values()
                if "improvement" in r
            ]
        )

        avg_throughput_improvement = np.mean(
            [
                r["improvement"]["throughput_increase_pct"]
                for r in benchmark_results.values()
                if "improvement" in r
            ]
        )

        fastest_model = min(
            [
                (name, r["onnx"]["latency"]["p50_ms"])
                for name, r in benchmark_results.items()
                if "onnx" in r
            ],
            key=lambda x: x[1],
        )[0]

        consolidated_report = {
            "report_date": datetime.now().isoformat(),
            "hardware": hardware_info,
            "num_runs": self.num_runs,
            "warmup_runs": self.warmup_runs,
            "models": benchmark_results,
            "summary": {
                "total_models": len(benchmark_results),
                "avg_latency_improvement_pct": float(avg_latency_improvement),
                "avg_throughput_improvement_pct": float(avg_throughput_improvement),
                "fastest_model": fastest_model,
            },
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(consolidated_report, f, indent=2)

        logger.info(f"Benchmark report saved to {output_path}")
        logger.info(f"Average latency improvement: {avg_latency_improvement:.1f}%")
        logger.info(f"Fastest model: {fastest_model}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    benchmark = ONNXBenchmark(num_runs=1000, warmup_runs=10)

    models_to_benchmark = {
        "xgboost": {
            "original": "models/baselines/xgboost_optimized.pkl",
            "onnx": "models/onnx/xgboost.onnx",
        },
        "lightgbm": {
            "original": "models/baselines/lightgbm_test_-gbm_v1.pkl",
            "onnx": "models/onnx/lightgbm.onnx",
        },
    }

    print("Benchmarking ONNX models...")
    results = benchmark.benchmark_all_models(models_to_benchmark)

    benchmark.generate_report(results, "models/benchmarks/onnx_comparison.json")

    print("\nBenchmark complete. Results saved to models/benchmarks/onnx_comparison.json")
