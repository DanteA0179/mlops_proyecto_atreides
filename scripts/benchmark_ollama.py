"""
Benchmark script for Ollama + Llama 3.2 3B model.

This script measures:
- Latency (average, p50, p95, p99)
- Throughput (tokens/second)
- VRAM usage during inference
- Performance across different prompt sizes

Results are saved to reports/ollama_benchmark_results.json
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any
import requests
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaBenchmark:
    """Benchmark tool for Ollama API performance testing."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize benchmark tool.

        Parameters
        ----------
        host : str
            Ollama API host URL
        model : str
            Model name to benchmark
        """
        self.host = host
        self.model = model
        self.api_url = f"{host}/api/generate"

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate text using Ollama API.

        Parameters
        ----------
        prompt : str
            Input prompt

        Returns
        -------
        dict
            API response with generation metrics
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        start_time = time.time()
        response = requests.post(self.api_url, json=payload, timeout=120)
        end_time = time.time()

        response.raise_for_status()
        data = response.json()

        # Add wall clock latency
        data["wall_clock_latency_ms"] = (end_time - start_time) * 1000

        return data

    def benchmark_prompt(self, prompt: str, num_runs: int = 5) -> Dict[str, Any]:
        """
        Benchmark a single prompt multiple times.

        Parameters
        ----------
        prompt : str
            Test prompt
        num_runs : int
            Number of benchmark runs

        Returns
        -------
        dict
            Aggregated benchmark metrics
        """
        logger.info(f"Benchmarking prompt (length: {len(prompt)} chars)")

        latencies = []
        token_counts = []
        tokens_per_second_list = []

        for i in range(num_runs):
            logger.info(f"Run {i+1}/{num_runs}")
            try:
                result = self.generate(prompt)

                # Extract metrics
                latency = result["wall_clock_latency_ms"]
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 0)

                latencies.append(latency)
                token_counts.append(eval_count)

                # Calculate tokens/second from eval metrics
                if eval_duration > 0:
                    tokens_per_sec = (eval_count / eval_duration) * 1e9
                    tokens_per_second_list.append(tokens_per_sec)

            except Exception as e:
                logger.error(f"Error in run {i+1}: {e}")
                continue

        # Calculate statistics
        return {
            "num_runs": len(latencies),
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99)
            },
            "tokens_generated": {
                "mean": statistics.mean(token_counts),
                "min": min(token_counts),
                "max": max(token_counts)
            },
            "tokens_per_second": {
                "mean": statistics.mean(tokens_per_second_list) if tokens_per_second_list else 0,
                "min": min(tokens_per_second_list) if tokens_per_second_list else 0,
                "max": max(tokens_per_second_list) if tokens_per_second_list else 0
            }
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run full benchmark suite with different prompt sizes.

        Returns
        -------
        dict
            Complete benchmark results
        """
        logger.info("=" * 70)
        logger.info("Starting Ollama Benchmark")
        logger.info(f"Model: {self.model}")
        logger.info(f"Host: {self.host}")
        logger.info("=" * 70)

        # Test prompts of different sizes
        prompts = {
            "short": "What is energy consumption?",
            "medium": (
                "Explain energy consumption in industrial settings. "
                "Discuss the main factors that affect energy usage in manufacturing plants, "
                "including equipment efficiency, load types, and operational schedules. "
                "Provide specific examples from steel production facilities."
            ),
            "long": (
                "Analyze energy consumption patterns in steel manufacturing plants. "
                "Consider the following aspects: "
                "1. Different types of equipment and their power requirements "
                "2. The relationship between production load (Light, Medium, Maximum) and energy usage "
                "3. Temporal patterns in energy consumption throughout the day and week "
                "4. The impact of reactive power and power factor on overall efficiency "
                "5. CO2 emissions and their correlation with energy usage "
                "6. Strategies for optimizing energy consumption during different production phases "
                "Provide a comprehensive analysis with specific recommendations for reducing "
                "energy costs while maintaining production quality and output."
            )
        }

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "host": self.host,
            "benchmarks": {}
        }

        for prompt_type, prompt in prompts.items():
            logger.info(f"\nBenchmarking {prompt_type} prompt...")
            results["benchmarks"][prompt_type] = self.benchmark_prompt(prompt)
            time.sleep(2)  # Cool down between tests

        # Overall summary
        all_latencies = []
        for bench_data in results["benchmarks"].values():
            all_latencies.append(bench_data["latency_ms"]["mean"])

        results["summary"] = {
            "average_latency_ms": statistics.mean(all_latencies),
            "max_latency_ms": max(all_latencies),
            "criteria_validation": {
                "latency_under_2s": statistics.mean(all_latencies) < 2000,
                "vram_under_8gb": "Manual check required (use nvidia-smi)"
            }
        }

        logger.info("\n" + "=" * 70)
        logger.info("Benchmark Complete")
        logger.info(f"Average latency: {results['summary']['average_latency_ms']:.2f}ms")
        logger.info(f"Latency < 2s: {results['summary']['criteria_validation']['latency_under_2s']}")
        logger.info("=" * 70)

        return results

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """
        Calculate percentile of data.

        Parameters
        ----------
        data : list
            Numeric data
        percentile : float
            Percentile to calculate (0-100)

        Returns
        -------
        float
            Percentile value
        """
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        floor_index = int(index)
        ceil_index = min(floor_index + 1, len(sorted_data) - 1)

        if floor_index == ceil_index:
            return sorted_data[floor_index]

        # Linear interpolation
        fraction = index - floor_index
        return sorted_data[floor_index] * (1 - fraction) + sorted_data[ceil_index] * fraction


def main():
    """Main execution function."""
    # Initialize benchmark
    benchmark = OllamaBenchmark()

    # Run benchmark
    results = benchmark.run_full_benchmark()

    # Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    output_file = reports_dir / "ollama_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
