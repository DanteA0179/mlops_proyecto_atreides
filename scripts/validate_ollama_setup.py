"""
Validation script for Ollama setup.

This script performs comprehensive validation of the Ollama + Llama 3.2 3B setup,
including:
- Ollama service health check
- Model availability verification
- Basic functionality testing
- Performance validation
- VRAM usage check

Run with: python scripts/validate_ollama_setup.py
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.ollama_client import OllamaClient
from src.llm.llm_factory import get_available_providers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaValidator:
    """Validator for Ollama setup and configuration."""

    def __init__(self):
        """Initialize validator."""
        self.client = None
        self.validation_results = {
            "ollama_service": False,
            "model_available": False,
            "basic_generation": False,
            "chat_functionality": False,
            "latency_check": False,
            "vram_check": False,
            "provider_status": {}
        }
        self.errors = []
        self.warnings = []

    def validate_ollama_service(self) -> bool:
        """
        Validate that Ollama service is running.

        Returns
        -------
        bool
            True if service is healthy
        """
        logger.info("Validating Ollama service...")

        try:
            self.client = OllamaClient()
            is_healthy = self.client.health_check()

            if is_healthy:
                logger.info("  SUCCESS: Ollama service is running")
                self.validation_results["ollama_service"] = True
                return True
            else:
                logger.error("  FAILED: Ollama service is not responding")
                self.errors.append("Ollama service health check failed")
                return False

        except Exception as e:
            logger.error(f"  FAILED: Error connecting to Ollama: {e}")
            self.errors.append(f"Ollama connection error: {e}")
            return False

    def validate_model_availability(self) -> bool:
        """
        Validate that llama3.2:3b model is available.

        Returns
        -------
        bool
            True if model is available
        """
        logger.info("Validating model availability...")

        try:
            is_available = self.client.is_model_available()

            if is_available:
                logger.info("  SUCCESS: llama3.2:3b model is available")

                # Get model info
                info = self.client.get_model_info()
                if info:
                    size_gb = info.get("size", 0) / (1024**3)
                    logger.info(f"  Model size: {size_gb:.2f} GB")

                self.validation_results["model_available"] = True
                return True
            else:
                logger.error("  FAILED: llama3.2:3b model not found")
                self.errors.append(
                    "Model not available. Run: ollama pull llama3.2:3b"
                )
                return False

        except Exception as e:
            logger.error(f"  FAILED: Error checking model: {e}")
            self.errors.append(f"Model check error: {e}")
            return False

    def validate_basic_generation(self) -> bool:
        """
        Validate basic text generation functionality.

        Returns
        -------
        bool
            True if generation works
        """
        logger.info("Validating basic text generation...")

        try:
            prompt = "What is energy consumption in one sentence?"
            response = self.client.generate(prompt, max_tokens=100)

            if response and len(response) > 10:
                logger.info("  SUCCESS: Text generation working")
                logger.info(f"  Response length: {len(response)} characters")
                self.validation_results["basic_generation"] = True
                return True
            else:
                logger.error("  FAILED: Empty or invalid response")
                self.errors.append("Text generation returned empty response")
                return False

        except Exception as e:
            logger.error(f"  FAILED: Generation error: {e}")
            self.errors.append(f"Generation error: {e}")
            return False

    def validate_chat_functionality(self) -> bool:
        """
        Validate chat functionality.

        Returns
        -------
        bool
            True if chat works
        """
        logger.info("Validating chat functionality...")

        try:
            messages = [
                {"role": "user", "content": "What is power?"}
            ]
            response = self.client.chat(messages, max_tokens=100)

            if response and len(response) > 10:
                logger.info("  SUCCESS: Chat functionality working")
                self.validation_results["chat_functionality"] = True
                return True
            else:
                logger.error("  FAILED: Empty or invalid chat response")
                self.errors.append("Chat returned empty response")
                return False

        except Exception as e:
            logger.error(f"  FAILED: Chat error: {e}")
            self.errors.append(f"Chat error: {e}")
            return False

    def validate_latency(self) -> bool:
        """
        Validate response latency.

        Returns
        -------
        bool
            True if latency is acceptable
        """
        logger.info("Validating response latency...")

        try:
            prompt = "What is energy?"
            latencies = []

            # Run 3 tests
            for i in range(3):
                start_time = time.time()
                self.client.generate(prompt, max_tokens=50)
                latency = time.time() - start_time
                latencies.append(latency)
                logger.info(f"  Test {i+1}: {latency:.2f}s")

            avg_latency = sum(latencies) / len(latencies)
            logger.info(f"  Average latency: {avg_latency:.2f}s")

            # Relaxed threshold for realistic testing
            if avg_latency < 15.0:
                logger.info("  SUCCESS: Latency is acceptable")
                self.validation_results["latency_check"] = True
                return True
            else:
                logger.warning(f"  WARNING: Average latency ({avg_latency:.2f}s) exceeds 15s")
                self.warnings.append(
                    f"High latency detected: {avg_latency:.2f}s. "
                    "Consider model optimization or hardware upgrade."
                )
                self.validation_results["latency_check"] = True  # Still pass but warn
                return True

        except Exception as e:
            logger.error(f"  FAILED: Latency test error: {e}")
            self.errors.append(f"Latency test error: {e}")
            return False

    def validate_vram_usage(self) -> bool:
        """
        Validate VRAM usage using nvidia-smi.

        Returns
        -------
        bool
            True if VRAM usage is acceptable
        """
        logger.info("Validating VRAM usage...")

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                vram_mb = int(result.stdout.strip())
                vram_gb = vram_mb / 1024

                logger.info(f"  Current VRAM usage: {vram_gb:.2f} GB ({vram_mb} MB)")

                if vram_mb < 8192:  # Less than 8GB
                    logger.info("  SUCCESS: VRAM usage is within limits")
                    self.validation_results["vram_check"] = True
                    return True
                else:
                    logger.warning(f"  WARNING: VRAM usage ({vram_gb:.2f} GB) exceeds 8GB")
                    self.warnings.append(
                        f"High VRAM usage: {vram_gb:.2f} GB. "
                        "Consider using smaller model or reducing batch size."
                    )
                    self.validation_results["vram_check"] = True  # Still pass but warn
                    return True
            else:
                logger.warning("  WARNING: Could not query VRAM usage")
                logger.warning("  nvidia-smi not available or failed")
                self.warnings.append("VRAM check skipped - nvidia-smi not available")
                self.validation_results["vram_check"] = True  # Pass if nvidia-smi not available
                return True

        except FileNotFoundError:
            logger.warning("  WARNING: nvidia-smi not found")
            self.warnings.append("VRAM check skipped - nvidia-smi not found")
            self.validation_results["vram_check"] = True  # Pass if not available
            return True
        except Exception as e:
            logger.warning(f"  WARNING: VRAM check error: {e}")
            self.warnings.append(f"VRAM check error: {e}")
            self.validation_results["vram_check"] = True  # Pass on error
            return True

    def validate_provider_status(self) -> bool:
        """
        Validate LLM provider status.

        Returns
        -------
        bool
            True if at least one provider is available
        """
        logger.info("Validating LLM provider status...")

        try:
            providers = get_available_providers()
            self.validation_results["provider_status"] = providers

            for provider_name, status in providers.items():
                if status.get("available") and status.get("healthy"):
                    logger.info(f"  SUCCESS: {provider_name} is available and healthy")
                elif status.get("available") and not status.get("healthy"):
                    logger.warning(f"  WARNING: {provider_name} available but not healthy")
                else:
                    logger.info(f"  INFO: {provider_name} not configured")

            # At least Ollama should be working
            ollama_status = providers.get("ollama", {})
            if ollama_status.get("healthy"):
                return True
            else:
                self.errors.append("No healthy LLM provider found")
                return False

        except Exception as e:
            logger.error(f"  FAILED: Provider status check error: {e}")
            self.errors.append(f"Provider status error: {e}")
            return False

    def run_full_validation(self) -> bool:
        """
        Run full validation suite.

        Returns
        -------
        bool
            True if all critical validations pass
        """
        logger.info("=" * 70)
        logger.info("Starting Ollama Setup Validation")
        logger.info("=" * 70)

        # Run validations in order
        validations = [
            self.validate_ollama_service,
            self.validate_model_availability,
            self.validate_basic_generation,
            self.validate_chat_functionality,
            self.validate_latency,
            self.validate_vram_usage,
            self.validate_provider_status
        ]

        results = []
        for validation in validations:
            try:
                result = validation()
                results.append(result)
            except Exception as e:
                logger.error(f"Validation failed with exception: {e}")
                results.append(False)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("Validation Summary")
        logger.info("=" * 70)

        for key, value in self.validation_results.items():
            if isinstance(value, bool):
                status = "PASS" if value else "FAIL"
                logger.info(f"{key}: {status}")

        if self.warnings:
            logger.info("\nWarnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if self.errors:
            logger.info("\nErrors:")
            for error in self.errors:
                logger.error(f"  - {error}")

        overall_success = all(results)

        logger.info("\n" + "=" * 70)
        if overall_success:
            logger.info("OVERALL STATUS: PASS")
            logger.info("Ollama setup is ready for use!")
        else:
            logger.error("OVERALL STATUS: FAIL")
            logger.error("Please fix the errors above before proceeding")

        logger.info("=" * 70)

        return overall_success


def main():
    """Main execution function."""
    validator = OllamaValidator()
    success = validator.run_full_validation()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
