"""
ONNX Model Validator for Energy Optimization Copilot.

This module provides functionality to validate ONNX model exports
by comparing predictions with original models to ensure numerical equivalence.

Supports validation for:
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Ensembles: Ridge Stacking, LightGBM Stacking
- Foundation Models: Chronos-2 (zero-shot, fine-tuned, covariates)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXValidator:
    """
    ONNX model validator with support for multiple model types.

    Validates that ONNX models produce identical predictions to original models
    within a specified numerical tolerance.

    Attributes
    ----------
    tolerance : float
        Maximum allowed difference between predictions
    num_samples : int
        Number of test samples to validate
    """

    def __init__(self, tolerance: float = 1e-4, num_samples: int = 100):
        """
        Initialize ONNX validator.

        Parameters
        ----------
        tolerance : float, default=1e-5
            Maximum allowed absolute difference between predictions
        num_samples : int, default=100
            Number of test samples to generate for validation
        """
        self.tolerance = tolerance
        self.num_samples = num_samples
        logger.info(f"ONNXValidator initialized with tolerance={tolerance}, samples={num_samples}")

    def validate_xgboost(
        self,
        original_path: str,
        onnx_path: str,
        test_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate XGBoost ONNX model against original.

        Parameters
        ----------
        original_path : str
            Path to original XGBoost model (.pkl)
        onnx_path : str
            Path to ONNX model (.onnx)
        test_data : np.ndarray, optional
            Test data for validation. If None, generates random data.

        Returns
        -------
        dict[str, Any]
            Validation report with metrics and status

        Examples
        --------
        >>> validator = ONNXValidator()
        >>> report = validator.validate_xgboost(
        ...     "models/baselines/xgboost_optimized.pkl",
        ...     "models/onnx/xgboost.onnx"
        ... )
        >>> assert report["status"] == "PASSED"
        """
        logger.info(f"Validating XGBoost model: {onnx_path}")

        original_model = joblib.load(original_path)
        onnx_session = ort.InferenceSession(onnx_path)

        if test_data is None:
            test_data = self._generate_test_data(num_features=18)

        original_preds = original_model.predict(test_data)

        input_name = onnx_session.get_inputs()[0].name
        onnx_preds = onnx_session.run(None, {input_name: test_data.astype(np.float32)})[0]

        if onnx_preds.ndim > 1:
            onnx_preds = onnx_preds.flatten()

        return self._compare_predictions(
            original_preds,
            onnx_preds,
            model_type="xgboost",
            original_path=original_path,
            onnx_path=onnx_path,
        )

    def validate_lightgbm(
        self,
        original_path: str,
        onnx_path: str,
        test_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate LightGBM ONNX model against original.

        Parameters
        ----------
        original_path : str
            Path to original LightGBM model (.pkl)
        onnx_path : str
            Path to ONNX model (.onnx)
        test_data : np.ndarray, optional
            Test data for validation

        Returns
        -------
        dict[str, Any]
            Validation report
        """
        logger.info(f"Validating LightGBM model: {onnx_path}")

        original_model = joblib.load(original_path)

        if hasattr(original_model, "named_steps"):
            logger.info("Detected sklearn Pipeline, extracting final estimator")
            model_for_inference = original_model.named_steps["model"]
        else:
            model_for_inference = original_model

        onnx_session = ort.InferenceSession(onnx_path)

        if test_data is None:
            test_data = self._generate_test_data(model=model_for_inference)

        original_preds = model_for_inference.predict(test_data)

        input_name = onnx_session.get_inputs()[0].name
        onnx_preds = onnx_session.run(None, {input_name: test_data.astype(np.float32)})[0]

        if onnx_preds.ndim > 1:
            onnx_preds = onnx_preds.flatten()

        return self._compare_predictions(
            original_preds,
            onnx_preds,
            model_type="lightgbm",
            original_path=original_path,
            onnx_path=onnx_path,
        )

    def validate_catboost(
        self,
        original_path: str,
        onnx_path: str,
        test_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate CatBoost ONNX model against original.

        Parameters
        ----------
        original_path : str
            Path to original CatBoost model (.pkl)
        onnx_path : str
            Path to ONNX model (.onnx)
        test_data : np.ndarray, optional
            Test data for validation

        Returns
        -------
        dict[str, Any]
            Validation report
        """
        logger.info(f"Validating CatBoost model: {onnx_path}")

        original_model = joblib.load(original_path)
        onnx_session = ort.InferenceSession(onnx_path)

        if test_data is None:
            test_data = self._generate_test_data(num_features=18)

        original_preds = original_model.predict(test_data)

        input_name = onnx_session.get_inputs()[0].name
        onnx_preds = onnx_session.run(None, {input_name: test_data.astype(np.float32)})[0]

        if onnx_preds.ndim > 1:
            onnx_preds = onnx_preds.flatten()

        return self._compare_predictions(
            original_preds,
            onnx_preds,
            model_type="catboost",
            original_path=original_path,
            onnx_path=onnx_path,
        )

    def validate_ensemble(
        self,
        original_path: str,
        onnx_dir: str,
        ensemble_type: str = "ridge",
        test_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate stacking ensemble ONNX models against original.

        Validates both base models and meta-model predictions.

        Parameters
        ----------
        original_path : str
            Path to original ensemble model (.pkl)
        onnx_dir : str
            Directory containing ONNX base models and meta-model
        ensemble_type : str, default='ridge'
            Type of ensemble ('ridge' or 'lightgbm')
        test_data : np.ndarray, optional
            Test data for validation

        Returns
        -------
        dict[str, Any]
            Validation report for ensemble
        """
        logger.info(f"Validating {ensemble_type} ensemble: {onnx_dir}")

        ensemble = joblib.load(original_path)
        onnx_dir_path = Path(onnx_dir)

        if test_data is None:
            if hasattr(ensemble, "base_models_"):
                first_base = list(ensemble.base_models_.values())[0]
                if hasattr(first_base, "named_steps"):
                    first_base = first_base.named_steps["model"]
                test_data = self._generate_test_data(model=first_base)
            else:
                test_data = self._generate_test_data(num_features=18)

        original_preds = ensemble.predict(test_data)

        base_preds = []
        for name in ensemble.base_models_.keys():
            base_onnx_path = onnx_dir_path / f"{ensemble_type}_base_{name}.onnx"
            if not base_onnx_path.exists():
                logger.warning(f"Base model ONNX not found: {base_onnx_path}")
                continue

            session = ort.InferenceSession(str(base_onnx_path))
            input_name = session.get_inputs()[0].name
            pred = session.run(None, {input_name: test_data.astype(np.float32)})[0]

            if pred.ndim > 1:
                pred = pred.flatten()

            base_preds.append(pred)

        base_preds_array = np.column_stack(base_preds)

        meta_onnx_path = onnx_dir_path / f"{ensemble_type}_meta.onnx"
        if not meta_onnx_path.exists():
            raise FileNotFoundError(f"Meta-model ONNX not found: {meta_onnx_path}")

        meta_session = ort.InferenceSession(str(meta_onnx_path))
        input_name = meta_session.get_inputs()[0].name
        onnx_preds = meta_session.run(None, {input_name: base_preds_array.astype(np.float32)})[0]

        if onnx_preds.ndim > 1:
            onnx_preds = onnx_preds.flatten()

        return self._compare_predictions(
            original_preds,
            onnx_preds,
            model_type=f"{ensemble_type}_ensemble",
            original_path=original_path,
            onnx_path=str(onnx_dir),
        )

    def validate_chronos2(
        self,
        original_path: str,
        onnx_path: str,
        test_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate Chronos-2 ONNX model against original.

        Parameters
        ----------
        original_path : str
            Path to original Chronos-2 model
        onnx_path : str
            Path to ONNX model (.onnx)
        test_data : np.ndarray, optional
            Test time series data

        Returns
        -------
        dict[str, Any]
            Validation report

        Notes
        -----
        Chronos-2 validation may have higher tolerance due to model complexity.
        """
        logger.info(f"Validating Chronos-2 model: {onnx_path}")
        logger.warning("Chronos-2 validation is experimental and may not be fully accurate")

        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as e:
            logger.error("chronos-forecasting and torch required for validation")
            raise ImportError("chronos-forecasting and torch required") from e

        if "amazon" in str(original_path):
            pipeline = ChronosPipeline.from_pretrained(
                original_path,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        else:
            pipeline = ChronosPipeline.from_pretrained(
                str(original_path),
                device_map="cpu",
                torch_dtype=torch.float32,
            )

        if test_data is None:
            context_length = 512
            test_data = np.random.randn(1, context_length).astype(np.float32)

        with torch.no_grad():
            context_tensor = torch.from_numpy(test_data)
            original_output = pipeline.model(context_tensor)

            if isinstance(original_output, tuple):
                original_preds = original_output[0].numpy()
            else:
                original_preds = original_output.numpy()

        onnx_session = ort.InferenceSession(onnx_path)
        input_name = onnx_session.get_inputs()[0].name
        onnx_preds = onnx_session.run(None, {input_name: test_data})[0]

        return self._compare_predictions(
            original_preds.flatten()[:100],
            onnx_preds.flatten()[:100],
            model_type="chronos2",
            original_path=original_path,
            onnx_path=onnx_path,
        )

    def _generate_test_data(self, num_features: int = 18, model=None) -> np.ndarray:
        """
        Generate random test data for validation.

        Parameters
        ----------
        num_features : int, default=18
            Number of features
        model : Any, optional
            Model to infer feature count from

        Returns
        -------
        np.ndarray
            Test data array of shape (num_samples, num_features)
        """
        if model is not None and hasattr(model, "n_features_"):
            num_features = model.n_features_
        elif model is not None and hasattr(model, "n_features_in_"):
            num_features = model.n_features_in_

        np.random.seed(42)
        return np.random.randn(self.num_samples, num_features).astype(np.float32)

    def _compare_predictions(
        self,
        original_preds: np.ndarray,
        onnx_preds: np.ndarray,
        model_type: str,
        original_path: str,
        onnx_path: str,
    ) -> dict[str, Any]:
        """
        Compare predictions and generate validation report.

        Parameters
        ----------
        original_preds : np.ndarray
            Predictions from original model
        onnx_preds : np.ndarray
            Predictions from ONNX model
        model_type : str
            Type of model being validated
        original_path : str
            Path to original model
        onnx_path : str
            Path to ONNX model

        Returns
        -------
        dict[str, Any]
            Validation report with metrics and status
        """
        diff = np.abs(original_preds - onnx_preds)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        std_diff = np.std(diff)

        relative_error = np.mean(np.abs(diff / (np.abs(original_preds) + 1e-10))) * 100

        identical_count = np.sum(diff < 1e-10)
        within_tolerance = np.sum(diff < self.tolerance)

        status = "PASSED" if max_diff < self.tolerance else "FAILED"

        report = {
            "model_type": model_type,
            "original_path": original_path,
            "onnx_path": onnx_path,
            "validation_date": datetime.now().isoformat(),
            "num_samples": len(original_preds),
            "tolerance": self.tolerance,
            "metrics": {
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "median_diff": float(median_diff),
                "std_diff": float(std_diff),
                "relative_error_pct": float(relative_error),
                "identical_predictions": int(identical_count),
                "within_tolerance": int(within_tolerance),
                "within_tolerance_pct": float(within_tolerance / len(original_preds) * 100),
            },
            "status": status,
        }

        if status == "PASSED":
            logger.info(f"✓ Validation PASSED for {model_type} (max_diff={max_diff:.2e})")
        else:
            logger.error(f"✗ Validation FAILED for {model_type} (max_diff={max_diff:.2e})")

        return report

    def generate_report(self, validation_results: list[dict[str, Any]], output_path: str) -> None:
        """
        Generate consolidated validation report.

        Parameters
        ----------
        validation_results : list[dict[str, Any]]
            List of validation results from multiple models
        output_path : str
            Path to save report JSON
        """
        passed = sum(1 for r in validation_results if r["status"] == "PASSED")
        failed = sum(1 for r in validation_results if r["status"] == "FAILED")

        consolidated_report = {
            "report_date": datetime.now().isoformat(),
            "total_models": len(validation_results),
            "passed": passed,
            "failed": failed,
            "success_rate_pct": (
                (passed / len(validation_results) * 100) if validation_results else 0
            ),
            "tolerance": self.tolerance,
            "num_samples": self.num_samples,
            "results": validation_results,
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(consolidated_report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")
        logger.info(f"Summary: {passed}/{len(validation_results)} models passed validation")

    def validate_all_models(
        self,
        models_config: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        """
        Validate multiple models.

        Parameters
        ----------
        models_config : dict[str, dict[str, str]]
            Dictionary mapping model names to paths
            Format: {"model_name": {"original": "path", "onnx": "path"}}

        Returns
        -------
        list[dict[str, Any]]
            List of validation reports
        """
        results = []

        for model_name, paths in models_config.items():
            try:
                logger.info(f"Validating {model_name}...")

                if "xgboost" in model_name and "ensemble" not in model_name:
                    result = self.validate_xgboost(paths["original"], paths["onnx"])
                elif "lightgbm" in model_name and "ensemble" not in model_name:
                    result = self.validate_lightgbm(paths["original"], paths["onnx"])
                elif "catboost" in model_name:
                    result = self.validate_catboost(paths["original"], paths["onnx"])
                elif "ensemble" in model_name:
                    ensemble_type = "ridge" if "ridge" in model_name else "lightgbm"
                    result = self.validate_ensemble(
                        paths["original"],
                        paths["onnx"],
                        ensemble_type=ensemble_type,
                    )
                elif "chronos" in model_name:
                    result = self.validate_chronos2(paths["original"], paths["onnx"])
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to validate {model_name}: {e}")
                results.append(
                    {
                        "model_type": model_name,
                        "status": "ERROR",
                        "error": str(e),
                        "validation_date": datetime.now().isoformat(),
                    }
                )

        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    validator = ONNXValidator(tolerance=1e-5, num_samples=100)

    models_to_validate = {
        "xgboost": {
            "original": "models/baselines/xgboost_optimized.pkl",
            "onnx": "models/onnx/xgboost.onnx",
        },
        "lightgbm": {
            "original": "models/baselines/lightgbm_test_-gbm_v1.pkl",
            "onnx": "models/onnx/lightgbm.onnx",
        },
    }

    print("Validating ONNX models...")
    results = validator.validate_all_models(models_to_validate)

    validator.generate_report(results, "models/benchmarks/validation_report.json")

    print("\nValidation complete. Results saved to models/benchmarks/validation_report.json")
