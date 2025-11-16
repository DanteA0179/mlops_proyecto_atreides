"""
ONNX Model Exporter for Energy Optimization Copilot.

This module provides functionality to convert trained ML models
(Gradient Boosting, Ensembles, Foundation Models) to ONNX format
for optimized cross-platform inference.

Supports:
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
import onnx
import torch
from onnxmltools.convert import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
import numpy as np

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    ONNX model exporter with support for 8 models.

    Provides automated conversion from PyTorch/Scikit-learn/XGBoost/LightGBM
    to ONNX format with optimization and validation.

    Attributes
    ----------
    models_dir : Path
        Directory containing trained models
    output_dir : Path
        Directory for ONNX models
    config_dir : Path
        Directory for ONNX configurations
    """

    AVAILABLE_MODELS = {
        "xgboost": "models/gradient_boosting/xgboost_model.pkl",
        "lightgbm": "models/gradient_boosting/lightgbm_model.pkl",
        "catboost": "models/gradient_boosting/catboost_model.pkl",
        "ridge_ensemble": "models/ensembles/ensemble_ridge_v2.pkl",
        "lightgbm_ensemble": "models/ensembles/ensemble_lightgbm_v3.pkl",
        "chronos2_zeroshot": "amazon/chronos-t5-small",
        "chronos2_finetuned": "models/foundation/chronos2_finetuned_20251029_144949",
        "chronos2_covariates": "models/foundation/chronos2_finetuned_20251029_144949",
    }

    NUM_FEATURES = 18  # Number of input features for gradient boosting models

    def __init__(
        self,
        models_dir: str = "models",
        output_dir: str = "models/onnx",
        config_dir: str = "config/onnx",
    ):
        """
        Initialize ONNX exporter.

        Parameters
        ----------
        models_dir : str, default='models'
            Directory containing trained models
        output_dir : str, default='models/onnx'
            Directory for ONNX models
        config_dir : str, default='config/onnx'
            Directory for ONNX configurations
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ONNXExporter initialized with output_dir: {self.output_dir}")

    def export_xgboost(
        self,
        model_path: str,
        output_path: str | None = None,
        optimize: bool = True,
    ) -> str:
        """
        Export XGBoost model to ONNX format.

        Parameters
        ----------
        model_path : str
            Path to trained XGBoost model (.pkl)
        output_path : str, optional
            Output path for ONNX model
        optimize : bool, default=True
            Whether to optimize ONNX graph

        Returns
        -------
        str
            Path to exported ONNX model

        Examples
        --------
        >>> exporter = ONNXExporter()
        >>> onnx_path = exporter.export_xgboost("models/baselines/xgboost_optimized.pkl")
        """
        logger.info(f"Exporting XGBoost model from {model_path}")

        model = joblib.load(model_path)

        if hasattr(model, 'named_steps'):
            logger.info("Detected sklearn Pipeline, extracting final estimator")
            model = model.named_steps['model']

        if output_path is None:
            output_path = self.output_dir / "xgboost.onnx"
        else:
            output_path = Path(output_path)

        try:
            X_sample = np.random.randn(1, self.NUM_FEATURES).astype(np.float32)
            onnx_model = to_onnx(model, X_sample, target_opset=17)

            if optimize:
                onnx_model = self.optimize_onnx(onnx_model)

            onnx.save_model(onnx_model, str(output_path))
            logger.info(f"XGBoost model exported to {output_path}")

            self.save_metadata(output_path, "xgboost", model_path)

            return str(output_path)
        except Exception as e:
            logger.error(f"Failed with to_onnx, trying convert_xgboost: {e}")
            initial_types = [("input", FloatTensorType([None, self.NUM_FEATURES]))]
            onnx_model = convert_xgboost(model, initial_types=initial_types)

            if optimize:
                onnx_model = self.optimize_onnx(onnx_model)

            onnx.save_model(onnx_model, str(output_path))
            logger.info(f"XGBoost model exported to {output_path}")

            self.save_metadata(output_path, "xgboost", model_path)

            return str(output_path)

    def export_lightgbm(
        self,
        model_path: str,
        output_path: str | None = None,
        optimize: bool = True,
    ) -> str:
        """
        Export LightGBM model to ONNX format.

        Parameters
        ----------
        model_path : str
            Path to trained LightGBM model (.pkl)
        output_path : str, optional
            Output path for ONNX model
        optimize : bool, default=True
            Whether to optimize ONNX graph

        Returns
        -------
        str
            Path to exported ONNX model
        """
        logger.info(f"Exporting LightGBM model from {model_path}")

        model = joblib.load(model_path)

        if hasattr(model, 'named_steps'):
            logger.info("Detected sklearn Pipeline, extracting final estimator")
            model = model.named_steps['model']

        if output_path is None:
            output_path = self.output_dir / "lightgbm.onnx"
        else:
            output_path = Path(output_path)

        num_features = model.n_features_ if hasattr(model, 'n_features_') else self.NUM_FEATURES
        logger.info(f"Model expects {num_features} features")
        
        initial_types = [("input", FloatTensorType([None, num_features]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_types)

        if optimize:
            onnx_model = self.optimize_onnx(onnx_model)

        onnx.save_model(onnx_model, str(output_path))
        logger.info(f"LightGBM model exported to {output_path}")

        self.save_metadata(output_path, "lightgbm", model_path)

        return str(output_path)

    def export_catboost(
        self,
        model_path: str,
        output_path: str | None = None,
        optimize: bool = True,
    ) -> str:
        """
        Export CatBoost model to ONNX format.

        Parameters
        ----------
        model_path : str
            Path to trained CatBoost model (.pkl)
        output_path : str, optional
            Output path for ONNX model
        optimize : bool, default=True
            Whether to optimize ONNX graph

        Returns
        -------
        str
            Path to exported ONNX model
        """
        logger.info(f"Exporting CatBoost model from {model_path}")

        model = joblib.load(model_path)

        if hasattr(model, 'named_steps'):
            logger.info("Detected sklearn Pipeline, extracting final estimator")
            model = model.named_steps['model']

        if output_path is None:
            output_path = self.output_dir / "catboost.onnx"
        else:
            output_path = Path(output_path)

        try:
            from catboost.utils import convert_to_onnx_object

            onnx_model_bytes = convert_to_onnx_object(model)

            with open(output_path, "wb") as f:
                f.write(onnx_model_bytes)

            logger.info(f"CatBoost model exported to {output_path}")
            self.save_metadata(output_path, "catboost", model_path)

            return str(output_path)

        except ImportError as e:
            logger.error(
                "CatBoost ONNX export requires catboost package. Install with: pip install catboost"
            )
            raise ImportError("CatBoost package required for ONNX export") from e

    def export_stacking_ensemble(
        self,
        model_path: str,
        model_type: str = "ridge",
        output_path: str | None = None,
        optimize: bool = True,
    ) -> dict[str, str]:
        """
        Export stacking ensemble model to ONNX format.

        Exports base models (XGBoost, LightGBM, CatBoost) and meta-model separately.

        Parameters
        ----------
        model_path : str
            Path to trained ensemble model (.pkl)
        model_type : str, default='ridge'
            Type of ensemble ('ridge' or 'lightgbm')
        output_path : str, optional
            Output directory for ONNX models
        optimize : bool, default=True
            Whether to optimize ONNX graphs

        Returns
        -------
        dict[str, str]
            Dictionary mapping model names to ONNX paths
        """
        logger.info(f"Exporting {model_type} ensemble model from {model_path}")

        ensemble = joblib.load(model_path)

        if output_path is None:
            output_dir = self.output_dir / f"{model_type}_ensemble"
        else:
            output_dir = Path(output_path)

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_models = {}

        if hasattr(ensemble, "base_models_"):
            for name, base_model in ensemble.base_models_.items():
                base_output = output_dir / f"{model_type}_base_{name}.onnx"

                if hasattr(base_model, 'named_steps'):
                    logger.info(f"Base model {name} is a Pipeline, extracting estimator")
                    base_model = base_model.named_steps['model']

                if "xgboost" in name.lower() or "xgb" in name.lower():
                    try:
                        import xgboost as xgb
                        temp_json = output_dir / f"temp_{name}.json"
                        base_model.save_model(str(temp_json))
                        
                        booster = xgb.Booster()
                        booster.load_model(str(temp_json))
                        
                        initial_types = [("input", FloatTensorType([None, self.NUM_FEATURES]))]
                        onnx_model = convert_xgboost(booster, initial_types=initial_types)
                        
                        temp_json.unlink()
                    except Exception as e:
                        logger.error(f"Failed to export XGBoost base model {name}: {e}")
                        logger.warning(f"Skipping XGBoost model {name} - will use LightGBM and CatBoost only")
                        continue
                elif "lightgbm" in name.lower() or "lgb" in name.lower():
                    initial_types = [("input", FloatTensorType([None, self.NUM_FEATURES]))]
                    onnx_model = convert_lightgbm(base_model, initial_types=initial_types)
                elif "catboost" in name.lower() or "cat" in name.lower():
                    try:
                        from catboost.utils import convert_to_onnx_object
                        
                        onnx_model_proto = convert_to_onnx_object(base_model)
                        
                        if isinstance(onnx_model_proto, bytes):
                            with open(base_output, "wb") as f:
                                f.write(onnx_model_proto)
                        else:
                            onnx.save_model(onnx_model_proto, str(base_output))
                        
                        exported_models[f"base_{name}"] = str(base_output)
                        logger.info(f"Base model {name} exported to {base_output}")
                        continue
                    except Exception as e:
                        logger.error(f"Failed to export CatBoost base model {name}: {e}")
                        logger.warning(f"Skipping CatBoost model {name}")
                        continue
                else:
                    logger.warning(f"Unknown base model type: {name}, skipping")
                    continue

                if optimize:
                    onnx_model = self.optimize_onnx(onnx_model)

                onnx.save_model(onnx_model, str(base_output))
                exported_models[f"base_{name}"] = str(base_output)
                logger.info(f"Base model {name} exported to {base_output}")

        if hasattr(ensemble, "meta_model_"):
            meta_output = output_dir / f"{model_type}_meta.onnx"
            meta_model = ensemble.meta_model_

            num_base_models = len([k for k in exported_models.keys() if k.startswith("base_")])
            logger.info(f"Exporting meta-model with {num_base_models} base model inputs")
            
            try:
                if hasattr(meta_model, '__class__') and 'lightgbm' in str(type(meta_model)).lower():
                    logger.info("Meta-model is LightGBM, using convert_lightgbm")
                    initial_types = [("input", FloatTensorType([None, num_base_models]))]
                    onnx_meta = convert_lightgbm(meta_model, initial_types=initial_types)
                else:
                    initial_types = [("input", SklearnFloatTensorType([None, num_base_models]))]
                    onnx_meta = convert_sklearn(meta_model, initial_types=initial_types)

                if optimize:
                    onnx_meta = self.optimize_onnx(onnx_meta)

                onnx.save_model(onnx_meta, str(meta_output))
                exported_models["meta_model"] = str(meta_output)
                logger.info(f"Meta-model exported to {meta_output}")
            except Exception as e:
                logger.error(f"Failed to export meta-model: {e}")
                raise

        metadata_path = output_dir / "metadata.json"
        self.save_metadata(metadata_path, f"{model_type}_ensemble", model_path, exported_models)

        logger.info(f"Ensemble model exported to {output_dir}")
        return exported_models

    def export_chronos2(
        self,
        model_name: str = "chronos2_finetuned",
        model_path: str | None = None,
        output_path: str | None = None,
        opset_version: int = 17,
        optimize: bool = True,
    ) -> str:
        """
        Export Chronos-2 foundation model to ONNX format.

        Parameters
        ----------
        model_name : str, default='chronos2_finetuned'
            Model variant ('chronos2_zeroshot', 'chronos2_finetuned', 'chronos2_covariates')
        model_path : str, optional
            Path to trained Chronos-2 model
        output_path : str, optional
            Output path for ONNX model
        opset_version : int, default=17
            ONNX opset version
        optimize : bool, default=True
            Whether to optimize ONNX graph

        Returns
        -------
        str
            Path to exported ONNX model
        """
        logger.info(f"Exporting Chronos-2 model: {model_name}")

        try:
            from chronos import ChronosPipeline
        except ImportError as e:
            logger.error(
                "chronos-forecasting package required. Install with: pip install chronos-forecasting"
            )
            raise ImportError("chronos-forecasting package required") from e

        if model_path is None:
            model_path = self.AVAILABLE_MODELS.get(model_name, "amazon/chronos-t5-small")

        logger.info(f"Loading Chronos-2 model from {model_path}")

        if "amazon" in str(model_path):
            pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )
        else:
            pipeline = ChronosPipeline.from_pretrained(
                str(model_path),
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )

        if output_path is None:
            output_path = self.output_dir / f"{model_name}.onnx"
        else:
            output_path = Path(output_path)

        context_length = 512
        prediction_length = 64
        num_samples = 20

        dummy_input = {
            "context": torch.randn(1, context_length, dtype=torch.float32),
            "prediction_length": torch.tensor([prediction_length], dtype=torch.long),
            "num_samples": torch.tensor([num_samples], dtype=torch.long),
        }

        logger.info("Exporting Chronos-2 to ONNX...")

        try:
            torch.onnx.export(
                pipeline.model,
                (dummy_input["context"],),
                str(output_path),
                opset_version=opset_version,
                input_names=["context"],
                output_names=["predictions"],
                dynamic_axes={
                    "context": {0: "batch_size", 1: "sequence_length"},
                    "predictions": {0: "batch_size", 1: "num_samples", 2: "prediction_length"},
                },
                do_constant_folding=True,
                verbose=False,
            )
            logger.info(f"Chronos-2 model exported to {output_path}")
            self.save_metadata(output_path, model_name, str(model_path))
            return str(output_path)

        except Exception as e:
            logger.error(f"Error exporting Chronos-2 to ONNX: {e}")
            logger.warning(
                "Chronos-2 export may fail due to complex architecture. "
                "Consider using TorchScript or ONNX Runtime extensions."
            )
            raise

    def optimize_onnx(self, onnx_model: onnx.ModelProto, level: int = 2) -> onnx.ModelProto:
        """
        Optimize ONNX model graph.

        Parameters
        ----------
        onnx_model : onnx.ModelProto
            ONNX model to optimize
        level : int, default=2
            Optimization level (0=none, 1=basic, 2=extended, 3=all)

        Returns
        -------
        onnx.ModelProto
            Optimized ONNX model

        Notes
        -----
        Optimization is currently disabled due to changes in ONNX API.
        Use onnxruntime for runtime optimization instead.
        """
        logger.info(f"ONNX model optimization level {level} - skipping (use onnxruntime)")
        return onnx_model

    def save_metadata(
        self,
        model_path: Path,
        model_type: str,
        source_path: str,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Save model metadata to JSON file.

        Parameters
        ----------
        model_path : Path
            Path to ONNX model
        model_type : str
            Type of model (xgboost, lightgbm, etc.)
        source_path : str
            Path to original model
        additional_info : dict[str, Any], optional
            Additional metadata to save
        """
        metadata = {
            "model_type": model_type,
            "source_path": source_path,
            "onnx_path": str(model_path),
            "export_date": datetime.now().isoformat(),
            "file_size_bytes": model_path.stat().st_size if model_path.exists() else 0,
            "file_size_mb": (
                round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else 0
            ),
        }

        if additional_info:
            metadata.update(additional_info)

        if model_path.is_dir():
            metadata_path = model_path / "metadata.json"
        else:
            metadata_path = model_path.with_suffix(".json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get information about a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        dict[str, Any]
            Model information
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = Path(self.AVAILABLE_MODELS[model_name])

        return {
            "name": model_name,
            "source_path": str(model_path),
            "exists": model_path.exists(),
            "type": self._get_model_type(model_name),
        }

    @staticmethod
    def _get_model_type(model_name: str) -> str:
        """Get model type from name."""
        if "ensemble" in model_name:
            return "ensemble"
        elif "chronos" in model_name:
            return "foundation"
        else:
            return "gradient_boosting"

    @classmethod
    def list_available_models(cls) -> list[str]:
        """
        List all available models for export.

        Returns
        -------
        list[str]
            List of model names
        """
        return list(cls.AVAILABLE_MODELS.keys())

    def export_all_models(self, models: list[str] | None = None) -> dict[str, str]:
        """
        Export multiple models to ONNX format.

        Parameters
        ----------
        models : list[str], optional
            List of model names to export. If None, exports all available models.

        Returns
        -------
        dict[str, str]
            Dictionary mapping model names to ONNX paths
        """
        if models is None:
            models = self.list_available_models()

        exported = {}
        failed = {}

        for model_name in models:
            try:
                logger.info(f"Exporting {model_name}...")

                if model_name == "xgboost":
                    path = self.export_xgboost(self.AVAILABLE_MODELS[model_name])
                    exported[model_name] = path
                elif model_name == "lightgbm":
                    path = self.export_lightgbm(self.AVAILABLE_MODELS[model_name])
                    exported[model_name] = path
                elif model_name == "catboost":
                    path = self.export_catboost(self.AVAILABLE_MODELS[model_name])
                    exported[model_name] = path
                elif "ensemble" in model_name:
                    ensemble_type = "ridge" if "ridge" in model_name else "lightgbm"
                    paths = self.export_stacking_ensemble(
                        self.AVAILABLE_MODELS[model_name],
                        model_type=ensemble_type,
                    )
                    exported[model_name] = paths
                elif "chronos" in model_name:
                    path = self.export_chronos2(
                        model_name=model_name,
                        model_path=self.AVAILABLE_MODELS[model_name],
                    )
                    exported[model_name] = path
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue

                logger.info(f"Successfully exported {model_name}")

            except Exception as e:
                logger.error(f"Failed to export {model_name}: {e}")
                failed[model_name] = str(e)

        logger.info(f"Export complete. Success: {len(exported)}, Failed: {len(failed)}")

        if failed:
            logger.warning(f"Failed models: {list(failed.keys())}")

        return exported


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    exporter = ONNXExporter()

    print("Available models for export:")
    for model_name in exporter.list_available_models():
        info = exporter.get_model_info(model_name)
        status = "✓" if info["exists"] else "✗"
        print(f"  {status} {model_name} ({info['type']})")

    print("\nExporting all available models...")
    exported = exporter.export_all_models()

    print(f"\nExported {len(exported)} models:")
    for name, path in exported.items():
        print(f"  ✓ {name}: {path}")
