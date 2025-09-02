"""
Singleton model manager that loads all EWCL models once at application startup.
This ensures models stay in memory between requests and eliminates cold start delays.
"""

import os
import pickle
import joblib
import sys
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

# Add backend_bundle to path for feature extractor
_BACKEND_BUNDLE_PATH = str(Path(__file__).resolve().parents[2] / "backend_bundle")
if _BACKEND_BUNDLE_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_BUNDLE_PATH)

try:
    from meta.ewcl_feature_extractor_v2 import EWCLFeatureExtractor
    FEATURE_EXTRACTOR = EWCLFeatureExtractor()
except ImportError as e:
    print(f"Warning: Could not import EWCLFeatureExtractor: {e}")
    FEATURE_EXTRACTOR = None

class ModelSingleton:
    """Singleton class that loads and holds all EWCL models in memory."""
    
    _instance: Optional['ModelSingleton'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models: Dict[str, Any] = {}
            self.feature_orders: Dict[str, list] = {}
            self._load_all_models()
            self._initialized = True
    
    def _load_model_from_path(self, name: str, path: str) -> Optional[Any]:
        """Load a single model from the given path."""
        if not path or not os.path.exists(path):
            print(f"[{name}] Model path not found: {path}")
            return None
        
        try:
            # Try joblib first, fall back to pickle
            try:
                model = joblib.load(path)
            except Exception:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            
            print(f"[{name}] ✅ Loaded model from {path}")
            return model
        except Exception as e:
            print(f"[{name}] ❌ Failed to load model from {path}: {e}")
            return None
    
    def _load_all_models(self):
        """Load all EWCL models from environment variables."""
        print("🚀 Loading all EWCL models into memory...")
        
        # Model specifications with their environment variables and feature orders
        model_specs = {
            "ewclv1": {
                "env_var": "EWCLV1_MODEL_PATH",
                "features": [
                    "is_unknown_aa", "hydropathy", "polarity", "vdw_volume", "flexibility", "bulkiness", "helix_prop", "sheet_prop", "charge_pH7", "scd_local",
                    "hydro_w5_mean", "hydro_w5_std", "hydro_w5_min", "hydro_w5_max", "polar_w5_mean", "polar_w5_std", "polar_w5_min", "polar_w5_max", "vdw_w5_mean", "vdw_w5_std",
                    "vdw_w5_min", "vdw_w5_max", "flex_w5_mean", "flex_w5_std", "flex_w5_min", "flex_w5_max", "bulk_w5_mean", "bulk_w5_std", "bulk_w5_min", "bulk_w5_max",
                    "helix_prop_w5_mean", "helix_prop_w5_std", "helix_prop_w5_min", "helix_prop_w5_max", "sheet_prop_w5_mean", "sheet_prop_w5_std", "sheet_prop_w5_min", "sheet_prop_w5_max", "charge_w5_mean", "charge_w5_std",
                    "charge_w5_min", "charge_w5_max", "entropy_w5", "low_complex_w5", "comp_bias_w5", "uversky_dist_w5", "hydro_w11_mean", "hydro_w11_std", "hydro_w11_min", "hydro_w11_max",
                    "polar_w11_mean", "polar_w11_std", "polar_w11_min", "polar_w11_max", "vdw_w11_mean", "vdw_w11_std", "vdw_w11_min", "vdw_w11_max", "flex_w11_mean", "flex_w11_std",
                    "flex_w11_min", "flex_w11_max", "bulk_w11_mean", "bulk_w11_std", "bulk_w11_min", "bulk_w11_max", "helix_prop_w11_mean", "helix_prop_w11_std", "helix_prop_w11_min", "helix_prop_w11_max",
                    "sheet_prop_w11_mean", "sheet_prop_w11_std", "sheet_prop_w11_min", "sheet_prop_w11_max", "charge_w11_mean", "charge_w11_std", "charge_w11_min", "charge_w11_max", "entropy_w11", "low_complex_w11",
                    "comp_bias_w11", "uversky_dist_w11", "hydro_w25_mean", "hydro_w25_std", "hydro_w25_min", "hydro_w25_max", "polar_w25_mean", "polar_w25_std", "polar_w25_min", "polar_w25_max",
                    "vdw_w25_mean", "vdw_w25_std", "vdw_w25_min", "vdw_w25_max", "flex_w25_mean", "flex_w25_std", "flex_w25_min", "flex_w25_max", "bulk_w25_mean", "bulk_w25_std",
                    "bulk_w25_min", "bulk_w25_max", "helix_prop_w25_mean", "helix_prop_w25_std", "helix_prop_w25_min", "helix_prop_w25_max", "sheet_prop_w25_mean", "sheet_prop_w25_std", "sheet_prop_w25_min", "sheet_prop_w25_max",
                    "charge_w25_mean", "charge_w25_std", "charge_w25_min", "charge_w25_max", "entropy_w25", "low_complex_w25", "comp_bias_w25", "uversky_dist_w25", "hydro_w50_mean", "hydro_w50_std",
                    "hydro_w50_min", "hydro_w50_max", "polar_w50_mean", "polar_w50_std", "polar_w50_min", "polar_w50_max", "vdw_w50_mean", "vdw_w50_std", "vdw_w50_min", "vdw_w50_max",
                    "flex_w50_mean", "flex_w50_std", "flex_w50_min", "flex_w50_max", "bulk_w50_mean", "bulk_w50_std", "bulk_w50_min", "bulk_w50_max", "helix_prop_w50_mean", "helix_prop_w50_std",
                    "helix_prop_w50_min", "helix_prop_w50_max", "sheet_prop_w50_mean", "sheet_prop_w50_std", "sheet_prop_w50_min", "sheet_prop_w50_max", "charge_w50_mean", "charge_w50_std", "charge_w50_min", "charge_w50_max",
                    "entropy_w50", "low_complex_w50", "comp_bias_w50", "uversky_dist_w50", "hydro_w100_mean", "hydro_w100_std", "hydro_w100_min", "hydro_w100_max", "polar_w100_mean", "polar_w100_std",
                    "polar_w100_min", "polar_w100_max", "vdw_w100_mean", "vdw_w100_std", "vdw_w100_min", "vdw_w100_max", "flex_w100_mean", "flex_w100_std", "flex_w100_min", "flex_w100_max",
                    "bulk_w100_mean", "bulk_w100_std", "bulk_w100_min", "bulk_w100_max", "helix_prop_w100_mean", "helix_prop_w100_std", "helix_prop_w100_min", "helix_prop_w100_max", "sheet_prop_w100_mean", "sheet_prop_w100_std",
                    "sheet_prop_w100_min", "sheet_prop_w100_max", "charge_w100_mean", "charge_w100_std", "charge_w100_min", "charge_w100_max", "entropy_w100", "low_complex_w100", "comp_bias_w100", "uversky_dist_w100",
                    "comp_D", "comp_Y", "comp_F", "comp_M", "comp_V", "comp_R", "comp_P", "comp_A", "comp_L", "comp_I",
                    "comp_T", "comp_W", "comp_Q", "comp_N", "comp_K", "comp_E", "comp_G", "comp_S", "comp_H", "comp_C",
                    "comp_frac_aromatic", "comp_frac_positive", "comp_frac_negative", "comp_frac_polar", "comp_frac_aliphatic", "comp_frac_proline", "comp_frac_glycine", "in_poly_P_run_ge3", "in_poly_E_run_ge3", "in_poly_K_run_ge3",
                    "in_poly_Q_run_ge3", "in_poly_S_run_ge3", "in_poly_G_run_ge3", "in_poly_D_run_ge3", "in_poly_N_run_ge3", "A", "R", "N", "D", "C",
                    "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P",
                    "S", "T", "W", "Y", "V", "pssm_entropy", "pssm_max_score", "pssm_variance", "has_pssm_data"
                ]
            },
            "ewclv1-m": {
                "env_var": "EWCLV1_M_MODEL_PATH",
                "features": [
                    "is_unknown_aa", "hydropathy", "polarity", "vdw_volume", "flexibility", "bulkiness", "helix_prop", "sheet_prop", "charge_pH7", "scd_local",
                    "hydro_w5_mean", "hydro_w5_std", "hydro_w5_min", "hydro_w5_max", "polar_w5_mean", "polar_w5_std", "polar_w5_min", "polar_w5_max", "vdw_w5_mean", "vdw_w5_std",
                    "vdw_w5_min", "vdw_w5_max", "flex_w5_mean", "flex_w5_std", "flex_w5_min", "flex_w5_max", "bulk_w5_mean", "bulk_w5_std", "bulk_w5_min", "bulk_w5_max",
                    "helix_prop_w5_mean", "helix_prop_w5_std", "helix_prop_w5_min", "helix_prop_w5_max", "sheet_prop_w5_mean", "sheet_prop_w5_std", "sheet_prop_w5_min", "sheet_prop_w5_max", "charge_w5_mean", "charge_w5_std",
                    "charge_w5_min", "charge_w5_max", "entropy_w5", "low_complex_w5", "comp_bias_w5", "uversky_dist_w5", "hydro_w11_mean", "hydro_w11_std", "hydro_w11_min", "hydro_w11_max",
                    "polar_w11_mean", "polar_w11_std", "polar_w11_min", "polar_w11_max", "vdw_w11_mean", "vdw_w11_std", "vdw_w11_min", "vdw_w11_max", "flex_w11_mean", "flex_w11_std",
                    "flex_w11_min", "flex_w11_max", "bulk_w11_mean", "bulk_w11_std", "bulk_w11_min", "bulk_w11_max", "helix_prop_w11_mean", "helix_prop_w11_std", "helix_prop_w11_min", "helix_prop_w11_max",
                    "sheet_prop_w11_mean", "sheet_prop_w11_std", "sheet_prop_w11_min", "sheet_prop_w11_max", "charge_w11_mean", "charge_w11_std", "charge_w11_min", "charge_w11_max", "entropy_w11", "low_complex_w11",
                    "comp_bias_w11", "uversky_dist_w11", "hydro_w25_mean", "hydro_w25_std", "hydro_w25_min", "hydro_w25_max", "polar_w25_mean", "polar_w25_std", "polar_w25_min", "polar_w25_max",
                    "vdw_w25_mean", "vdw_w25_std", "vdw_w25_min", "vdw_w25_max", "flex_w25_mean", "flex_w25_std", "flex_w25_min", "flex_w25_max", "bulk_w25_mean", "bulk_w25_std",
                    "bulk_w25_min", "bulk_w25_max", "helix_prop_w25_mean", "helix_prop_w25_std", "helix_prop_w25_min", "helix_prop_w25_max", "sheet_prop_w25_mean", "sheet_prop_w25_std", "sheet_prop_w25_min", "sheet_prop_w25_max",
                    "charge_w25_mean", "charge_w25_std", "charge_w25_min", "charge_w25_max", "entropy_w25", "low_complex_w25", "comp_bias_w25", "uversky_dist_w25", "hydro_w50_mean", "hydro_w50_std",
                    "hydro_w50_min", "hydro_w50_max", "polar_w50_mean", "polar_w50_std", "polar_w50_min", "polar_w50_max", "vdw_w50_mean", "vdw_w50_std", "vdw_w50_min", "vdw_w50_max",
                    "flex_w50_mean", "flex_w50_std", "flex_w50_min", "flex_w50_max", "bulk_w50_mean", "bulk_w50_std", "bulk_w50_min", "bulk_w50_max", "helix_prop_w50_mean", "helix_prop_w50_std",
                    "helix_prop_w50_min", "helix_prop_w50_max", "sheet_prop_w50_mean", "sheet_prop_w50_std", "sheet_prop_w50_min", "sheet_prop_w50_max", "charge_w50_mean", "charge_w50_std", "charge_w50_min", "charge_w50_max",
                    "entropy_w50", "low_complex_w50", "comp_bias_w50", "uversky_dist_w50", "hydro_w100_mean", "hydro_w100_std", "hydro_w100_min", "hydro_w100_max", "polar_w100_mean", "polar_w100_std",
                    "polar_w100_min", "polar_w100_max", "vdw_w100_mean", "vdw_w100_std", "vdw_w100_min", "vdw_w100_max", "flex_w100_mean", "flex_w100_std", "flex_w100_min", "flex_w100_max",
                    "bulk_w100_mean", "bulk_w100_std", "bulk_w100_min", "bulk_w100_max", "helix_prop_w100_mean", "helix_prop_w100_std", "helix_prop_w100_min", "helix_prop_w100_max", "sheet_prop_w100_mean", "sheet_prop_w100_std",
                    "sheet_prop_w100_min", "sheet_prop_w100_max", "charge_w100_mean", "charge_w100_std", "charge_w100_min", "charge_w100_max", "entropy_w100", "low_complex_w100", "comp_bias_w100", "uversky_dist_w100",
                    "comp_D", "comp_Y", "comp_F", "comp_M", "comp_V", "comp_R", "comp_P", "comp_A", "comp_L", "comp_I",
                    "comp_T", "comp_W", "comp_Q", "comp_N", "comp_K", "comp_E", "comp_G", "comp_S", "comp_H", "comp_C",
                    "comp_frac_aromatic", "comp_frac_positive", "comp_frac_negative", "comp_frac_polar", "comp_frac_aliphatic", "comp_frac_proline", "comp_frac_glycine", "in_poly_P_run_ge3", "in_poly_E_run_ge3", "in_poly_K_run_ge3",
                    "in_poly_Q_run_ge3", "in_poly_S_run_ge3", "in_poly_G_run_ge3", "in_poly_D_run_ge3", "in_poly_N_run_ge3", "A", "R", "N", "D", "C",
                    "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P",
                    "S", "T", "W", "Y", "V", "pssm_entropy", "pssm_max_score", "pssm_variance", "pssm_native", "pssm_top1",
                    "pssm_top2", "pssm_gap", "pssm_sum_hydrophobic", "pssm_sum_polar", "pssm_sum_charged"
                ]
            },
            "ewclv1-p3": {
                "env_var": "EWCLV1_P3_MODEL_PATH",
                "features": []  # Will be loaded from feature info if available
            },
            "ewclv1-c": {
                "env_var": "EWCLV1_C_MODEL_PATH", 
                "features": []  # Will be loaded from feature info if available
            }
        }
        
        loaded_count = 0
        for model_name, spec in model_specs.items():
            model_path = os.environ.get(spec["env_var"])
            model = self._load_model_from_path(model_name, model_path)
            
            if model is not None:
                self.models[model_name] = model
                self.feature_orders[model_name] = spec["features"]
                loaded_count += 1
        
        print(f"✅ Successfully loaded {loaded_count}/4 EWCL models into memory")
        if loaded_count < 4:
            missing = [name for name in model_specs.keys() if name not in self.models]
            print(f"⚠️  Missing models: {missing}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self.models.get(name)
    
    def get_feature_order(self, name: str) -> list:
        """Get the feature order for a model."""
        return self.feature_orders.get(name, [])
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded."""
        return name in self.models
    
    def get_loaded_models(self) -> list:
        """Get list of loaded model names."""
        return list(self.models.keys())
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using the specified model."""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Ensure feature order matches what the model expects
        feature_order = self.get_feature_order(model_name)
        if feature_order:
            # Add missing features with default values
            for feat in feature_order:
                if feat not in X.columns:
                    X[feat] = 0.0
            # Reorder to match model expectation
            X = X[feature_order]
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
        else:
            return np.clip(model.predict(X), 0.0, 1.0)


# Global singleton instance
_model_manager: Optional[ModelSingleton] = None

def get_model_manager() -> ModelSingleton:
    """Get the global model manager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelSingleton()
    return _model_manager

def get_model(name: str) -> Optional[Any]:
    """Get a loaded model by name."""
    return get_model_manager().get_model(name)

def predict_with_model(model_name: str, X: pd.DataFrame) -> np.ndarray:
    """Make predictions using a loaded model."""
    return get_model_manager().predict_proba(model_name, X)

def get_feature_extractor():
    """Get the feature extractor instance."""
    return FEATURE_EXTRACTOR