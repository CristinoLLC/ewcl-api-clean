"""
Singleton model manager that loads all EWCL models once at application startup.
This ensures models stay in memory between requests and eliminates cold start delays.
"""

import os
import pickle
import joblib
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

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
        
        # Simple model specifications
        model_specs = {
            "ewclv1": "EWCLV1_MODEL_PATH",
            "ewclv1-m": "EWCLV1_M_MODEL_PATH", 
            "ewclv1-p3": "EWCLV1_P3_MODEL_PATH",
            "ewclv1-c": "EWCLV1_C_MODEL_PATH"
        }
        
        loaded_count = 0
        for model_name, env_var in model_specs.items():
            model_path = os.environ.get(env_var)
            model = self._load_model_from_path(model_name, model_path)
            
            if model is not None:
                self.models[model_name] = model
                loaded_count += 1
        
        print(f"✅ Successfully loaded {loaded_count}/4 EWCL models into memory")
        if loaded_count < 4:
            missing = [name for name in model_specs.keys() if name not in self.models]
            print(f"⚠️  Missing models: {missing}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self.models.get(name)
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded."""
        return name in self.models
    
    def get_loaded_models(self) -> list:
        """Get list of loaded model names."""
        return list(self.models.keys())

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

# For backwards compatibility, but Railway won't use these
class ModelManager:
    """Legacy ModelManager class for backwards compatibility."""
    
    def __init__(self):
        self.singleton = get_model_manager()
    
    async def initialize(self):
        """Initialize - no-op since singleton loads on creation."""
        pass
    
    def get_model(self, name: str):
        return self.singleton.get_model(name)