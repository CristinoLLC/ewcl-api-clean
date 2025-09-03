"""
Simple model manager for Railway deployment - no singleton pattern needed.
Models are loaded once at startup and stored in a simple global dict.
"""

import os
import joblib  # Changed from pickle to joblib
import json
from typing import Optional, Any, Dict
import threading

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model registry with relative paths from project root
MODEL_REGISTRY = {
    "ewclv1": os.path.join(BASE_DIR, "models", "disorder", "ewclv1.pkl"),
    "ewclv1-m": os.path.join(BASE_DIR, "models", "disorder", "ewclv1-M.pkl"), 
    "ewclv1-p3": os.path.join(BASE_DIR, "models", "pdb", "ewclv1p3.pkl"),
    "ewclv1-c": os.path.join(BASE_DIR, "models", "clinvar", "ewclv1-C.pkl"),
}

# Configuration registry (non-model files)
CONFIG_REGISTRY = {
    "ewclv1-c-features": os.path.join(BASE_DIR, "backend", "config", "ewclv1-c_features.json")
}

# Cache for loaded models
_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()

def get_model(model_name: str) -> Optional[Any]:
    """Get a model from the registry, loading it if not already cached."""
    with _cache_lock:
        # Check cache first
        if model_name in _model_cache:
            return _model_cache[model_name]
        
        # Try to load from model registry
        if model_name in MODEL_REGISTRY:
            model_path = MODEL_REGISTRY[model_name]
            try:
                if os.path.exists(model_path):
                    # Use joblib to load the model
                    model = joblib.load(model_path)
                    _model_cache[model_name] = model
                    print(f"[model_manager] Loaded model {model_name} from {model_path}")
                    return model
                else:
                    print(f"[model_manager] Model file not found: {model_path}")
                    return None
            except Exception as e:
                print(f"[model_manager] Error loading model {model_name}: {e}")
                return None
        
        # Try to load from config registry  
        elif model_name in CONFIG_REGISTRY:
            config_path = CONFIG_REGISTRY[model_name]
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    _model_cache[model_name] = config
                    print(f"[model_manager] Loaded config {model_name} from {config_path}")
                    return config
                else:
                    print(f"[model_manager] Config file not found: {config_path}")
                    return None
            except Exception as e:
                print(f"[model_manager] Error loading config {model_name}: {e}")
                return None
        
        else:
            print(f"[model_manager] Unknown model/config: {model_name}")
            return None

def get_available_models() -> Dict[str, str]:
    """Get a dictionary of available models and their status."""
    status = {}
    
    # Check models
    for model_name, model_path in MODEL_REGISTRY.items():
        if os.path.exists(model_path):
            status[model_name] = "available"
        else:
            status[model_name] = "not_found"
    
    # Check configs
    for config_name, config_path in CONFIG_REGISTRY.items():
        if os.path.exists(config_path):
            status[config_name] = "available"
        else:
            status[config_name] = "not_found"
    
    return status

def clear_cache():
    """Clear the model cache."""
    with _cache_lock:
        _model_cache.clear()
        print("[model_manager] Cache cleared")