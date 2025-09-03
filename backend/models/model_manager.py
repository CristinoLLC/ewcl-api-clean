"""
Simple model manager for Railway deployment - no singleton pattern needed.
Models are loaded once at startup and stored in a simple global dict.
"""

import os
import joblib
import json
from typing import Optional, Any, Dict
import threading
from pathlib import Path

# Use a forward reference for ModelBundle to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.models.loader import ModelBundle

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model registry with paths that will be used by the loader
MODEL_PATHS = {
    "ewclv1": "/app/models/disorder/ewclv1.pkl",
    "ewclv1-m": "/app/models/disorder/ewclv1-M.pkl",
    "ewclv1-p3": "/app/models/pdb/ewclv1p3.pkl",
    "ewclv1-c": "/app/models/clinvar/ewclv1-C.pkl",
}

# Configuration registry (non-model files)
CONFIG_REGISTRY = {
    "ewclv1-c-features": "/app/backend/config/ewclv1-c_features.json"
}

# Cache for loaded models
_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_models_loaded = False

def _initialize_models():
    """
    Load all models using the loader. This should only be called once.
    """
    global _models_loaded
    with _cache_lock:
        if _models_loaded:
            return

        # Import loader here to avoid circular dependency at startup
        from backend.models.loader import load_all

        print("[model_manager] Initializing all models from /app/models...")
        
        # The loader expects a base directory where it can find `models` and `meta`
        # In the Docker container, the app root is /app
        app_root = Path("/app")
        
        try:
            # The `load_all` function will scan for models in `/app/models`
            # and use environment variables as overrides.
            bundles = load_all(app_root)
            
            for name, bundle in bundles.items():
                _model_cache[name] = bundle.model
                print(f"[model_manager] Loaded model '{name}' successfully.")
                
                if bundle.feature_info:
                    _model_cache[f"{name}-features"] = bundle.feature_info
            
            _models_loaded = True
            print("[model_manager] All models initialized.")

        except Exception as e:
            print(f"[model_manager] CRITICAL: Failed to initialize models with loader: {e}")
            # This is a fatal error, but we don't raise it to allow health checks to run
            # and report the state of the application.
            _models_loaded = False


def get_model(model_name: str) -> Optional[Any]:
    """
    Get a model from the registry. Initializes all models on first call.
    """
    global _models_loaded
    if not _models_loaded:
        _initialize_models()

    with _cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        # Fallback for configuration files that are not part of bundles
        if model_name in CONFIG_REGISTRY:
            config_path = CONFIG_REGISTRY[model_name]
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    _model_cache[model_name] = config
                    print(f"[model_manager] Loaded config '{model_name}' from {config_path}")
                    return config
                else:
                    print(f"[model_manager] Config file not found: {config_path}")
            except Exception as e:
                print(f"[model_manager] Error loading config '{model_name}': {e}")

        print(f"[model_manager] Model or config '{model_name}' not found in cache.")
        return None

def get_available_models() -> Dict[str, str]:
    """Get a dictionary of available models and their status."""
    status = {}
    
    # Check models from the cache
    for model_key in MODEL_PATHS.keys():
        if model_key in _model_cache:
            status[model_key] = "available"
        else:
            # Check if the file exists, even if not loaded
            model_path = MODEL_PATHS[model_key]
            if os.path.exists(model_path):
                status[model_key] = "not_loaded"
            else:
                status[model_key] = "not_found"

    # Check configs
    for config_name, config_path in CONFIG_REGISTRY.items():
        if config_name in _model_cache:
            status[config_name] = "available"
        elif os.path.exists(config_path):
            status[config_name] = "not_loaded"
        else:
            status[config_name] = "not_found"
    
    return status

def clear_cache():
    """Clear the model cache."""
    global _models_loaded
    with _cache_lock:
        _model_cache.clear()
        _models_loaded = False
        print("[model_manager] Cache cleared and models marked for reload.")