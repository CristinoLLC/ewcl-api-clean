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
    "ewclv1-c": "/app/models/clinvar/C_Full_model.pkl",  # Full 47-feature model
    "ewclv1-c-43": "/app/models/clinvar/C_43_model.pkl",  # 43-feature model
}

# Configuration registry (non-model files)
CONFIG_REGISTRY = {
    # Removed ewclv1-c-features - using hardcoded features in router now
}

# Cache for loaded models
_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_models_loaded = False

def _initialize_models():
    """
    Load all models using the simple loader. This should only be called once.
    """
    global _models_loaded
    with _cache_lock:
        if _models_loaded:
            return

        # Import the correct loader function
        from backend.models.loader import load_model_forgiving

        print("[model_manager] Initializing all models...")
        
        # Load each model individually using environment variables
        model_env_mapping = {
            "ewclv1": "EWCLV1_MODEL_PATH",
            "ewclv1-m": "EWCLV1_M_MODEL_PATH", 
            "ewclv1-p3": "EWCLV1_P3_MODEL_PATH",
            "ewclv1-c": "EWCLV1_C_MODEL_PATH",
            "ewclv1-c-43": "EWCLV1_C_43_MODEL_PATH"  # Add 43-feature ClinVar model
        }
        
        # Override with hardcoded paths to ensure correct models are loaded
        hardcoded_paths = {
            "ewclv1-c": "/app/models/clinvar/C_Full_model.pkl",  # Force use of C_Full_model.pkl
            "ewclv1-c-43": "/app/models/clinvar/C_43_model.pkl"   # Force use of C_43_model.pkl
        }
        
        for model_name, env_var in model_env_mapping.items():
            # Use hardcoded path if available, otherwise use environment variable
            if model_name in hardcoded_paths:
                model_path = hardcoded_paths[model_name]
                print(f"[model_manager] Using hardcoded path for {model_name}: {model_path}")
            else:
                model_path = os.environ.get(env_var)
            
            print(f"[model_manager] Attempting to load {model_name} from {model_path}")
            
            if model_path and os.path.exists(model_path):
                try:
                    model = load_model_forgiving(model_path)
                    _model_cache[model_name] = model
                    print(f"[model_manager] ✅ Loaded model '{model_name}' from {model_path}")
                except Exception as e:
                    print(f"[model_manager] ❌ Failed to load {model_name} from {model_path}: {e}")
                    import traceback
                    print(f"[model_manager] Traceback: {traceback.format_exc()}")
            else:
                print(f"[model_manager] ❌ Model path not found for {model_name}: {model_path}")
        
        _models_loaded = True
        print("[model_manager] Model initialization complete.")

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

def load_all_models():
    """Simple function to satisfy import - individual routers load their own models."""
    print("[model_manager] Individual routers will load models as needed.")
    pass

def get_loaded_models():
    """Return empty dict since individual routers handle loading."""
    return {}

def is_loaded(model_name: str) -> bool:
    """Check if a model is actually loaded in the cache."""
    if not _models_loaded:
        _initialize_models()
    
    with _cache_lock:
        return model_name in _model_cache and _model_cache[model_name] is not None