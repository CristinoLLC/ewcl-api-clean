"""
Simple model store for EWCL API routers.
Provides a clean interface for loading and accessing trained models.
"""

from __future__ import annotations
import os, joblib, threading
from pathlib import Path

_lock = threading.Lock()
_cache = {}  # name -> model

ENV_MAP = {
    "ewclv1":   "EWCLV1_MODEL_PATH",
    "ewclv1-m": "EWCLV1_M_MODEL_PATH",
    "ewclv1-p3":"EWCLV1_P3_MODEL_PATH",
    "ewclv1-c": "EWCLV1_C_MODEL_PATH",
}

def get_model(name: str):
    """Load once per process from the ENV path. No magic, no indirection."""
    name = name.lower()
    env = ENV_MAP.get(name)
    if not env:
        raise RuntimeError(f"[model_store] unknown model key: {name}")
    path = os.environ.get(env)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"[model_store] {name} path not found: {path}")

    with _lock:
        if name in _cache:
            return _cache[name]
        model = joblib.load(path)
        _cache[name] = model
        return model