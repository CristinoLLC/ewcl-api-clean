import pathlib
import joblib
import functools
import logging

_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = _ROOT / "models" / "poly_ridge_ewcl.pkl"

@functools.lru_cache(maxsize=1)   # ensures it loads once
def get_poly_ridge():
    """Load the polynomial ridge regression model with caching"""
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"✅ Loaded poly_ridge_ewcl model from {MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load poly_ridge_ewcl model: {e}")
        raise
