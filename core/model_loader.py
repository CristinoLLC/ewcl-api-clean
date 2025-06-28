import pathlib
import joblib
import functools
import logging

_ROOT = pathlib.Path(__file__).resolve().parent.parent
POLY_MODEL_PATH = _ROOT / "models" / "poly_ridge_ewcl.pkl"
PHYSICS_MODEL_PATH = _ROOT / "models" / "physics_based_ewcl_model.pkl"
HALLUCINATION_MODEL_PATH = _ROOT / "models" / "hallucination_detector_model.pkl"

@functools.lru_cache(maxsize=1)
def get_poly_ridge():
    """Load the polynomial ridge regression model with caching"""
    try:
        model = joblib.load(POLY_MODEL_PATH)
        logging.info(f"✅ Loaded poly_ridge_ewcl model from {POLY_MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load poly_ridge_ewcl model: {e}")
        raise

@functools.lru_cache(maxsize=1)
def get_physics_model():
    """Load the physics-based EWCL model with caching"""
    try:
        model = joblib.load(PHYSICS_MODEL_PATH)
        logging.info(f"✅ Loaded physics_based_ewcl model from {PHYSICS_MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load physics_based_ewcl model: {e}")
        raise

@functools.lru_cache(maxsize=1)
def get_hallucination_model():
    """Load the hallucination detector model with caching"""
    try:
        model = joblib.load(HALLUCINATION_MODEL_PATH)
        logging.info(f"✅ Loaded hallucination_detector model from {HALLUCINATION_MODEL_PATH}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load hallucination_detector model: {e}")
        raise
