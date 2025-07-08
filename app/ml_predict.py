"""
Load & run the ML models for refinement and hallucination detection
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Load models once at start-up ----------
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

try:
    REGRESSOR = joblib.load(MODEL_DIR / "ewcl_regressor_model.pkl")
    HALLUC = joblib.load(MODEL_DIR / "ewcl_residue_hallucination_detector_model.pkl")
    # Optional third model
    # REFINER = joblib.load(MODEL_DIR / "ewcl_residue_refiner_model.pkl")
except FileNotFoundError as e:
    print(f"Warning: Model file not found: {e}")
    REGRESSOR = None
    HALLUC = None

# ---------- Feature definitions ----------
REG_FEATURES = ["bfactor"]          # Minimal example - expand to your trained features
HAL_FEATURES = ["cl_diff"]          # Minimal example - expand to your trained features

def get_refined_cl(df: pd.DataFrame) -> pd.DataFrame:
    """Add cl_model column using regressor."""
    if REGRESSOR is None:
        # Fallback if model not loaded
        df["cl_model"] = df["cl"]  # Use raw CL as fallback
        df["cl_diff"] = 0.0
        return df
    
    # Use regressor to refine CL predictions
    df["cl_model"] = REGRESSOR.predict(df[REG_FEATURES])
    df["cl_diff"] = (df["cl_model"] - df["cl"]).abs()
    return df

def detect_hallucinations(df: pd.DataFrame) -> pd.DataFrame:
    """Add hallucination columns."""
    if HALLUC is None:
        # Fallback if model not loaded
        df["hallucination"] = False
        df["halluc_score"] = 0.0
        return df
    
    X = df[HAL_FEATURES]
    df["hallucination"] = HALLUC.predict(X)
    df["halluc_score"] = HALLUC.predict_proba(X)[:, 1]
    return df
