"""
ML helpers for regressor and hallucination detection
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Load models at import time
try:
    REGRESSOR = joblib.load(MODEL_DIR / "ewcl_regressor_model.pkl")
    HIGH_MODEL = joblib.load(MODEL_DIR / "ewcl_residue_local_high_model.pkl")
    HIGH_SCALER = joblib.load(MODEL_DIR / "ewcl_residue_local_high_scaler.pkl")
    HALLUC_MODEL = joblib.load(MODEL_DIR / "hallucination_detector_model.pkl")
except FileNotFoundError as e:
    print(f"Warning: Model file not found: {e}")
    REGRESSOR = HIGH_MODEL = HIGH_SCALER = HALLUC_MODEL = None

# Clean feature lists - remove any legacy features
REG_FEATS = ["bfactor", "bfactor_norm", "hydro_entropy",
             "charge_entropy", "bfactor_curv", "bfactor_curv_entropy",
             "bfactor_curv_flips"]

HAL_FEATS = ["cl_diff", "cl_diff_slope",
             "cl_diff_curv", "cl_diff_flips",
             "bfactor", "bfactor_norm",
             "hydro_entropy", "charge_entropy",
             "bfactor_curv", "bfactor_curv_entropy", "bfactor_curv_flips"]

def add_main_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Adds column cl_pred from REGRESSOR - no legacy logic"""
    if REGRESSOR is None:
        df["cl_pred"] = df["cl"]  # Fallback
        return df
    
    # Clean prediction using only specified features
    X = df[REG_FEATS]
    df["cl_pred"] = REGRESSOR.predict(X)
    return df

def add_high_refinement(df: pd.DataFrame) -> pd.DataFrame:
    """Run the high-correlation refiner - no legacy transforms"""
    if HIGH_MODEL is None or HIGH_SCALER is None:
        df["cl_refined"] = df["cl"]  # Fallback
        return df
    
    X = df[REG_FEATS]
    X_scaled = HIGH_SCALER.transform(X)
    df["cl_refined"] = HIGH_MODEL.predict(X_scaled)
    return df

def add_hallucination(df: pd.DataFrame) -> pd.DataFrame:
    """Clean hallucination detection - no placeholder logic"""
    if HALLUC_MODEL is None:
        df["hallucination"] = False
        df["halluc_score"] = 0.0
        return df
    
    # Calculate difference features
    df["cl_diff"] = (df["cl_pred"] - df["cl"]).abs()
    df["cl_diff_slope"] = np.gradient(df["cl_diff"])
    df["cl_diff_curv"] = np.gradient(df["cl_diff_slope"])
    df["cl_diff_flips"] = pd.Series(
        np.sign(df["cl_diff_slope"])
    ).diff().abs().fillna(0)

    X = df[HAL_FEATS]
    df["hallucination"] = HALLUC_MODEL.predict(X)
    df["halluc_score"] = HALLUC_MODEL.predict_proba(X)[:, 1]
    return df
