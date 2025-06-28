from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel

from core.model_loader import get_hallucination_model

router = APIRouter()

class ResidueInput(BaseModel):
    protein: str
    residue_id: int
    cl: float = None
    plddt: float = None
    b_factor: float = None
    slope: float = None
    curvature: float = None
    ewcl_std_window: float = None
    abs_diff_cl_plddt: float = None
    abs_diff_cl_bfactor: float = None

@router.post("/predict-physics-ewcl")
async def predict_physics_ewcl(data: List[ResidueInput]):
    """
    Physics-based collapse likelihood using simple entropy-inspired formula
    """
    try:
        df = pd.DataFrame([d.dict() for d in data])
        
        # Simple physics-based calculation: normalize B-factor to [0,1]
        if "b_factor" not in df.columns or df["b_factor"].isna().all():
            raise HTTPException(status_code=400, detail="B-factor data required for physics-based EWCL")
        
        # Physics-based CL: higher B-factor = higher collapse likelihood
        df["cl"] = (df["b_factor"] / df["b_factor"].max()).clip(0, 1)
        
        return df[["protein", "residue_id", "cl"]].to_dict(orient="records")
        
    except Exception as e:
        logging.exception(f"❌ Error in physics EWCL prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Physics EWCL prediction failed: {str(e)}")

@router.post("/predict-hallucination")
async def predict_hallucination(data: List[ResidueInput]):
    """
    Predict hallucination scores using trained model
    """
    try:
        df = pd.DataFrame([d.dict() for d in data])
        
        # Required features for hallucination detection
        required_features = [
            "cl", "slope", "curvature",
            "ewcl_std_window", "abs_diff_cl_plddt", "abs_diff_cl_bfactor"
        ]
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}"
            )
        
        # Drop rows with missing feature values
        df = df.dropna(subset=required_features)
        
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid rows after removing missing feature values"
            )
        
        # Load hallucination model and predict
        hallucination_model = get_hallucination_model()
        X = df[required_features]
        df["hallucination_score"] = hallucination_model.predict_proba(X)[:, 1]
        
        return df[["protein", "residue_id", "hallucination_score"]].to_dict(orient="records")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"❌ Error in hallucination prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Hallucination prediction failed: {str(e)}")
