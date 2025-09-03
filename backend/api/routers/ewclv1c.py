from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from backend.models.model_manager import get_model  # Use new model manager

router = APIRouter()

_MODEL_NAME = "ewclv1c"
MODEL = None
FEATURES_CONFIG = None

def _load_model():
    global MODEL, FEATURES_CONFIG
    try:
        MODEL = get_model("ewclv1-c")  # Note: uses hyphen in model manager
        FEATURES_CONFIG = get_model("ewclv1-c-features")
        if MODEL and FEATURES_CONFIG:
            print(f"[info] Loaded EWCLv1-C model and features config")
        else:
            print(f"[warn] EWCLv1-C model or features not found in model manager")
    except Exception as e:
        print(f"[warn] Failed to load EWCLv1-C model: {e}")

# Initialize model
_load_model()

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not MODEL or not FEATURES_CONFIG:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame(request.data)
        # Ensure the dataframe has the required features
        missing_features = [feature for feature in FEATURES_CONFIG if feature not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

        # Predict using the model
        predictions = MODEL.predict(df[FEATURES_CONFIG])
        return PredictionResponse(predictions=predictions.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))