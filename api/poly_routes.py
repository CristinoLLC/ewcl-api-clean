from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np
import logging

from schemas import PolyPredictRequest, PolyPredictResponse
from core.model_loader import get_poly_ridge

router = APIRouter()

@router.post("/predict-poly", response_model=PolyPredictResponse)
def predict_poly(req: PolyPredictRequest):
    """
    Predict EWCL scores using the polynomial ridge regression model
    """
    try:
        # Convert features to numpy array
        X = np.array([
            [r.b_factor, r.plddt, r.hydropathy, r.charge]
            for r in req.residues
        ])
        
        # Get the model and make predictions
        model = get_poly_ridge()
        preds = model.predict(X).tolist()
        
        # Return predictions
        return PolyPredictResponse(cl_scores=preds)
    
    except Exception as e:
        logging.exception(f"❌ Error during polynomial prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
