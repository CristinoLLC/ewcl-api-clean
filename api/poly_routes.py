from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np
import logging

from schemas import PolyPredictRequest, PolyPredictResponse, AIScore, SourceInfo
from core.model_loader import get_poly_ridge
from core.utils import classify_risk_and_color

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
        
        scores_data = []
        for i, pred_score in enumerate(preds):
            risk_info = classify_risk_and_color(pred_score)
            scores_data.append(
                AIScore(
                    residue_id=i + 1,
                    cl=round(pred_score, 6),
                    risk_class=risk_info["risk_class"],
                    color_hex=risk_info["color_hex"]
                )
            )

        source = req.source if req.source else SourceInfo()

        # Return predictions with unified structure
        return PolyPredictResponse(
            model="poly_ridge_v1",
            mode="ai",
            interpretation="Higher = collapse",
            source=source,
            n_residues=len(scores_data),
            scores=scores_data
        )
    
    except Exception as e:
        logging.exception(f"❌ Error during polynomial prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
