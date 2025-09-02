from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from backend.api.utils.clinvar_loader import get_clinvar_model
from backend.api.utils import clinvar_parser as cp

router = APIRouter(prefix="/clinvar", tags=["clinvar"])

class VariantRequest(BaseModel):
    protein_id: str
    sequence: Optional[str] = None
    seq_cache_key: Optional[str] = None
    variants: List[Dict]

class VariantResult(BaseModel):
    position: int
    ref: str
    alt: str
    pathogenic_prob: float
    class_: str = "benign"
    confidence: float

class VariantResponse(BaseModel):
    id: str
    model: str
    count: int
    variants: List[VariantResult]

@router.post("/analyze-variants", response_model=VariantResponse)
async def analyze_variants(request: VariantRequest):
    """
    Analyze protein variants for pathogenicity using EWCLv1-C model.
    Accepts either inline sequence or cached sequence key.
    """
    try:
        model = get_clinvar_model()
        
        # Get sequence (either inline or from cache - for now just use inline)
        if not request.sequence:
            raise HTTPException(400, "Sequence required (seq_cache_key not implemented yet)")
        
        sequence = request.sequence.upper().strip()
        if not sequence:
            raise HTTPException(400, "Empty sequence provided")
        
        if not request.variants:
            raise HTTPException(400, "No variants provided")
        
        # Build feature matrix using the real parser with hardcoded features
        X = cp.build_features(sequence, request.variants, cp.EWCLV1_C_FEATURES)
        
        # Predict pathogenicity
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X)
        
        # Calculate confidence (higher for more extreme predictions)
        conf = 1.0 - np.abs(probs - 0.5) * 2.0
        conf = np.clip(conf, 0.0, 1.0)
        
        # Build response
        results = []
        for i, variant in enumerate(request.variants):
            prob = float(probs[i])
            results.append(VariantResult(
                position=int(variant.get("position") or variant.get("pos")),
                ref=str(variant.get("ref") or variant.get("wt", "X")),
                alt=str(variant.get("alt") or variant.get("mut", "X")),
                pathogenic_prob=prob,
                class_="likely_pathogenic" if prob > 0.5 else "likely_benign",
                confidence=float(conf[i])
            ))
        
        return VariantResponse(
            id=request.protein_id,
            model="ewclv1-c",
            count=len(results),
            variants=results
        )
        
    except Exception as e:
        raise HTTPException(500, f"ClinVar analysis failed: {e}")

@router.get("/health")
def health():
    """Check ClinVar model health."""
    try:
        model = get_clinvar_model()
        return {
            "ok": True,
            "model": "ewclv1-c",
            "loaded": model is not None,
            "features": len(cp.EWCLV1_C_FEATURES),
            "ready": True
        }
    except Exception as e:
        return {
            "ok": False,
            "model": "ewclv1-c", 
            "loaded": False,
            "error": str(e),
            "ready": False
        }