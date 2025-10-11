"""
Disorder Router - Privacy-friendly alias for EWCLv1-M model
Provides /disorder/ endpoints that map to the ewclv1-M functionality
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import os

router = APIRouter(prefix="/disorder", tags=["disorder"])
log = logging.getLogger("disorder")

@router.post("/analyze-fasta-m")
async def analyze_fasta_m(file: UploadFile = File(...)):
    """
    Privacy-friendly disorder prediction endpoint.
    Maps to ewclv1-M model functionality.
    """
    try:
        # Import the ewclv1-M router function
        from backend.api.routers.ewclv1_M import analyze_fasta_ewclv1m
        
        # Call the actual ewclv1-M endpoint
        result = await analyze_fasta_ewclv1m(file)
        
        # Return the result with disorder-friendly naming
        return {
            "id": result.get("id"),
            "model": "disorder-predictor",  # Privacy-friendly model name
            "length": result.get("length"),
            "residues": result.get("residues"),
            "diagnostics": result.get("diagnostics")
        }
        
    except Exception as e:
        log.exception("[disorder] analysis failed")
        raise HTTPException(status_code=500, detail=f"Disorder analysis failed: {e}")

@router.get("/health")
def disorder_health():
    """
    Health check endpoint for disorder prediction service.
    Maps to ewclv1-M health check.
    """
    try:
        # Import the ewclv1-M health check function
        from backend.api.routers.ewclv1_M import health_check
        
        # Call the actual ewclv1-M health check
        health_result = health_check()
        
        # Return with disorder-friendly naming
        return {
            "service": "disorder-predictor",
            "status": health_result.get("status"),
            "model_path": health_result.get("model_path"),
            "model_loaded": health_result.get("model_loaded"),
            "has_feature_names": health_result.get("has_feature_names"),
            "feature_count": health_result.get("feature_count"),
            "loader_used": health_result.get("loader_used"),
            "hardcoded_features": health_result.get("hardcoded_features"),
            "sklearn_warnings_suppressed": health_result.get("sklearn_warnings_suppressed"),
            "error": health_result.get("error") if health_result.get("status") == "error" else None
        }
        
    except Exception as e:
        log.exception("[disorder] health check failed")
        return {
            "service": "disorder-predictor",
            "status": "error",
            "error": str(e)
        }
