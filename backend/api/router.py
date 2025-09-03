from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from backend.config import settings
from backend.models.loader import load_all
# Remove the broken imports - we'll use our individual routers instead
# from backend.features.ewclv1m import prepare_features_ewclv1m
# from backend.features.ewclv1 import prepare_features_ewclv1


router = APIRouter(prefix="/ewcl", tags=["ewcl"])

BUNDLE = settings.ewcl_bundle_dir
MODELS = load_all(BUNDLE)
if not MODELS:
    raise RuntimeError(f"No EWCL models found in {BUNDLE}")

def _require_list(mb) -> list[str]:
    fi = mb.feature_info
    if isinstance(fi, dict):
        feats = fi.get("all_features")
        if isinstance(feats, list):
            return feats
    if isinstance(fi, list):
        return fi
    return []

REQUIRES = {k: _require_list(v) for k, v in MODELS.items()}


class ResidueSample(BaseModel):
    residue_index: Optional[int] = None
    sequence_only: Optional[bool] = False
    features: Dict[str, Any]

class PredictionRequest(BaseModel):
    samples: List[ResidueSample]

# Legacy models for backward compatibility
class PredictEWCLv1M(BaseModel):
    sequence_only: bool = Field(False, description="Force PSSM defaults (if True)")
    features: Dict[str, Any]


class PredictEWCLv1(BaseModel):
    features: Dict[str, Any]


class PredictEWCLv1C(BaseModel):
    features: Dict[str, Any]


class PredictEWCLv1P3(BaseModel):
    features: Dict[str, Any]


# Simple compatibility router - the main functionality is in individual routers now
@router.get("/health")
def health():
    return {
        "status": "ok", 
        "message": "EWCL API with REAL features implementation",
        "available_endpoints": [
            "/ewcl/analyze-fasta/ewclv1-m",
            "/ewcl/analyze-pdb/ewclv1-p3", 
            "/clinvar/ewclv1-C/analyze-variants",
            "/models"
        ]
    }

# Legacy compatibility - redirect to new endpoints
@router.get("/predict/ewclv1m/schema")
def schema_ewclv1m():
    return {
        "model": "ewclv1m",
        "message": "Please use the new endpoint: /ewcl/analyze-fasta/ewclv1-m",
        "features": "255 REAL features including sequence properties, composition, flexibility, etc."
    }

@router.get("/predict/ewclv1p3/schema")
def schema_ewclv1p3():
    return {
        "model": "ewclv1p3", 
        "message": "Please use the new endpoint: /ewcl/analyze-pdb/ewclv1-p3",
        "features": "302 REAL features including hydropathy, charge, secondary structure, etc."
    }

@router.get("/predict/ewclv1c/schema")
def schema_ewclv1c():
    return {
        "model": "ewclv1c",
        "message": "Please use the new endpoint: /clinvar/ewclv1-C/analyze-variants", 
        "features": "47 REAL features including position, sequence_length, delta_hydropathy, etc."
    }


