from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

from backend.config import settings
from backend.models.loader import load_all
from backend.features.ewclv1m import prepare_features_ewclv1m
from backend.features.ewclv1 import prepare_features_ewclv1


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


class PredictEWCLv1M(BaseModel):
    sequence_only: bool = Field(False, description="Force PSSM defaults (if True)")
    features: Dict[str, Any]


class PredictEWCLv1(BaseModel):
    features: Dict[str, Any]


class PredictEWCLv1C(BaseModel):
    features: Dict[str, Any]


class PredictEWCLv1P3(BaseModel):
    features: Dict[str, Any]


@router.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODELS.keys()), "bundle_dir": str(BUNDLE)}


@router.get("/predict/ewclv1m/schema")
def schema_ewclv1m():
    if "ewclv1m" not in MODELS:
        raise HTTPException(404, "ewclv1m not loaded")
    fi = MODELS["ewclv1m"].feature_info
    return {
        "model": "ewclv1m",
        "required_features": fi.get("all_features", []),
        "works_without_pssm": fi.get("works_without_pssm", True),
        "notes": "Set sequence_only=true to use PSSM defaults."
    }


@router.get("/predict/ewclv1/schema")
def schema_ewclv1():
    if "ewclv1" not in MODELS:
        raise HTTPException(404, "ewclv1 not loaded")
    fi = MODELS["ewclv1"].feature_info
    return {
        "model": "ewclv1",
        "required_features": fi.get("all_features", []),
        "notes": "Classic model; no PSSM defaults automatically."
    }


@router.get("/predict/ewclv1p3/schema")
def schema_ewclv1p3():
    if "ewclv1p3" not in MODELS:
        raise HTTPException(404, "ewclv1p3 not loaded")
    fi = MODELS["ewclv1p3"].feature_info
    return {
        "model": "ewclv1p3",
        "required_features": fi.get("all_features", []),
        "notes": "Main PDB model; provide named features matching training."
    }


@router.get("/predict/ewclv1c/schema")
def schema_ewclv1c():
    if "ewclv1c" not in MODELS:
        raise HTTPException(404, "ewclv1c not loaded")
    fi = MODELS["ewclv1c"].feature_info
    return {
        "model": "ewclv1c",
        "required_features": fi.get("all_features", []),
        "notes": "ClinVar gate model (no PSSM defaults)."
    }


@router.post("/predict/ewclv1m")
def predict_ewclv1m(req: PredictEWCLv1M):
    if "ewclv1m" not in MODELS:
        raise HTTPException(404, "ewclv1m not loaded")
    feats_required = REQUIRES["ewclv1m"]
    X = prepare_features_ewclv1m(req.model_dump(), feats_required)
    try:
        prob = float(MODELS["ewclv1m"].predict_proba(X).iloc[0])
    except Exception as e:
        raise HTTPException(422, f"Feature mismatch: {e}")
    return {"model": "ewclv1m", "prob": prob}


@router.post("/predict/ewclv1")
def predict_ewclv1(req: PredictEWCLv1):
    if "ewclv1" not in MODELS:
        raise HTTPException(404, "ewclv1 not loaded")
    feats_required = REQUIRES["ewclv1"]
    X = prepare_features_ewclv1(req.model_dump(), feats_required)
    try:
        prob = float(MODELS["ewclv1"].predict_proba(X).iloc[0])
    except Exception as e:
        raise HTTPException(422, f"Feature mismatch: {e}")
    return {"model": "ewclv1", "prob": prob}


@router.post("/predict/ewclv1c")
def predict_ewclv1c(req: PredictEWCLv1C):
    if "ewclv1c" not in MODELS:
        raise HTTPException(404, "ewclv1c not loaded")
    # ClinVar gate may have its own feature set; accept as-is and rely on loader ordering if present
    X = prepare_features_ewclv1(req.model_dump(), MODELS["ewclv1c"].feature_info.get("all_features", []))
    try:
        prob = float(MODELS["ewclv1c"].predict_proba(X).iloc[0])
    except Exception as e:
        raise HTTPException(422, f"Feature mismatch: {e}")
    return {"model": "ewclv1c", "prob": prob}


@router.post("/predict/ewclv1p3")
def predict_ewclv1p3(req: PredictEWCLv1P3):
    if "ewclv1p3" not in MODELS:
        raise HTTPException(404, "ewclv1p3 not loaded")
    feats_required = REQUIRES.get("ewclv1p3", MODELS["ewclv1p3"].feature_info.get("all_features", []))
    X = prepare_features_ewclv1(req.model_dump(), feats_required)
    try:
        prob = float(MODELS["ewclv1p3"].predict_proba(X).iloc[0])
    except Exception as e:
        raise HTTPException(422, f"Feature mismatch: {e}")
    return {"model": "ewclv1p3", "prob": prob}


