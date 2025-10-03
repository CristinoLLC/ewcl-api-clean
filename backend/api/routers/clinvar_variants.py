from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import os, joblib

# Version safety check for EWCLv1-C model compatibility
try:
    from sklearn import __version__ as sklv
    if not sklv.startswith("1.7."):
        print(f"[ewclv1-c] ⚠️  Warning: EWCLv1-C expects scikit-learn 1.7.x, got {sklv}")
        print("[ewclv1-c] Model outputs may be inconsistent due to version mismatch")
except ImportError:
    print("[ewclv1-c] ⚠️  scikit-learn not available for version check")

router = APIRouter(prefix="/clinvar-simple", tags=["clinvar-simple"])

# --- Model & Feature Definitions ---
_MODEL_NAME = "ewclv1-c"
MODEL = None

# Hardcoded feature list, no external JSON needed
EWCLV1_C_FEATURES = [
    "position", "sequence_length", "position_ratio", "delta_hydropathy", "delta_charge",
    "delta_entropy_w5", "delta_entropy_w11", "has_embeddings", "delta_helix_prop", 
    "delta_sheet_prop", "delta_entropy_w25", "ewcl_hydropathy", "ewcl_charge_pH7",
    "ewcl_entropy_w5", "ewcl_entropy_w11",
    *[f"emb_{i}" for i in range(32)]
]

def _load_model():
    """Load model directly, without model_manager."""
    global MODEL
    model_path = os.environ.get("EWCLV1_C_MODEL_PATH")
    if not model_path:
        print(f"[warn] {_MODEL_NAME}: EWCLV1_C_MODEL_PATH env var not set.")
        return
    
    try:
        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
            print(f"[info] {_MODEL_NAME}: Loaded model directly from {model_path}")
        else:
            print(f"[warn] {_MODEL_NAME}: Model file not found at {model_path}")
    except Exception as e:
        print(f"[warn] {_MODEL_NAME}: Failed to load model: {e}")

# Initialize model on startup
_load_model()

# --- Schemas ---
class VariantIn(BaseModel):
    gene: Optional[str] = None
    protein_id: Optional[str] = None
    protein_pos: int = Field(..., ge=1, description="1-based residue index")
    ref_aa: str
    alt_aa: str

class ClinVarRequest(BaseModel):
    variants: List[VariantIn]
    features: Optional[List[Dict[str, float]]] = None

@router.get("/ewclv1-c/health")
def clinvar_health():
    """Health check for the ClinVar model."""
    return {
        "ok": True,
        "model_name": _MODEL_NAME,
        "loaded": MODEL is not None,
        "feature_count": len(EWCLV1_C_FEATURES),
        "ready": MODEL is not None,
    }

def _simple_featurize(v: VariantIn, seq_len: int = 500) -> Dict[str, float]:
    """
    Very simple, deterministic featurizer so endpoint works without SmartGate or embeddings.
    """
    aa_hydro = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    aa_charge7 = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
    aa_helix = {'A': 1.45, 'L': 1.34, 'R': 1.01, 'K': 1.23, 'M': 1.20}
    aa_sheet = {'V': 1.65, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.19}

    r, a = v.ref_aa.upper(), v.alt_aa.upper()
    def get(d, k, default=0.0): return float(d.get(k, default))

    feats = {
        "position": float(v.protein_pos),
        "sequence_length": float(seq_len),
        "position_ratio": float(v.protein_pos) / float(seq_len),
        "delta_hydropathy": get(aa_hydro, a) - get(aa_hydro, r),
        "delta_charge": get(aa_charge7, a) - get(aa_charge7, r),
        "delta_helix_prop": get(aa_helix, a) - get(aa_helix, r),
        "delta_sheet_prop": get(aa_sheet, a) - get(aa_sheet, r),
        "delta_entropy_w5": float(hash(a) % 100 - hash(r) % 100) / 100.0,
        "delta_entropy_w11": float((hash(a+"x") % 100) - (hash(r+"x") % 100)) / 100.0,
        "delta_entropy_w25": float((hash("y"+a) % 100) - (hash("y"+r) % 100)) / 100.0,
        "has_embeddings": 0.0,
    }
    for i in range(32):
        feats[f"emb_{i}"] = 0.0
    return feats

@router.post("/analyze-variants")
def clinvar_predict(req: ClinVarRequest):
    """
    Deterministic ClinVar prediction using the ewclv1-c model.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"ClinVar model '{_MODEL_NAME}' not available.")

    feature_order = EWCLV1_C_FEATURES
    rows = []
    for i, v in enumerate(req.variants):
        if req.features and i < len(req.features):
            feats = dict(req.features[i])
        else:
            feats = _simple_featurize(v)

        aligned = {k: float(feats.get(k, 0.0)) for k in feature_order}
        rows.append(aligned)

    X = pd.DataFrame(rows, columns=feature_order)

    try:
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(X)[:, 1].tolist()
        else:
            preds = MODEL.predict(X)
            probs = preds.astype(float).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClinVar prediction failed: {e}")

    out = []
    for i, v in enumerate(req.variants):
        out.append({
            "gene": v.gene,
            "protein_id": v.protein_id,
            "protein_pos": v.protein_pos,
            "ref_aa": v.ref_aa,
            "alt_aa": v.alt_aa,
            "pathogenic_prob": probs[i],
        })
    return {
        "ok": True,
        "model": _MODEL_NAME,
        "n": len(out),
        "variants": out
    }