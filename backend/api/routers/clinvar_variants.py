from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import os

# Use new model manager instead of singleton
from backend.models.model_manager import get_model

router = APIRouter(prefix="/clinvar", tags=["clinvar"])

# --- Fallback feature order (if model lacks feature_names_in_) ---
FALLBACK_FEATURE_ORDER = [
    # sequence
    "position", "sequence_length", "position_ratio",
    # delta features
    "delta_hydropathy", "delta_charge", "delta_helix_prop", "delta_sheet_prop",
    "delta_entropy_w5", "delta_entropy_w11", "delta_entropy_w25",
    # embeddings
    "has_embeddings",
    *[f"emb_{i}" for i in range(32)],
    # optional extras if your model was trained with them (uncomment if needed)
    # "is_conserved", "structural_score", "disorder_score", "coverage_score",
]

class VariantIn(BaseModel):
    gene: Optional[str] = None
    protein_id: Optional[str] = None
    protein_pos: int = Field(..., ge=1, description="1-based residue index")
    ref_aa: str
    alt_aa: str

class ClinVarRequest(BaseModel):
    variants: List[VariantIn]
    # Optional: let clients pass precomputed features directly
    features: Optional[List[Dict[str, float]]] = None

@router.get("/ewclv1-c/health")
def clinvar_health():
    """
    Load the ClinVar model and report feature count if available.
    No SmartGate, no external features JSON required.
    """
    try:
        # Use new model manager
        clf = get_model("ewclv1-c")
        if clf is None:
            return {"ok": False, "model_name": "ewclv1-c", "loaded": False, "error": "Model not loaded"}
        
        feats = getattr(clf, "feature_names_in_", None)
        return {
            "ok": True,
            "model_name": "ewclv1-c",
            "loaded": True,
            "feature_count": int(len(feats)) if feats is not None else None,
            "uses_feature_names_in": bool(feats is not None),
        }
    except Exception as e:
        return {"ok": False, "model_name": "ewclv1-c", "loaded": False, "error": str(e)}

def _simple_featurize(v: VariantIn, seq_len: int = 500) -> Dict[str, float]:
    """
    Very simple, deterministic featurizer so endpoint works without SmartGate or embeddings.
    Replace with your true featurization later.
    """
    aa_hydro = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    aa_charge7 = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
    aa_helix = {'A': 1.45, 'L': 1.34, 'R': 1.01, 'K': 1.23, 'M': 1.20}  # toy
    aa_sheet = {'V': 1.65, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.19}  # toy

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
        # deterministic pseudo-entropy by AA identity (toy; replace with real)
        "delta_entropy_w5": float(hash(a) % 100 - hash(r) % 100) / 100.0,
        "delta_entropy_w11": float((hash(a+"x") % 100) - (hash(r+"x") % 100)) / 100.0,
        "delta_entropy_w25": float((hash("y"+a) % 100) - (hash("y"+r) % 100)) / 100.0,
        "has_embeddings": 0.0,
    }
    # pad emb_0..emb_31 as zeros for now
    for i in range(32):
        feats[f"emb_{i}"] = 0.0
    return feats

@router.post("/analyze-variants")
def clinvar_predict(req: ClinVarRequest):
    """
    Deterministic ClinVar prediction using the ewclv1-c model.
    Accepts `variants` and optional explicit `features`.
    """
    try:
        # Use new model manager
        clf = get_model("ewclv1-c")
        if clf is None:
            raise Exception("Model not loaded in model manager")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClinVar model not available: {e}")

    feature_names = getattr(clf, "feature_names_in_", None)
    if feature_names is not None:
        feature_order = list(map(str, feature_names))  # ensure list[str]
    else:
        feature_order = FALLBACK_FEATURE_ORDER

    rows = []
    for i, v in enumerate(req.variants):
        if req.features and i < len(req.features):
            feats = dict(req.features[i])
        else:
            feats = _simple_featurize(v)

        # align to model's feature order
        aligned = {k: float(feats.get(k, 0.0)) for k in feature_order}
        rows.append(aligned)

    X = pd.DataFrame(rows, columns=feature_order)

    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[:, 1].tolist()
        else:
            preds = clf.predict(X)
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
        "model": "ewclv1-c",
        "n": len(out),
        "variants": out
    }