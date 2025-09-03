from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from functools import lru_cache
import os, json, numpy as np, joblib, math
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/clinvar", tags=["clinvar"])

class VariantRequest(BaseModel):
    protein_id: str
    position: int = Field(..., ge=1)
    from_aa: str
    to_aa: str
    variant_type: str  # substitution | deletion | insertion | delins
    sequence_length: Optional[int] = None
    features: Dict[str, float] = {}

class VariantsPayload(BaseModel):
    variants: List[VariantRequest]
    threshold: float = 0.5

@lru_cache(maxsize=1)
def _load_model_and_features():
    """Load single ClinVar model and ordered feature list (cached)."""
    model_path = os.getenv("EWCLV1_C_MODEL_PATH")
    feat_path = os.getenv("EWCLV1_C_FEATURES_PATH")

    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"ClinVar model not found at {model_path}")
    if not feat_path or not os.path.exists(feat_path):
        # Fallback: accept lowercase variant if provided file name differs
        alt = feat_path.replace("EWCLv1-C_features.json", "ewclv1-c_features.json") if feat_path else None
        if alt and os.path.exists(alt):
            feat_path = alt
        else:
            raise RuntimeError(f"ClinVar features JSON not found at {feat_path}")

    model = joblib.load(model_path)
    with open(feat_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "features" in data:
        feats = [str(x) for x in data["features"]]
    elif isinstance(data, list):
        feats = [str(x) for x in data]
    else:
        raise RuntimeError("Unrecognized feature list JSON format")
    return model, feats

def _align(feat_dict: Dict[str, float], order: List[str]):
    x = np.zeros(len(order), dtype=np.float32)
    missing = []
    for i, name in enumerate(order):
        if name in feat_dict:
            try:
                x[i] = float(feat_dict[name])
            except Exception:
                x[i] = 0.0
        else:
            missing.append(name)
    coverage = 1.0 - (len(missing) / len(order) if order else 0.0)
    return x, missing, coverage

def _prob_to_confidence(p: float) -> float:
    p = min(max(p, 1e-8), 1 - 1e-8)
    ent = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))  # entropy in bits (0..1)
    return float(1.0 - ent)

@router.get("/health")
def health():
    try:
        model, feats = _load_model_and_features()
        return {"ok": True, "model": "ewclv1-c", "loaded": True, "n_features": len(feats)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/analyze-variants")
def analyze_variants(payload: VariantsPayload):
    model, order = _load_model_and_features()
    if not payload.variants:
        return {"model": "ewclv1-c", "n_variants": 0, "variants": []}

    X_list, miss_list, cov_list = [], [], []
    for v in payload.variants:
        x, missing, cov = _align(v.features or {}, order)
        X_list.append(x); miss_list.append(missing); cov_list.append(cov)

    X = np.vstack(X_list)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            raw = model.predict(X)
            probs = np.clip(raw, 0.0, 1.0)
    except Exception as e:
        raise HTTPException(500, f"ClinVar model inference failed: {e}")

    out = []
    for v, p, missing, cov in zip(payload.variants, probs, miss_list, cov_list):
        label = "pathogenic" if p >= payload.threshold else "benign"
        out.append({
            "protein_id": v.protein_id,
            "position": v.position,
            "from_aa": v.from_aa,
            "to_aa": v.to_aa,
            "variant_type": v.variant_type,
            "sequence_length": v.sequence_length,
            "probability": float(p),
            "class": label,
            "confidence": _prob_to_confidence(float(p)),
            "coverage": cov,
            "missing_features": missing
        })

    return {"model": "ewclv1-c", "n_variants": len(out), "variants": out}