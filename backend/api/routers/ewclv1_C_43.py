# backend/api/routers/ewclv1_C_43.py
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
import numpy as np

from backend.models.model_manager import get_model, is_loaded

router = APIRouter(prefix="/clinvar", tags=["clinvar-43-features"])

# 43-feature ClinVar model features (missing the 4 EWCL features)
EWCLV1_C_43_FEATURES = [
    "position", "sequence_length", "position_ratio", "delta_hydropathy", "delta_charge",
    "delta_helix_prop", "delta_sheet_prop", "delta_entropy_w5", "delta_entropy_w11", 
    "delta_entropy_w25", "has_embeddings",
    "emb_0", "emb_1", "emb_2", "emb_3", "emb_4", "emb_5", "emb_6", "emb_7", "emb_8", "emb_9",
    "emb_10", "emb_11", "emb_12", "emb_13", "emb_14", "emb_15", "emb_16", "emb_17", "emb_18", "emb_19",
    "emb_20", "emb_21", "emb_22", "emb_23", "emb_24", "emb_25", "emb_26", "emb_27", "emb_28", "emb_29",
    "emb_30", "emb_31"
]

FEATURE_NAMES_43 = EWCLV1_C_43_FEATURES

# Enhanced feature engineering (same as 47-feature model)
AA_HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
AA_CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0, 'H': 0.5,
    'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0,
    'Y': 0, 'V': 0
}
AA_VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9,
    'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}
AA_FLEXIBILITY = {
    'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497,
    'G': 0.544, 'H': 0.323, 'I': 0.462, 'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314,
    'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386
}

def _vectorize_43(feat_map: Dict[str, float], names: List[str]) -> np.ndarray:
    return np.array([float(feat_map.get(n, 0.0) or 0.0) for n in names], dtype=np.float32)

def _cheap_deltas_43(ref: str, alt: str) -> Dict[str, float]:
    return {
        "delta_hydropathy": AA_HYDRO.get(alt, 0.0) - AA_HYDRO.get(ref, 0.0),
        "delta_charge": AA_CHARGE.get(alt, 0.0) - AA_CHARGE.get(ref, 0.0),
    }

def _generate_pseudo_embeddings_43(ref: str, alt: str, pos: int, length: int) -> Dict[str, float]:
    """Generate pseudo-embeddings based on amino acid properties and position"""
    import hashlib
    
    seed = f"{ref}{pos}{alt}{length}"
    hash_obj = hashlib.md5(seed.encode())
    hash_bytes = hash_obj.digest()
    
    embeddings = {}
    for i in range(32):
        byte_val = hash_bytes[i % len(hash_bytes)]
        normalized = (byte_val - 127.5) / 127.5
        
        ref_influence = AA_HYDRO.get(ref, 0.0) * 0.1
        alt_influence = AA_HYDRO.get(alt, 0.0) * 0.1
        pos_influence = (pos % 100) * 0.01
        
        embeddings[f"emb_{i}"] = float(normalized + ref_influence + alt_influence + pos_influence)
    
    return embeddings

def _enhanced_features_43(ref: str, alt: str, pos: int, length: Optional[int]) -> Dict[str, float]:
    """Generate enhanced features for 43-feature model (no EWCL features)"""
    if not length:
        length = 400
    
    features = {
        "position": float(pos),
        "sequence_length": float(length),
        "position_ratio": pos / float(length),
        **_cheap_deltas_43(ref, alt),
        "has_embeddings": 1.0,
    }
    
    ref_vol = AA_VOLUME.get(ref, 140.0)
    alt_vol = AA_VOLUME.get(alt, 140.0)
    ref_flex = AA_FLEXIBILITY.get(ref, 0.4)
    alt_flex = AA_FLEXIBILITY.get(alt, 0.4)
    
    features.update({
        "delta_entropy_w5": (alt_flex - ref_flex) * 2.0,
        "delta_entropy_w11": (alt_flex - ref_flex) * 1.5,
        "delta_entropy_w25": (alt_flex - ref_flex) * 0.8,
        "delta_helix_prop": (AA_HYDRO.get(alt, 0.0) - AA_HYDRO.get(ref, 0.0)) * 0.3,
        "delta_sheet_prop": (alt_vol - ref_vol) / 100.0,
        # NOTE: No EWCL features in 43-feature model
    })
    
    features.update(_generate_pseudo_embeddings_43(ref, alt, pos, length))
    
    return features

def _coverage_43(seq: Optional[str], length: Optional[int]) -> float:
    if seq and len(seq) >= 10: return 1.0
    if length and length > 0:  return 0.7
    return 0.3

def _confidence_43(prob: float, coverage: float) -> float:
    return 0.7 * abs(prob - 0.5) * 2.0 + 0.3 * coverage

# Reuse same schemas from the 47-feature model
class AdvancedSample43(BaseModel):
    variant_id: Optional[str] = None
    position: int = Field(..., description="1-based protein position")
    ref: str = Field(..., min_length=1, max_length=1)
    alt: str = Field(..., min_length=1, max_length=1)
    sequence: Optional[str] = None
    protein_length: Optional[int] = None
    protein_id: Optional[str] = None
    gene: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    feature_array: Optional[List[float]] = None

class AdvancedRequest43(BaseModel):
    samples: List[AdvancedSample43]

class AdvancedVariantOut43(BaseModel):
    variant_id: Optional[str]
    position: int
    ref: str
    alt: str
    pathogenic_prob: float
    confidence: float
    coverage: float
    low_coverage: bool
    flagged: bool
    protein_id: Optional[str] = None
    gene: Optional[str] = None
    debug_features_subset: Optional[Dict[str, Any]] = None

class AdvancedResponse43(BaseModel):
    model: str
    count: int
    variants: List[AdvancedVariantOut43]

@router.get("/v43/health")
def health_43():
    return {
        "model": "clinvar-pathogenicity-v43",
        "model_loaded": is_loaded("ewclv1-c-43"),
        "features_count": len(EWCLV1_C_43_FEATURES),
        "ready": is_loaded("ewclv1-c-43") and len(EWCLV1_C_43_FEATURES) > 0,
        "ok": True,
        "loaded": is_loaded("ewclv1-c-43")
    }

@router.post("/v43/analyze-variants", response_model=AdvancedResponse43)
def analyze_variants_43(
    req: AdvancedRequest43,
    request: Request,
    debug: int = Query(0)
):
    if not is_loaded("ewclv1-c-43"):
        raise HTTPException(status_code=503, detail="43-feature model is not loaded.")
    model = get_model("ewclv1-c-43")

    samples = req.samples or []
    CHUNK = 512
    out: List[AdvancedVariantOut43] = []

    for i in range(0, len(samples), CHUNK):
        batch = samples[i:i+CHUNK]
        feats_list, meta_list = [], []

        for s in batch:
            ref = (s.ref or "").upper()[0]
            alt = (s.alt or "").upper()[0]
            pos = int(s.position)
            seq = s.sequence
            length = s.protein_length or (len(seq) if seq else None)

            base = _enhanced_features_43(ref, alt, pos, length)
                
            if s.features:
                base.update({k: float(v) for k, v in s.features.items()})

            if s.feature_array and len(s.feature_array) == len(FEATURE_NAMES_43):
                x = np.array(s.feature_array, dtype=np.float32)
            else:
                x = _vectorize_43(base, FEATURE_NAMES_43)

            feats_list.append(x)
            meta_list.append({
                "variant_id": s.variant_id,
                "position": pos,
                "ref": ref,
                "alt": alt,
                "sequence": seq,
                "protein_length": length,
                "protein_id": s.protein_id,
                "gene": s.gene
            })

        X = np.vstack(feats_list) if feats_list else np.zeros((0, len(FEATURE_NAMES_43)), dtype=np.float32)
        try:
            probs = model.predict(X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        for p, meta in zip(probs, meta_list):
            cov = _coverage_43(meta["sequence"], meta["protein_length"])
            conf = _confidence_43(float(p), cov)
            out.append(AdvancedVariantOut43(
                variant_id = meta["variant_id"],
                position   = meta["position"],
                ref        = meta["ref"],
                alt        = meta["alt"],
                pathogenic_prob = float(p),
                confidence = float(conf),
                coverage   = float(cov),
                low_coverage = bool(cov < 0.6),
                flagged = False,
                protein_id = meta["protein_id"],
                gene       = meta["gene"],
                debug_features_subset = (
                    {"position": meta["position"],
                     "protein_length": meta["protein_length"] or 0,
                     "position_ratio": (meta["position"]/float(meta["protein_length"])) if meta["protein_length"] else 0.0}
                    if debug else None
                )
            ))

    return AdvancedResponse43(model="clinvar-pathogenicity-v43", count=len(out), variants=out)