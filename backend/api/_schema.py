from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
import math

def _safe_num(x):
    if x is None: 
        return None
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
    except Exception:
        return None
    return float(x)

def compute_confidence(cl: float) -> float:
    # 0.5 → 0.0 (uncertain), 0/1 → 1.0 (confident)
    if cl is None:
        return 0.0
    cl = max(0.0, min(1.0, float(cl)))
    return 1.0 - abs(cl - 0.5) * 2.0

class ResidueOut(BaseModel):
    residue_index: int = Field(..., ge=1)
    aa: str
    cl: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    # optional "most important" echoes if available
    hydropathy: Optional[float] = None
    charge: Optional[float] = None
    helix_prop: Optional[float] = None
    sheet_prop: Optional[float] = None

class SequenceResponse(BaseModel):
    id: str
    model: Literal["ewclv1", "ewclv1-m", "ewclv1-p3"]
    length: int = Field(..., ge=0)
    residues: List[ResidueOut]

def build_residue(
    idx: int,
    aa: str,
    cl: float,
    feats: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "residue_index": idx,
        "aa": aa,
        "cl": _safe_num(cl) or 0.0,
        "confidence": compute_confidence(_safe_num(cl) or 0.0),
        "hydropathy": _safe_num(feats.get("hydropathy")),
        "charge": _safe_num(feats.get("charge_pH7") or feats.get("charge")),
        "helix_prop": _safe_num(feats.get("helix_prop")),
        "sheet_prop": _safe_num(feats.get("sheet_prop")),
    }