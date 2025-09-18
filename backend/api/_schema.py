"""
Pydantic schemas for EWCL-H API
==============================

Request and response models for hallucination detection endpoints.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, conlist
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

class ResidueScore(BaseModel):
    """Per-residue scores and flags with comprehensive data for frontend."""
    pos: int
    aa: Optional[str] = None  # Add amino acid for frontend display
    ewcl: float
    plddt: Optional[float] = None
    bfactor: Optional[float] = None  # Include B-factor for X-ray structures
    confidence: Optional[float] = None  # Unified confidence score
    confidence_type: Optional[str] = None  # "plddt", "bfactor", or "none"
    H: Optional[float] = None
    is_high_H: Optional[bool] = None
    is_disagree: Optional[bool] = None
    # Additional features for correlation analysis
    hydropathy: Optional[float] = None
    charge_pH7: Optional[float] = None
    curvature: Optional[float] = None

class HallucinationRequest(BaseModel):
    """Request for hallucination detection."""
    uniprot: Optional[str] = Field(None, description="UniProt accession if known")
    chains: Optional[List[str]] = Field(None, description="Subset of chain IDs to process")
    lambda_h: Optional[float] = Field(0.871, description="Hallucination sensitivity parameter")
    tau: Optional[float] = Field(0.5, description="High hallucination threshold")
    plddt_strict: Optional[float] = Field(70.0, description="Confident pLDDT threshold")
    ewcl_source: Optional[Literal["pdb_model", "af_proxy"]] = Field("pdb_model", description="EWCL source")
    af_proxy_csv: Optional[str] = Field(None, description="Path to AF-proxy CSV")

class HallucinationResponse(BaseModel):
    """Response for a single chain analysis."""
    status: Literal["ok", "no_confidence_available"]
    unit: Optional[str] = None
    uniprot: Optional[str] = None
    chain_id: str

    # NEW: source + overlap bookkeeping
    ewcl_source: Literal["pdb_model", "af_proxy"]
    confidence_type: Literal["plddt", "bfactor", "none"]
    n_res_total: int
    n_ewcl_finite: int
    n_plddt_finite: int
    n_overlap_used: int

    n_res: int
    mean_EWCL: Optional[float] = None
    mean_pLDDT: Optional[float] = None
    p95_H: Optional[float] = None
    frac_high_H: Optional[float] = None
    frac_disagree: Optional[float] = None
    flagged: Optional[bool] = None

    residues: conlist(ResidueScore, min_length=1)
    warnings: List[str] = []

class MultiChainHallucinationResponse(BaseModel):
    """Response for multi-chain analysis."""
    results: List[HallucinationResponse]

class OverlayMeta(BaseModel):
    """Metadata for overlay JSON."""
    uniprot: str
    unit: str
    lambda_h: float
    tau: float
    plddt_strict: float
    status: str

class OverlayResponse(BaseModel):
    """Overlay JSON for 3D viewer."""
    meta: OverlayMeta
    residues: List[ResidueScore]