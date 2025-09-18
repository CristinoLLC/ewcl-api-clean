"""
EWCL-H Hallucination Detection API Router
=========================================

FastAPI router for hallucination detection endpoints.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict
import re
import os
import numpy as np
import pandas as pd
import tempfile
import logging

# Version safety check for EWCLp3 model compatibility
try:
    from sklearn import __version__ as sklv
    if not sklv.startswith("1.7."):
        print(f"[ewcl] ⚠️  Warning: EWCLp3 artifact expects scikit-learn 1.7.x, got {sklv}")
        print("[ewcl] Model outputs may be inconsistent due to version mismatch")
except ImportError:
    print("[ewcl] ⚠️  scikit-learn not available for version check")

from backend.api._schema import (
    HallucinationResponse, 
    MultiChainHallucinationResponse, 
    ResidueScore
)
from backend.api.guards import (
    guard_uploaded_path, 
    validate_chains, 
    validate_hallucination_params
)
from backend.api.parsers.structures import load_chains, extract_residue_table
from backend.api.services.ewclp3 import ewclp3_predict
from backend.api.services.hallucination import compute_h_per_res

router = APIRouter(prefix="/api/hallucination", tags=["ewcl-h", "hallucination"])

# Initialize structured logger
logger = logging.getLogger("ewcl.hallucination")

# --- helpers: AF-proxy support ------------------------------------------------

def _infer_accession_from_unit(unit_name: str) -> Optional[str]:
    """
    Accepts filenames like 'AF-P41208-F1.cif' and returns 'P41208'.
    Falls back to None if no UniProt-like token is found.
    """
    if not unit_name:
        return None
    m = re.search(r'AF-([A-Z0-9]+)-F\d', unit_name.upper())
    return m.group(1) if m else None

def _load_ewcl_af_map(csv_path: str) -> pd.DataFrame:
    """
    Loads per-residue AF-proxy EWCL with columns:
      - uniprot (accession)
      - uniprot_pos (1-based)
      - ewcl (float)
    Supports .csv and .csv.gz
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"EWCL_AF CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # normalize column names flexibly
    cols = {c.lower(): c for c in df.columns}
    need = ["uniprot", "uniprot_pos", "ewcl"]
    for n in need:
        if n not in cols:
            raise ValueError(f"EWCL_AF CSV missing column '{n}'. Has: {list(df.columns)}")
    df = df[[cols["uniprot"], cols["uniprot_pos"], cols["ewcl"]]].copy()
    df[cols["uniprot"]] = df[cols["uniprot"]].astype(str).str.upper()
    df[cols["uniprot_pos"]] = df[cols["uniprot_pos"]].astype(int)
    df.rename(columns={cols["uniprot"]:"uniprot",
                       cols["uniprot_pos"]:"uniprot_pos",
                       cols["ewcl"]:"ewcl"}, inplace=True)
    return df

def _build_pos2ewcl_from_af(df: pd.DataFrame, acc: str) -> Dict[int, float]:
    """
    Filters df to one accession and returns {pos -> ewcl}.
    """
    if acc is None:
        return {}
    sub = df[df["uniprot"] == acc.upper()]
    return dict(zip(sub["uniprot_pos"].astype(int), sub["ewcl"].astype(float)))

@router.post("/evaluate", response_model=MultiChainHallucinationResponse)
async def evaluate_hallucination(
    file: UploadFile = File(..., description="PDB or mmCIF structure file"),
    uniprot: Optional[str] = Form(None, description="UniProt accession"),
    chains: Optional[str] = Form(None, description="Comma-separated chain IDs"),

    # H config (frozen thresholds from benchmark)
    lambda_h: Optional[float] = Form(0.871, description="Hallucination sensitivity"),
    tau: Optional[float] = Form(0.5, description="High hallucination threshold"),
    plddt_strict: Optional[float] = Form(70.0, description="Confident pLDDT threshold"),

    # EWCL source control with tolerant defaults
    ewcl_source: Optional[str] = Form("pdb_model", description="EWCL source: 'pdb_model' or 'af_proxy'"),
    af_proxy_csv: Optional[str] = Form(None, description="Path to per-residue AF-proxy CSV"),
    
    # NEW: Optional overlay toggle (non-breaking)
    use_overlay: Optional[bool] = Form(False, description="Use precomputed EWCL_AF if available"),
):
    """
    Evaluate hallucination scores for protein structure.
    
    Computes EWCL disorder predictions and hallucination scores based on
    disagreement between EWCL and pLDDT confidence values.
    
    Frozen hallucination math:
    - H = EWCL - λ * (pLDDT / 100)
    - high if H ≥ τ  
    - disagree if (pLDDT ≥ plddt_strict) and (H ≥ τ)
    """
    # 🛠️ DEFENSIVE: Default to pdb_model if not specified
    if not ewcl_source:
        ewcl_source = "pdb_model"
    
    # Validate parameters
    try:
        validate_hallucination_params(lambda_h, tau, plddt_strict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parameter validation failed: {e}")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    # 🛠️ DEFENSIVE: Preload AF-proxy map with graceful fallback
    af_proxy_df = None
    if ewcl_source == "af_proxy":
        if not af_proxy_csv:
            print(f"[ewcl-h] AF-proxy requested but no CSV provided, falling back to pdb_model")
            ewcl_source = "pdb_model"
        else:
            try:
                af_proxy_df = _load_ewcl_af_map(af_proxy_csv)
                print(f"[ewcl-h] Successfully loaded AF-proxy CSV: {len(af_proxy_df)} records")
            except Exception as e:
                print(f"[ewcl-h] Failed to load AF-proxy CSV, falling back to pdb_model: {e}")
                ewcl_source = "pdb_model"
    
    try:
        # Validate file with improved error messages
        try:
            guard_uploaded_path(tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"File validation failed for '{file.filename}': {e}")
        
        # Extract chains with better error handling
        try:
            all_chains = load_chains(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse structure '{file.filename}': {type(e).__name__}: {e}"
            )
        
        # Validate requested chains
        if chains:
            requested_chains = [c.strip() for c in chains.split(",") if c.strip()]
        else:
            requested_chains = all_chains
        
        try:
            valid_chains = validate_chains(requested_chains, all_chains)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=f"Chain validation failed for '{file.filename}': {e}")
        
        acc = (uniprot or _infer_accession_from_unit(file.filename)) or None
        
        results: List[HallucinationResponse] = []
        
        for chain_id in valid_chains:
            try:
                # Parse residue data with better error context
                try:
                    pos_list, _, plddt, af_like, warnings = extract_residue_table(tmp_path, chain_id)
                except Exception as e:
                    print(f"[ewcl-h] Failed to extract residue table for chain {chain_id}: {e}")
                    pos_list, plddt, af_like, warnings = [], None, False, [f"Failed to parse chain {chain_id}: {e}"]
                
                # 🆕 ENHANCED: EWCL source selection with overlay support
                ewcl_source_used = "pdb_model"  # Default
                
                # Build EWCL vector from chosen source
                if ewcl_source == "pdb_model":
                    try:
                        pos2ewcl = ewclp3_predict(tmp_path, chain_id)
                        ewcl_source_used = "pdb_model"
                        
                        # 🆕 Optional overlay fallback
                        if use_overlay and uniprot:
                            try:
                                from backend.api.services.overlay_cache import fetch_ewcl_af_for
                                pos2ewcl_af = fetch_ewcl_af_for(uniprot, chain_id)
                                if pos2ewcl_af:
                                    pos2ewcl = pos2ewcl_af
                                    ewcl_source_used = "af_overlay"
                                    print(f"[ewcl-h] Using AF overlay for {uniprot} chain {chain_id}")
                            except ImportError:
                                print(f"[ewcl-h] Overlay cache not available, using pdb_model")
                            except Exception as e:
                                print(f"[ewcl-h] Overlay fetch failed: {e}, using pdb_model")
                                
                    except Exception as e:
                        print(f"[ewcl-h] EWCLp3 prediction failed for chain {chain_id}: {e}")
                        pos2ewcl = {}
                        warnings.append(f"EWCL prediction failed: {e}")
                        
                elif ewcl_source == "af_proxy":
                    pos2ewcl = _build_pos2ewcl_from_af(af_proxy_df, acc)
                    ewcl_source_used = "af_proxy"
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown ewcl_source: '{ewcl_source}'. Must be 'pdb_model' or 'af_proxy'.")
                
                # 🛠️ DEFENSIVE: Handle cases with no residue data gracefully
                if not pos_list:
                    print(f"[ewcl-h] No residues found for chain {chain_id}, creating minimal response")
                    results.append(HallucinationResponse(
                        status="no_residues_found",
                        unit=file.filename or "unknown",
                        uniprot=acc or uniprot,
                        chain_id=chain_id,
                        ewcl_source=ewcl_source_used,
                        confidence_type="none",
                        n_res_total=0,
                        n_ewcl_finite=0,
                        n_plddt_finite=0,
                        n_overlap_used=0,
                        n_res=0,
                        mean_EWCL=None,
                        mean_pLDDT=None,
                        p95_H=None,
                        frac_high_H=None,
                        frac_disagree=None,
                        flagged=None,
                        residues=[],
                        warnings=warnings + ["No residues found in chain"]
                    ))
                    continue
                
                # Build feature matrix for additional properties
                try:
                    with open(tmp_path, 'rb') as f:
                        raw_bytes = f.read()
                    from backend.api.routers.ewclv1p3_fresh import load_structure_unified, FeatureExtractor
                    pdb_data = load_structure_unified(raw_bytes)
                    chain_residues = pdb_data["residues"]
                    sequence = [r["aa"] for r in chain_residues]
                    confidence_values = [r.get("bfactor", 0.0) for r in chain_residues]
                    extractor = FeatureExtractor(sequence, confidence_values, pdb_data["source"])
                    feature_matrix = extractor.extract_all_features()
                except Exception as e:
                    print(f"[ewcl-h] Feature extraction failed, using minimal fallback: {e}")
                    sequence = ["X"] * len(pos_list)
                    feature_matrix = None

                # Align vectors and compute statistics
                n_res_total = len(pos_list)
                ewcl_scores, plddt_scores, aa_sequence, hydro, charge = [], [], [], [], []
                
                for i, pos in enumerate(pos_list):
                    ewcl_scores.append(pos2ewcl.get(pos, np.nan))
                    plddt_scores.append(plddt[i] if (plddt is not None and i < len(plddt)) else np.nan)
                    aa_sequence.append(sequence[i] if i < len(sequence) else "X")
                    
                    if feature_matrix is not None and i < len(feature_matrix):
                        # Fixed: Use proper pandas iloc access instead of .get() method
                        hydro_val = feature_matrix.iloc[i]["hydropathy_x"] if "hydropathy_x" in feature_matrix.columns else 0.0
                        charge_val = feature_matrix.iloc[i]["charge_pH7"] if "charge_pH7" in feature_matrix.columns else 0.0
                        hydro.append(float(hydro_val))
                        charge.append(float(charge_val))
                    else:
                        hydro.append(0.0)
                        charge.append(0.0)

                ewcl_array = np.asarray(ewcl_scores, dtype=float)
                conf_array = np.asarray(plddt_scores, dtype=float)

                # Determine confidence type (robust confidence extraction)
                confidence_type = "plddt" if af_like else ("bfactor" if np.isfinite(conf_array).any() else "none")

                # 🆕 ENHANCED: Proper overlap counters
                n_ewcl_finite = int(np.isfinite(ewcl_array).sum())
                n_plddt_finite = int(np.isfinite(conf_array).sum()) if conf_array is not None else 0
                overlap_mask = np.isfinite(ewcl_array) & np.isfinite(conf_array)
                n_overlap_used = int(overlap_mask.sum())

                # 🛠️ DEFENSIVE: Compute H only if we have sufficient data
                plddt_array = conf_array
                H, is_high, is_disagree, chain_stats = None, None, None, None
                
                if confidence_type != "none" and n_overlap_used > 0:
                    try:
                        H, is_high, is_disagree, chain_stats = compute_h_per_res(
                            ewcl=ewcl_array,
                            plddt=plddt_array,
                            lambda_h=lambda_h,
                            tau=tau,
                            plddt_strict=plddt_strict
                        )
                    except Exception as e:
                        print(f"[ewcl-h] H computation failed: {e}")
                        warnings.append(f"Hallucination score computation failed: {e}")

                # Build residue outputs
                residues = []
                for i, pos in enumerate(pos_list):
                    conf_val = float(conf_array[i]) if np.isfinite(conf_array[i]) else None
                    residues.append(ResidueScore(
                        pos=pos,
                        aa=aa_sequence[i],
                        ewcl=float(ewcl_array[i]) if np.isfinite(ewcl_array[i]) else 0.0,
                        plddt=(conf_val if confidence_type == "plddt" else None),
                        bfactor=(conf_val if confidence_type == "bfactor" else None),
                        confidence=conf_val,
                        confidence_type=confidence_type,
                        H=(float(H[i]) if (H is not None and np.isfinite(H[i])) else None),
                        is_high_H=(bool(is_high[i]) if is_high is not None else None),
                        is_disagree=(bool(is_disagree[i]) if is_disagree is not None else None),
                        hydropathy=hydro[i],
                        charge_pH7=charge[i],
                        curvature=0.0
                    ))

                # Summary statistics
                mean_EWCL = float(np.nanmean(ewcl_array)) if n_ewcl_finite else None
                mean_pLDDT = float(np.nanmean(plddt_array)) if confidence_type != "none" else None

                p95_H = frac_high_H = frac_disagree = None
                flagged = None
                if chain_stats is not None:
                    p95_H = chain_stats["p95_H"]
                    frac_high_H = chain_stats["frac_high_H"]
                    frac_disagree = chain_stats["frac_disagree"]
                    flagged = bool(frac_disagree is not None and frac_disagree >= 0.20)

                # 🆕 STRUCTURED LOGGING: Audit trail per chain
                model_hash = os.environ.get("EWCLP3_MODEL_SHA", "unknown")
                logger.info("ewcl_h_chain", extra=dict(
                    unit=file.filename, chain_id=chain_id, ewcl_source=ewcl_source_used,
                    mean_EWCL=mean_EWCL, mean_pLDDT=mean_pLDDT,
                    p95_H=p95_H, frac_high_H=frac_high_H, frac_disagree=frac_disagree,
                    flagged=flagged, n_res_total=n_res_total,
                    n_ewcl_finite=n_ewcl_finite, n_plddt_finite=n_plddt_finite, 
                    n_overlap_used=n_overlap_used, model_hash=model_hash
                ))

                # 🛠️ DEFENSIVE: Always return success status, even with no confidence
                status = "ok" if confidence_type != "none" else "no_confidence_available"

                results.append(HallucinationResponse(
                    status=status,
                    unit=file.filename or "unknown",
                    uniprot=acc or uniprot,
                    chain_id=chain_id,
                    ewcl_source=ewcl_source_used,  # Use the actual source used
                    confidence_type=confidence_type,
                    n_res_total=n_res_total,
                    n_ewcl_finite=n_ewcl_finite,
                    n_plddt_finite=n_plddt_finite,
                    n_overlap_used=n_overlap_used,
                    n_res=len(residues),
                    mean_EWCL=mean_EWCL,
                    mean_pLDDT=mean_pLDDT,
                    p95_H=p95_H,
                    frac_high_H=frac_high_H,
                    frac_disagree=frac_disagree,
                    flagged=flagged,
                    residues=residues,
                    warnings=warnings + ([] if confidence_type != "none" else ["No pLDDT/B-factor found."])
                ))
                
            except Exception as e:
                # 🛠️ DEFENSIVE: Better error context and continue processing other chains
                print(f"[ewcl-h] Chain {chain_id} processing failed: {type(e).__name__}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Processing failed for chain {chain_id} in '{file.filename}': {type(e).__name__}: {e}"
                )
        
        return MultiChainHallucinationResponse(results=results)
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/health")
def health_check():
    """Health check for EWCL-H service with frozen configuration."""
    try:
        from backend.api.services.ewclp3 import get_ewclp3_service
        service = get_ewclp3_service()
        model_hash = os.environ.get("EWCLP3_MODEL_SHA", "unknown")
        
        # Check scikit-learn version compatibility
        try:
            from sklearn import __version__ as sklv
            sklearn_compatible = sklv.startswith("1.7.")
            sklearn_version = sklv
        except ImportError:
            sklearn_compatible = False
            sklearn_version = "not_available"
        
        return {
            "ok": True,
            "service": "ewcl-h",
            "ewclp3_loaded": service.model is not None,
            "gemmi_available": True,
            "sklearn_version": sklearn_version,
            "sklearn_compatible": sklearn_compatible,
            "model_hash": model_hash,
            "defaults": {
                "lambda_h": 0.871,
                "tau": 0.5,
                "plddt_strict": 70.0,
                "ewcl_source": "pdb_model"
            },
            "ewcl_sources_supported": ["pdb_model", "af_proxy"],
            "file_formats_supported": ["mmCIF", "PDB"],
            "frozen": True  # Indicates this is the locked-down version
        }
    except Exception as e:
        return {
            "ok": False,
            "service": "ewcl-h",
            "error": str(e),
            "frozen": True
        }