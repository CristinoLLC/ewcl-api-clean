from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from io import StringIO
from datetime import datetime
from Bio import SeqIO
import os, numpy as np, pandas as pd
import logging
from functools import lru_cache
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Version safety check for EWCLv1 model compatibility
try:
    from sklearn import __version__ as sklv
    if not sklv.startswith("1.7."):
        print(f"[ewclv1] ⚠️  Warning: EWCLv1 expects scikit-learn 1.7.x, got {sklv}")
        print("[ewclv1] Model outputs may be inconsistent due to version mismatch")
except ImportError:
    print("[ewclv1] ⚠️  scikit-learn not available for version check")

from backend.models.feature_extractors.ewclv1_features import build_ewclv1_features
from backend.models.loader import load_model_forgiving

router = APIRouter(prefix="/disorder", tags=["disorder"])
log = logging.getLogger("ewclv1")

# ⚠️  IMPORTANT: MODEL LOADING NOTE
# The EWCL models are saved with joblib, NOT pickle!
# Always use load_model_forgiving() which handles joblib/pickle/cloudpickle automatically
# DO NOT use pickle.load() directly - it will fail with "invalid load key" errors

# ---- schema you supplied (trimmed to keys we expose) ----
FEATURE_SCHEMA = {
    "model_version": "EWCLv1_Robust_PSSM",
    "trained_timestamp": "20250828_2210",
    "feature_source": "V5",
    "works_without_pssm": True,
    # NOTE: we generate the actual model order from model.feature_names_in_ at runtime
}

class ResidueOut(BaseModel):
    i: int
    aa: str
    ewcl: float
    hydropathy: float
    charge: float
    flex: float
    curv: float
    bin5: int
    z: Optional[float] = None

class WindowOut(BaseModel):
    start: int
    end: int
    mean_h: float
    mean_q: float
    mean_flex: float
    mean_curv: float
    mean_ewcl: float

class GlobalsOut(BaseModel):
    mean_h: float
    mean_q: float
    mean_ewcl: float
    pos_rate_tau: float
    auc_bins: Dict[str, int]

class MetaOut(BaseModel):
    date: str
    window_size: int
    stride: int
    tau: float
    z_norm: str
    scales: Dict[str, str]

class IDRRegion(BaseModel):
    start: int
    end: int
    len: int
    mean_ewcl: float
    range: List[float]
    aa_enrichment: Dict[str, float]
    evidence: str
    notes: str

class ProvenanceOut(BaseModel):
    software: str
    commit: str
    hydropathy_ref: str
    pka_set: str
    curvature_proxy_ref: str
    flex_proxy_ref: str

class DiagnosticsOut(BaseModel):
    constant_predictions: bool
    used_feature_names: List[str]
    n_windows: int
    warnings: List[str]

class EwclOut(BaseModel):
    id: str
    model: str
    meta: MetaOut
    length: int
    residues: List[ResidueOut]
    windows: List[WindowOut]
    globals: GlobalsOut
    idr_regions: List[IDRRegion]
    provenance: ProvenanceOut
    diagnostics: DiagnosticsOut

def _model_cache_key(path: str) -> tuple:
    """Generate cache key based on file path, mtime, and size"""
    st = os.stat(path)
    return (path, st.st_mtime, st.st_size)

@lru_cache(maxsize=2)
def _load_model_cached(cache_key: tuple):
    """Cached model loader - avoids reloading on every request"""
    path = cache_key[0]
    log.info(f"[ewclv1] Loading model from {path}")
    
    # Suppress sklearn version warnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        mdl = load_model_forgiving(path)
    
    return mdl

def _load_model():
    """
    Load EWCL v1 model using robust cached loader.
    
    ⚠️  CRITICAL: EWCL models are saved with joblib, not pickle!
    Using pickle.load() directly will fail with "invalid load key" errors.
    Always use load_model_forgiving() which handles multiple formats.
    """
    path = os.environ.get("EWCLV1_MODEL_PATH", "/app/models/disorder/ewclv1.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"EWCLv1 model not found at {path}")
    
    model = _load_model_cached(_model_cache_key(path))
    
    # Ensure model has feature_names_in_ to avoid warnings
    if not hasattr(model, "feature_names_in_"):
        # Load feature names from schema file
        import json
        schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend_bundle", "meta", "EWCLv1_feature_info.json")
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
                feature_names = schema["all_features"]
                # Set the feature names on the model to avoid warnings
                model.feature_names_in_ = np.array(feature_names)
                log.info(f"[ewclv1] Added feature_names_in_ to model ({len(feature_names)} features)")
        except Exception as e:
            log.warning(f"[ewclv1] Could not load feature schema: {e}, using hardcoded order")
            # Hardcoded fallback - exact 249 features from schema
            hardcoded_features = [
                "is_unknown_aa", "hydropathy", "polarity", "vdw_volume", "flexibility", "bulkiness",
                "helix_prop", "sheet_prop", "charge_pH7", "scd_local",
                "hydro_w5_mean", "hydro_w5_std", "hydro_w5_min", "hydro_w5_max",
                "polar_w5_mean", "polar_w5_std", "polar_w5_min", "polar_w5_max",
                "vdw_w5_mean", "vdw_w5_std", "vdw_w5_min", "vdw_w5_max",
                "flex_w5_mean", "flex_w5_std", "flex_w5_min", "flex_w5_max",
                "bulk_w5_mean", "bulk_w5_std", "bulk_w5_min", "bulk_w5_max",
                "helix_prop_w5_mean", "helix_prop_w5_std", "helix_prop_w5_min", "helix_prop_w5_max",
                "sheet_prop_w5_mean", "sheet_prop_w5_std", "sheet_prop_w5_min", "sheet_prop_w5_max",
                "charge_w5_mean", "charge_w5_std", "charge_w5_min", "charge_w5_max",
                "entropy_w5", "low_complex_w5", "comp_bias_w5", "uversky_dist_w5",
                "hydro_w11_mean", "hydro_w11_std", "hydro_w11_min", "hydro_w11_max",
                "polar_w11_mean", "polar_w11_std", "polar_w11_min", "polar_w11_max",
                "vdw_w11_mean", "vdw_w11_std", "vdw_w11_min", "vdw_w11_max",
                "flex_w11_mean", "flex_w11_std", "flex_w11_min", "flex_w11_max",
                "bulk_w11_mean", "bulk_w11_std", "bulk_w11_min", "bulk_w11_max",
                "helix_prop_w11_mean", "helix_prop_w11_std", "helix_prop_w11_min", "helix_prop_w11_max",
                "sheet_prop_w11_mean", "sheet_prop_w11_std", "sheet_prop_w11_min", "sheet_prop_w11_max",
                "charge_w11_mean", "charge_w11_std", "charge_w11_min", "charge_w11_max",
                "entropy_w11", "low_complex_w11", "comp_bias_w11", "uversky_dist_w11",
                "hydro_w25_mean", "hydro_w25_std", "hydro_w25_min", "hydro_w25_max",
                "polar_w25_mean", "polar_w25_std", "polar_w25_min", "polar_w25_max",
                "vdw_w25_mean", "vdw_w25_std", "vdw_w25_min", "vdw_w25_max",
                "flex_w25_mean", "flex_w25_std", "flex_w25_min", "flex_w25_max",
                "bulk_w25_mean", "bulk_w25_std", "bulk_w25_min", "bulk_w25_max",
                "helix_prop_w25_mean", "helix_prop_w25_std", "helix_prop_w25_min", "helix_prop_w25_max",
                "sheet_prop_w25_mean", "sheet_prop_w25_std", "sheet_prop_w25_min", "sheet_prop_w25_max",
                "charge_w25_mean", "charge_w25_std", "charge_w25_min", "charge_w25_max",
                "entropy_w25", "low_complex_w25", "comp_bias_w25", "uversky_dist_w25",
                "hydro_w50_mean", "hydro_w50_std", "hydro_w50_min", "hydro_w50_max",
                "polar_w50_mean", "polar_w50_std", "polar_w50_min", "polar_w50_max",
                "vdw_w50_mean", "vdw_w50_std", "vdw_w50_min", "vdw_w50_max",
                "flex_w50_mean", "flex_w50_std", "flex_w50_min", "flex_w50_max",
                "bulk_w50_mean", "bulk_w50_std", "bulk_w50_min", "bulk_w50_max",
                "helix_prop_w50_mean", "helix_prop_w50_std", "helix_prop_w50_min", "helix_prop_w50_max",
                "sheet_prop_w50_mean", "sheet_prop_w50_std", "sheet_prop_w50_min", "sheet_prop_w50_max",
                "charge_w50_mean", "charge_w50_std", "charge_w50_min", "charge_w50_max",
                "entropy_w50", "low_complex_w50", "comp_bias_w50", "uversky_dist_w50",
                "hydro_w100_mean", "hydro_w100_std", "hydro_w100_min", "hydro_w100_max",
                "polar_w100_mean", "polar_w100_std", "polar_w100_min", "polar_w100_max",
                "vdw_w100_mean", "vdw_w100_std", "vdw_w100_min", "vdw_w100_max",
                "flex_w100_mean", "flex_w100_std", "flex_w100_min", "flex_w100_max",
                "bulk_w100_mean", "bulk_w100_std", "bulk_w100_min", "bulk_w100_max",
                "helix_prop_w100_mean", "helix_prop_w100_std", "helix_prop_w100_min", "helix_prop_w100_max",
                "sheet_prop_w100_mean", "sheet_prop_w100_std", "sheet_prop_w100_min", "sheet_prop_w100_max",
                "charge_w100_mean", "charge_w100_std", "charge_w100_min", "charge_w100_max",
                "entropy_w100", "low_complex_w100", "comp_bias_w100", "uversky_dist_w100",
                "comp_D", "comp_Y", "comp_F", "comp_M", "comp_V", "comp_R", "comp_P", "comp_A",
                "comp_L", "comp_I", "comp_T", "comp_W", "comp_Q", "comp_N", "comp_K", "comp_E",
                "comp_G", "comp_S", "comp_H", "comp_C",
                "comp_frac_aromatic", "comp_frac_positive", "comp_frac_negative", "comp_frac_polar",
                "comp_frac_aliphatic", "comp_frac_proline", "comp_frac_glycine",
                "in_poly_P_run_ge3", "in_poly_E_run_ge3", "in_poly_K_run_ge3", "in_poly_Q_run_ge3",
                "in_poly_S_run_ge3", "in_poly_G_run_ge3", "in_poly_D_run_ge3", "in_poly_N_run_ge3",
                "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
                "pssm_entropy", "pssm_max_score", "pssm_variance", "has_pssm_data"
            ]
            model.feature_names_in_ = np.array(hardcoded_features)
            log.info(f"[ewclv1] Added hardcoded feature_names_in_ to model ({len(hardcoded_features)} features)")
    
    return model

@router.get("/ewclv1/schema")
def ewclv1_schema():
    return FEATURE_SCHEMA

@router.post("/analyze-fasta", response_model=EwclOut)
async def analyze_fasta(file: UploadFile = File(...)):
    # --- parse FASTA ---
    try:
        data = (await file.read()).decode("utf-8", errors="ignore")
        rec = next(SeqIO.parse(StringIO(data), "fasta"))
        seq_id = rec.id
        seq = str(rec.seq).upper()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FASTA: {e}")

    if not seq:
        raise HTTPException(status_code=400, detail="Empty sequence")

    # --- build features (no PSSM by default; your model is marked works_without_pssm) ---
    try:
        block = build_ewclv1_features(seq, pssm=None, expand_aa_onehot=False)
        feats = block.all_df
        
        # Debug: Check feature extraction
        log.info(f"[ewclv1] Features extracted: {feats.shape}, columns: {list(feats.columns[:10])}")
        if "hydropathy" in feats.columns:
            hydro_range = f"{feats['hydropathy'].min():.2f} - {feats['hydropathy'].max():.2f}"
            log.info(f"[ewclv1] Hydropathy range: {hydro_range}")
        if "charge_pH7" in feats.columns:
            charge_range = f"{feats['charge_pH7'].min():.2f} - {feats['charge_pH7'].max():.2f}"
            log.info(f"[ewclv1] Charge range: {charge_range}")
            
    except Exception as e:
        log.exception("[disorder] feature building failed")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # --- load model & align column order ---
    try:
        mdl = _load_model()
    except Exception as e:
        log.exception("[disorder] model loading failed")
        raise HTTPException(status_code=503, detail=f"Disorder model not loaded: {e}")

    # Use model's feature order when available, otherwise use exact hardcoded order
    if hasattr(mdl, "feature_names_in_"):
        needed = list(mdl.feature_names_in_)
        missing = [c for c in needed if c not in feats.columns]
        if missing:
            log.error(f"[ewclv1] Feature mismatch; missing {len(missing)} columns: {missing[:10]}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature mismatch; missing {len(missing)} columns, e.g. {missing[:10]}"
            )
        log.info(f"[ewclv1] Model expects {len(needed)} features, aligned successfully")
    else:
        # Hardcode the exact 249 features from the schema
        log.warning("[ewclv1] Model missing feature_names_in_, using hardcoded feature order")
        needed = [
            "is_unknown_aa", "hydropathy", "polarity", "vdw_volume", "flexibility", "bulkiness",
            "helix_prop", "sheet_prop", "charge_pH7", "scd_local",
            "hydro_w5_mean", "hydro_w5_std", "hydro_w5_min", "hydro_w5_max",
            "polar_w5_mean", "polar_w5_std", "polar_w5_min", "polar_w5_max",
            "vdw_w5_mean", "vdw_w5_std", "vdw_w5_min", "vdw_w5_max",
            "flex_w5_mean", "flex_w5_std", "flex_w5_min", "flex_w5_max",
            "bulk_w5_mean", "bulk_w5_std", "bulk_w5_min", "bulk_w5_max",
            "helix_prop_w5_mean", "helix_prop_w5_std", "helix_prop_w5_min", "helix_prop_w5_max",
            "sheet_prop_w5_mean", "sheet_prop_w5_std", "sheet_prop_w5_min", "sheet_prop_w5_max",
            "charge_w5_mean", "charge_w5_std", "charge_w5_min", "charge_w5_max",
            "entropy_w5", "low_complex_w5", "comp_bias_w5", "uversky_dist_w5",
            "hydro_w11_mean", "hydro_w11_std", "hydro_w11_min", "hydro_w11_max",
            "polar_w11_mean", "polar_w11_std", "polar_w11_min", "polar_w11_max",
            "vdw_w11_mean", "vdw_w11_std", "vdw_w11_min", "vdw_w11_max",
            "flex_w11_mean", "flex_w11_std", "flex_w11_min", "flex_w11_max",
            "bulk_w11_mean", "bulk_w11_std", "bulk_w11_min", "bulk_w11_max",
            "helix_prop_w11_mean", "helix_prop_w11_std", "helix_prop_w11_min", "helix_prop_w11_max",
            "sheet_prop_w11_mean", "sheet_prop_w11_std", "sheet_prop_w11_min", "sheet_prop_w11_max",
            "charge_w11_mean", "charge_w11_std", "charge_w11_min", "charge_w11_max",
            "entropy_w11", "low_complex_w11", "comp_bias_w11", "uversky_dist_w11",
            "hydro_w25_mean", "hydro_w25_std", "hydro_w25_min", "hydro_w25_max",
            "polar_w25_mean", "polar_w25_std", "polar_w25_min", "polar_w25_max",
            "vdw_w25_mean", "vdw_w25_std", "vdw_w25_min", "vdw_w25_max",
            "flex_w25_mean", "flex_w25_std", "flex_w25_min", "flex_w25_max",
            "bulk_w25_mean", "bulk_w25_std", "bulk_w25_min", "bulk_w25_max",
            "helix_prop_w25_mean", "helix_prop_w25_std", "helix_prop_w25_min", "helix_prop_w25_max",
            "sheet_prop_w25_mean", "sheet_prop_w25_std", "sheet_prop_w25_min", "sheet_prop_w25_max",
            "charge_w25_mean", "charge_w25_std", "charge_w25_min", "charge_w25_max",
            "entropy_w25", "low_complex_w25", "comp_bias_w25", "uversky_dist_w25",
            "hydro_w50_mean", "hydro_w50_std", "hydro_w50_min", "hydro_w50_max",
            "polar_w50_mean", "polar_w50_std", "polar_w50_min", "polar_w50_max",
            "vdw_w50_mean", "vdw_w50_std", "vdw_w50_min", "vdw_w50_max",
            "flex_w50_mean", "flex_w50_std", "flex_w50_min", "flex_w50_max",
            "bulk_w50_mean", "bulk_w50_std", "bulk_w50_min", "bulk_w50_max",
            "helix_prop_w50_mean", "helix_prop_w50_std", "helix_prop_w50_min", "helix_prop_w50_max",
            "sheet_prop_w50_mean", "sheet_prop_w50_std", "sheet_prop_w50_min", "sheet_prop_w50_max",
            "charge_w50_mean", "charge_w50_std", "charge_w50_min", "charge_w50_max",
            "entropy_w50", "low_complex_w50", "comp_bias_w50", "uversky_dist_w50",
            "hydro_w100_mean", "hydro_w100_std", "hydro_w100_min", "hydro_w100_max",
            "polar_w100_mean", "polar_w100_std", "polar_w100_min", "polar_w100_max",
            "vdw_w100_mean", "vdw_w100_std", "vdw_w100_min", "vdw_w100_max",
            "flex_w100_mean", "flex_w100_std", "flex_w100_min", "flex_w100_max",
            "bulk_w100_mean", "bulk_w100_std", "bulk_w100_min", "bulk_w100_max",
            "helix_prop_w100_mean", "helix_prop_w100_std", "helix_prop_w100_min", "helix_prop_w100_max",
            "sheet_prop_w100_mean", "sheet_prop_w100_std", "sheet_prop_w100_min", "sheet_prop_w100_max",
            "charge_w100_mean", "charge_w100_std", "charge_w100_min", "charge_w100_max",
            "entropy_w100", "low_complex_w100", "comp_bias_w100", "uversky_dist_w100",
            "comp_D", "comp_Y", "comp_F", "comp_M", "comp_V", "comp_R", "comp_P", "comp_A",
            "comp_L", "comp_I", "comp_T", "comp_W", "comp_Q", "comp_N", "comp_K", "comp_E",
            "comp_G", "comp_S", "comp_H", "comp_C",
            "comp_frac_aromatic", "comp_frac_positive", "comp_frac_negative", "comp_frac_polar",
            "comp_frac_aliphatic", "comp_frac_proline", "comp_frac_glycine",
            "in_poly_P_run_ge3", "in_poly_E_run_ge3", "in_poly_K_run_ge3", "in_poly_Q_run_ge3",
            "in_poly_S_run_ge3", "in_poly_G_run_ge3", "in_poly_D_run_ge3", "in_poly_N_run_ge3",
            "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
            "pssm_entropy", "pssm_max_score", "pssm_variance", "has_pssm_data"
        ]
        
        missing = [c for c in needed if c not in feats.columns]
        if missing:
            log.error(f"[ewclv1] Feature mismatch; missing {len(missing)} columns: {missing[:10]}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature mismatch; missing {len(missing)} columns, e.g. {missing[:10]}"
            )
        log.info(f"[ewclv1] Using hardcoded feature order ({len(needed)} features)")

    # Ensure numeric dtypes and compact representation
    X = feats[needed].astype("float32", copy=False).to_numpy()

    # --- predict ---
    try:
        # accept either predict_proba ([:,1]) or decision_function->sigmoid
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)[:, 1]
        else:
            z = mdl.decision_function(X)
            p = 1 / (1 + np.exp(-z))
        
        # Debug: Check prediction values
        log.info(f"[ewclv1] Predictions range: {np.min(p):.3f} - {np.max(p):.3f}, mean: {np.mean(p):.3f}")
        if np.all(p == 0):
            log.error("[ewclv1] All predictions are zero - model prediction failed!")
            
    except Exception as e:
        log.exception("[disorder] inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Academic-grade response construction with real features only
    
    # Extract available real features from the model
    available_features = set(feats.columns)
    
    # Check which secondary structure features are available
    has_helix_prop = "helix_prop" in available_features
    has_sheet_prop = "sheet_prop" in available_features
    has_flexibility = "flexibility" in available_features
    
    # Calculate z-scores for per-protein normalization
    ewcl_mean = float(np.mean(p))
    ewcl_std = float(np.std(p)) if len(p) > 1 else 1.0
    z_scores = [(float(score) - ewcl_mean) / ewcl_std for score in p] if ewcl_std > 0 else [0.0] * len(p)
    
    # Build 5-bin classifications (0=very ordered, 4=very disordered)
    def ewcl_to_bin5(score):
        if score < 0.2: return 0
        elif score < 0.4: return 1
        elif score < 0.6: return 2
        elif score < 0.8: return 3
        else: return 4
    
    # Build residue data with real features only
    residues = []
    for i, aa in enumerate(seq):
        row = feats.iloc[i]
        
        # Debug: Check if values are coming through correctly
        ewcl_val = float(p[i]) if i < len(p) else 0.0
        hydro_val = float(row["hydropathy"]) if "hydropathy" in row and pd.notna(row["hydropathy"]) else 0.0
        charge_val = float(row["charge_pH7"]) if "charge_pH7" in row and pd.notna(row["charge_pH7"]) else 0.0
        
        # Use actual flexibility/curvature from model features or proxy  
        flex_val = 0.0
        if has_flexibility and "flexibility" in row and pd.notna(row["flexibility"]):
            flex_val = float(row["flexibility"])
        elif has_helix_prop and "helix_prop" in row and pd.notna(row["helix_prop"]):
            flex_val = float(row["helix_prop"])
        
        curv_val = 0.0
        if has_sheet_prop and "sheet_prop" in row and pd.notna(row["sheet_prop"]):
            curv_val = float(row["sheet_prop"])
            
        # Log for debugging if all values are zero
        if ewcl_val == 0.0 and hydro_val == 0.0 and charge_val == 0.0:
            log.warning(f"[ewclv1] All zero values for residue {i+1}: {aa}")
            
        residues.append({
            "i": i + 1,
            "aa": aa,
            "ewcl": ewcl_val,
            "hydropathy": hydro_val,
            "charge": charge_val,
            "flex": flex_val,
            "curv": curv_val,
            "bin5": ewcl_to_bin5(ewcl_val),
            "z": z_scores[i] if i < len(z_scores) else 0.0
        })
    
    # Calculate sliding windows with stride=5 for performance
    window_size = 41
    stride = 5
    windows = []
    
    for start_pos in range(0, len(seq), stride):
        window_start = max(0, start_pos - window_size // 2)
        window_end = min(len(seq), start_pos + window_size // 2 + 1)
        
        if window_end - window_start < 10:  # Skip tiny windows
            continue
            
        # Extract window data
        window_hydro = [feats.iloc[j]["hydropathy"] for j in range(window_start, window_end)]
        window_charge = [feats.iloc[j]["charge_pH7"] for j in range(window_start, window_end)]
        window_flex = [feats.iloc[j]["flexibility"] if has_flexibility else feats.iloc[j]["helix_prop"] if has_helix_prop else 0.22 for j in range(window_start, window_end)]
        window_curv = [feats.iloc[j]["sheet_prop"] if has_sheet_prop else 0.08 for j in range(window_start, window_end)]
        window_ewcl = [p[j] for j in range(window_start, window_end)]
        
        windows.append({
            "start": window_start + 1,
            "end": window_end,
            "mean_h": float(np.mean(window_hydro)),
            "mean_q": float(np.mean(window_charge)),
            "mean_flex": float(np.mean(window_flex)),
            "mean_curv": float(np.mean(window_curv)),
            "mean_ewcl": float(np.mean(window_ewcl))
        })
    
    # Global statistics with academic rigor
    threshold = 0.5
    pos_rate_tau = float(np.mean(p >= threshold))
    
    # AUC bins for distribution analysis
    auc_bins = {
        "0-0.2": int(np.sum((p >= 0.0) & (p < 0.2))),
        "0.2-0.4": int(np.sum((p >= 0.2) & (p < 0.4))),
        "0.4-0.6": int(np.sum((p >= 0.4) & (p < 0.6))),
        "0.6-0.8": int(np.sum((p >= 0.6) & (p < 0.8))),
        "0.8-1.0": int(np.sum((p >= 0.8) & (p <= 1.0)))
    }
    
    globals_data = {
        "mean_h": float(feats["hydropathy"].mean()),
        "mean_q": float(feats["charge_pH7"].mean()),
        "mean_ewcl": ewcl_mean,
        "pos_rate_tau": pos_rate_tau,
        "auc_bins": auc_bins
    }
    
    # Enhanced IDR regions with amino acid enrichment analysis
    idr_regions = []
    in_idr = False
    idr_start = None
    min_length = 10
    
    for i, cl_val in enumerate(p):
        if cl_val > threshold:
            if not in_idr:
                idr_start = i
                in_idr = True
        else:
            if in_idr and (i - idr_start >= min_length):
                # Analyze amino acid composition in IDR
                idr_seq = seq[idr_start:i]
                aa_counts = {aa: idr_seq.count(aa) for aa in "DEKRPGQSTN"}  # Disorder-prone AAs
                aa_enrichment = {aa: count / len(idr_seq) for aa, count in aa_counts.items() if count > 0}
                
                idr_regions.append({
                    "start": idr_start + 1,  # 1-based
                    "end": i,
                    "len": i - idr_start,
                    "mean_ewcl": float(np.mean(p[idr_start:i])),
                    "range": [float(np.min(p[idr_start:i])), float(np.max(p[idr_start:i]))],
                    "aa_enrichment": aa_enrichment,
                    "evidence": "sequence-only",
                    "notes": "passes τ and min_len"
                })
                in_idr = False
    
    # Handle IDR at sequence end
    if in_idr and (len(seq) - idr_start >= min_length):
        idr_seq = seq[idr_start:]
        aa_counts = {aa: idr_seq.count(aa) for aa in "DEKRPGQSTN"}
        aa_enrichment = {aa: count / len(idr_seq) for aa, count in aa_counts.items() if count > 0}
        
        idr_regions.append({
            "start": idr_start + 1,
            "end": len(seq),
            "len": len(seq) - idr_start,
            "mean_ewcl": float(np.mean(p[idr_start:])),
            "range": [float(np.min(p[idr_start:])), float(np.max(p[idr_start:]))],
            "aa_enrichment": aa_enrichment,
            "evidence": "sequence-only",
            "notes": "passes τ and min_len"
        })
    
    # Academic provenance and metadata
    git_sha = os.environ.get("GIT_SHA", "unknown")
    
    meta_data = {
        "date": datetime.utcnow().isoformat() + "Z",
        "window_size": window_size,
        "stride": stride,
        "tau": threshold,
        "z_norm": "per-protein",
        "scales": {
            "hydropathy": "Kyte-Doolittle",
            "charge": "net@pH7.0",
            "flex_proxy": "flexibility" if has_flexibility else "helix-prop",
            "curvature_proxy": "sheet-prop" if has_sheet_prop else "turn/coil propensity"
        }
    }
    
    provenance = {
        "software": "ewcl-sequencer 1.0.3",
        "commit": git_sha,
        "hydropathy_ref": "Kyte & Doolittle 1982",
        "pka_set": "EMBOSS",
        "curvature_proxy_ref": "sheet propensity (Chou-Fasman variant)" if has_sheet_prop else "turn/coil propensity",
        "flex_proxy_ref": "flexibility" if has_flexibility else "helix propensity"
    }
    
    # Enhanced diagnostics
    constant_preds = len(set(round(score, 6) for score in p)) <= 1
    if constant_preds:
        log.warning("[disorder] Constant predictions detected; check feature pipeline")
    
    # Get actual feature names used (first 10 for brevity)
    used_features = ["hydropathy", "charge_pH7"]
    if has_flexibility: used_features.append("flexibility")
    if has_helix_prop: used_features.append("helix_prop") 
    if has_sheet_prop: used_features.append("sheet_prop")
    
    diagnostics = {
        "constant_predictions": constant_preds,
        "used_feature_names": used_features,
        "n_windows": len(windows),
        "warnings": ["Constant predictions detected"] if constant_preds else []
    }

    return {
        "id": seq_id,
        "model": "EWCL_v1.0",
        "meta": meta_data,
        "length": len(seq),
        "residues": residues,
        "windows": windows,
        "globals": globals_data,
        "idr_regions": idr_regions,
        "provenance": provenance,
        "diagnostics": diagnostics
    }

@router.get("/health")
def health_check():
    """Health check endpoint for disorder prediction model"""
    try:
        model_path = os.environ.get("EWCLV1_MODEL_PATH", "/app/models/disorder/ewclv1.pkl")
        model_exists = os.path.exists(model_path)
        
        if model_exists:
            try:
                mdl = _load_model()
                has_feature_names = hasattr(mdl, "feature_names_in_")
                return {
                    "ok": True,
                    "model": "disorder-collapse",  # Generic name
                    "status": "healthy",
                    "model_loaded": True,
                    "loaded": True,  # For compatibility with models endpoint
                    "has_feature_names": has_feature_names,
                    "feature_count": len(mdl.feature_names_in_) if has_feature_names else 249,
                    "loader_used": "load_model_forgiving (joblib compatible)",
                    "hardcoded_features": not has_feature_names,
                    "sklearn_warnings_suppressed": True
                }
            except Exception as e:
                return {
                    "ok": False,
                    "model": "disorder-collapse", 
                    "status": "error",
                    "model_loaded": False,
                    "loaded": False,
                    "error": str(e),
                    "note": "Ensure model was saved with joblib, not pickle"
                }
        else:
            return {
                "ok": False,
                "model": "disorder-collapse",
                "status": "error", 
                "model_loaded": False,
                "loaded": False,
                "error": "Model file not found"
            }
    except Exception as e:
        return {
            "ok": False,
            "model": "disorder-collapse",
            "status": "error",
            "loaded": False,
            "error": str(e)
        }


