from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from io import StringIO
from Bio import SeqIO
import os, numpy as np, pandas as pd
import logging
from functools import lru_cache
import warnings
from sklearn.exceptions import InconsistentVersionWarning

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
    residue_index: int
    aa: str
    cl: float
    hydropathy: float
    charge_pH7: float
    helix_prop: float
    sheet_prop: float

class EwclOut(BaseModel):
    id: str
    model: str
    length: int
    residues: List[ResidueOut]
    diagnostics: dict = {}

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
    except Exception as e:
        log.exception("[disorder] inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Build response with diagnostics
    out_list = []
    idx = np.arange(1, len(seq)+1)
    for i, a in zip(idx, seq):
        r = {
            "residue_index": int(i),
            "aa": a,
            "cl": float(p[i-1]),
            "hydropathy": float(feats.iloc[i-1]["hydropathy"]),
            "charge_pH7": float(feats.iloc[i-1]["charge_pH7"]),
            "helix_prop": float(feats.iloc[i-1]["helix_prop"]),
            "sheet_prop": float(feats.iloc[i-1]["sheet_prop"]),
        }
        out_list.append(r)

    # Diagnostics for debugging
    constant_preds = len(set(round(x["cl"], 6) for x in out_list)) <= 1
    if constant_preds:
        log.warning("[disorder] Constant predictions detected; check feature pipeline")

    diagnostics = {
        "constant_predictions": constant_preds,
        "feature_count": len(needed),
        "used_feature_names": needed[:8],  # small peek for debugging
        "has_feature_names_in": hasattr(mdl, "feature_names_in_")
    }

    return {
        "id": seq_id,
        "model": "disorder-collapse",  # Generic name, not ewclv1
        "length": len(seq),
        "residues": out_list,
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


