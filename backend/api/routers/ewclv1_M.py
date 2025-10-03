from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
from functools import lru_cache
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import os, io, joblib, numpy as np
from Bio import SeqIO
import pandas as pd
import logging

# Version safety check for EWCLv1-M model compatibility
try:
    from sklearn import __version__ as sklv
    if not sklv.startswith("1.7."):
        print(f"[ewclv1-m] ⚠️  Warning: EWCLv1-M expects scikit-learn 1.7.x, got {sklv}")
        print("[ewclv1-m] Model outputs may be inconsistent due to version mismatch")
except ImportError:
    print("[ewclv1-m] ⚠️  scikit-learn not available for version check")

router = APIRouter()
log = logging.getLogger("ewclv1-m")

MODEL_NAME = "ewclv1-m"

def _model_cache_key(path: str) -> tuple:
    """Generate cache key based on file path, mtime, and size"""
    st = os.stat(path)
    return (path, st.st_mtime, st.st_size)

@lru_cache(maxsize=2)
def _load_model_cached(cache_key: tuple):
    """Cached model loader - avoids reloading on every request"""
    path = cache_key[0]
    log.info(f"[ewclv1-m] Loading model from {path}")
    
    # Suppress sklearn version warnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        model = joblib.load(path)
    
    return model

def _load_model():
    """Load EWCL v1-M model using robust cached loader with sklearn warning suppression"""
    global _FEATURE_ORDER
    
    model_path = os.environ.get("EWCLV1_M_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"EWCLv1-M model not found. Set EWCLV1_M_MODEL_PATH (got: {model_path})")
    
    model = _load_model_cached(_model_cache_key(model_path))
    
    # Ensure model has feature_names_in_ to avoid warnings
    if not hasattr(model, "feature_names_in_"):
        # Load feature names from schema file
        import json
        schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend_bundle", "meta", "EWCLv1-M_feature_info.json")
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
                feature_names = schema["all_features"]
                # Set the feature names on the model to avoid warnings
                model.feature_names_in_ = np.array(feature_names)
                _FEATURE_ORDER = list(feature_names)
                log.info(f"[ewclv1-m] Added feature_names_in_ to model ({len(feature_names)} features)")
                return model
        except Exception as e:
            log.warning(f"[ewclv1-m] Could not load feature schema: {e}, using hardcoded order")
    
    # Try to discover feature order from the model if we have feature_names_in_
    if hasattr(model, "feature_names_in_"):
        _FEATURE_ORDER = list(model.feature_names_in_)
        log.info(f"[ewclv1-m] Using model feature order ({len(_FEATURE_ORDER)} features)")
        return model
    
    # Fallback: hardcoded feature order - EXACT 255 features from EWCLv1-M schema
    log.warning("[ewclv1-m] Model missing feature_names_in_, using hardcoded feature order")
    _FEATURE_ORDER = [
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
        "pssm_entropy", "pssm_max_score", "pssm_variance", "pssm_native", "pssm_top1", "pssm_top2", 
        "pssm_gap", "pssm_sum_hydrophobic", "pssm_sum_polar", "pssm_sum_charged"
    ]
    
    # Set the feature names on the model to avoid future warnings
    model.feature_names_in_ = np.array(_FEATURE_ORDER)
    log.info(f"[ewclv1-m] Added hardcoded feature_names_in_ to model ({len(_FEATURE_ORDER)} features)")
    
    return model

# Global variables for caching
_MODEL = None
_FEATURE_ORDER: List[str] = []

_AA = set("ARNDCQEGHILKMFPSTWYV")
HYDRO = { 
    'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,
    'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}
POLAR = { 
    'R':52,'K':49,'D':49,'E':49,'Q':41,'N':40,'H':51,'Y':41,'W':42,'S':32,'T':32,
    'G':0,'P':27,'C':15,'A':8,'M':5,'F':5,'L':4,'V':4,'I':5
}
VDW  = { 
    'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
    'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
    'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0
}
FLEX = { 
    'G':1.00,'S':0.82,'D':0.80,'P':0.73,'N':0.73,'E':0.67,'Q':0.67,'K':0.62,
    'T':0.60,'R':0.60,'A':0.55,'W':0.54,'M':0.52,'H':0.52,'F':0.52,
    'Y':0.51,'I':0.47,'L':0.47,'V':0.46,'C':0.35
}
BULK = { 
    'G':3.4,'A':11.5,'S':9.2,'P':17.4,'V':21.6,'T':15.9,'C':13.5,'I':21.4,'L':21.4,'D':13.0,
    'Q':17.2,'K':15.7,'E':12.3,'N':12.8,'H':21.0,'F':19.8,'Y':18.0,'M':16.3,'R':14.3,'W':21.6
}
HELIX = { 
    'A':1.45,'C':0.77,'D':1.01,'E':1.51,'F':1.13,'G':0.53,'H':1.24,'I':1.08,'K':1.16,
    'L':1.34,'M':1.20,'N':0.67,'P':0.59,'Q':1.11,'R':0.79,'S':0.79,'T':0.82,'V':1.06,'W':1.14,'Y':0.61
}
SHEET = { 
    'A':0.97,'C':1.30,'D':0.54,'E':0.37,'F':1.38,'G':0.81,'H':0.71,'I':1.60,'K':0.74,
    'L':1.22,'M':1.67,'N':0.89,'P':0.62,'Q':1.10,'R':0.90,'S':0.72,'T':1.20,'V':1.70,'W':1.19,'Y':1.29
}
CHARGE = { 
    'D':-1,'E':-1,'K':+1,'R':+1,'H':0,  
    'A':0,'C':0,'Q':0,'N':0,'G':0,'I':0,'L':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0
}

def _sliding_stats(arr: np.ndarray, w: int):
    n = len(arr); hw = w//2
    mean = np.zeros(n); std = np.zeros(n)
    vmin = np.zeros(n); vmax = np.zeros(n)
    for i in range(n):
        s = max(0, i-hw); e = min(n, i+hw+1)
        win = arr[s:e]
        mean[i] = float(np.mean(win)) if len(win) else 0.0
        std[i]  = float(np.std(win)) if len(win) else 0.0
        vmin[i] = float(np.min(win)) if len(win) else 0.0
        vmax[i] = float(np.max(win)) if len(win) else 0.0
    return mean, std, vmin, vmax

def _entropy01(arr: np.ndarray, w: int, bins: int = 10):
    n = len(arr); hw = w//2
    out = np.zeros(n)
    for i in range(n):
        s = max(0, i-hw); e = min(n, i+hw+1)
        x = arr[s:e]
        if not len(x): 
            out[i] = 0.0
            continue
        xx = x
        if np.max(xx) > np.min(xx):
            xx = (xx - np.min(xx)) / (np.ptp(xx))
        hist, _ = np.histogram(xx, bins=bins, range=(0,1), density=True)
        p = hist + 1e-12
        p /= p.sum()
        out[i] = float(-(p * np.log(p)).sum())  
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out

def _sequencer_features(seq: str) -> Dict[int, Dict[str, float]]:
    L = len(seq)
    aa = [c if c in _AA else 'X' for c in seq]

    hyd = np.array([HYDRO.get(a, 0.0) for a in aa], float)
    pol = np.array([POLAR.get(a, 0.0) for a in aa], float)
    vol = np.array([VDW.get(a, 0.0) for a in aa], float)
    flex= np.array([FLEX.get(a, 0.0) for a in aa], float)
    bulk= np.array([BULK.get(a, 0.0) for a in aa], float)
    hel = np.array([HELIX.get(a, 0.0) for a in aa], float)
    sht = np.array([SHEET.get(a, 0.0) for a in aa], float)
    chg = np.array([CHARGE.get(a, 0.0) for a in aa], float)

    h5 = _sliding_stats(hyd, 5)
    p5 = _sliding_stats(pol, 5)
    c5 = _sliding_stats(chg, 5)
    h11m, h11s, _, _ = _sliding_stats(hyd, 11)
    p11m, p11s, _, _ = _sliding_stats(pol, 11)
    c11m, c11s, _, _ = _sliding_stats(chg, 11)

    scd = np.abs(c5[0])  

    hydro_ent = _entropy01(hyd, 11)
    charge_ent= _entropy01(np.abs(chg), 11)

    feats_by_pos = {}
    for i in range(L):
        feats_by_pos[i+1] = {
            "is_unknown_aa": 0.0 if aa[i] in _AA else 1.0,
            "hydropathy": float(hyd[i]),
            "polarity": float(pol[i]),
            "vdw_volume": float(vol[i]),
            "flexibility": float(flex[i]),
            "bulkiness": float(bulk[i]),
            "helix_prop": float(hel[i]),
            "sheet_prop": float(sht[i]),
            "charge_pH7": float(chg[i]),
            "scd_local": float(scd) if i < len(scd) else 0.0,
            "hydro_w5_mean": float(h5[0][i]),
            "hydro_w5_std": float(h5[1][i]),
            "hydro_w5_min": float(h5[2][i]),
            "hydro_w5_max": float(h5[3][i]),
            "polar_w5_mean": float(p5[0][i]),
            "polar_w5_std": float(p5[1][i]),
            "polar_w5_min": float(p5[2][i]),
            "polar_w5_max": float(p5[3][i]),
            "charge_w5_mean": float(c5[0][i]),
            "charge_w5_std": float(c5[1][i]),
            "charge_w5_min": float(c5[2][i]),
            "charge_w5_max": float(c5[3][i]),
            "hydro_w11_mean": float(h11m[i]),
            "hydro_w11_std": float(h11s[i]),
            "polar_w11_mean": float(p11m[i]),
            "polar_w11_std": float(p11s[i]),
            "charge_w11_mean": float(c11m[i]),
            "charge_w11_std": float(c11s[i]),
            "hydro_entropy_w11": float(hydro_ent[i]),
            "charge_entropy_w11": float(charge_ent[i]),
        }
    return feats_by_pos

def _feature_order():
    _load_model()
    return _FEATURE_ORDER

def _featurize(seq: str) -> pd.DataFrame:
    feats = _sequencer_features(seq)
    return pd.DataFrame.from_dict(feats, orient="index")

def _align_X(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    return df.reindex(columns=order, fill_value=0.0)

def _confidence(p: np.ndarray) -> np.ndarray:
    return np.abs(p - 0.5) * 2.0

@router.post("/ewcl/analyze-fasta/ewclv1-m")
async def analyze_fasta_ewclv1m(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        rec = next(SeqIO.parse(io.StringIO(raw.decode("utf-8", errors="ignore")), "fasta"))
        pid, seq = rec.id, str(rec.seq)
    except Exception as e:
        raise HTTPException(400, f"invalid FASTA: {e}")

    if not seq:
        raise HTTPException(400, "empty sequence")

    # Use the comprehensive feature extractor (same as EWCLv1 but with enhanced PSSM features)
    try:
        from backend.models.feature_extractors.ewclv1_features import build_ewclv1_features
        block = build_ewclv1_features(seq, pssm=None, expand_aa_onehot=False)
        df = block.all_df
        
        # Add the additional PSSM features that EWCLv1-M expects but EWCLv1 doesn't have
        # These are: pssm_native, pssm_top1, pssm_top2, pssm_gap, pssm_sum_hydrophobic, pssm_sum_polar, pssm_sum_charged
        n = len(seq)
        df["pssm_native"] = np.zeros(n)
        df["pssm_top1"] = np.zeros(n) 
        df["pssm_top2"] = np.zeros(n)
        df["pssm_gap"] = np.zeros(n)
        df["pssm_sum_hydrophobic"] = np.zeros(n)
        df["pssm_sum_polar"] = np.zeros(n)
        df["pssm_sum_charged"] = np.zeros(n)
        
    except Exception as e:
        log.exception("[ewclv1-m] feature building failed")
        raise HTTPException(500, f"feature building failed: {e}")

    n = len(seq)
    if len(df) != n:
        raise HTTPException(500, f"feature rows != length: {len(df)} vs {n}")

    try:
        model = _load_model()
        order = _FEATURE_ORDER
    except Exception as e:
        log.exception("[ewclv1-m] model loading failed")
        raise HTTPException(503, f"Model ewclv1-m not loaded: {e}")

    # Ensure we have all required features and align with model expectations
    missing = [c for c in order if c not in df.columns]
    if missing:
        log.error(f"[ewclv1-m] Feature mismatch; missing {len(missing)} columns: {missing[:10]}")
        raise HTTPException(500, f"Feature mismatch; missing {len(missing)} columns, e.g. {missing[:10]}")

    # Ensure numeric dtypes and compact representation
    X = df[order].astype("float32", copy=False).to_numpy()

    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[:, 1]
        else:
            z = model.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-z))
    except Exception as e:
        log.exception("[ewclv1-m] prediction failed")
        raise HTTPException(500, f"prediction failed: {e}")

    # Check for constant predictions
    constant_preds = len(set(round(x, 6) for x in p)) <= 1
    if constant_preds:
        log.warning("[ewclv1-m] Constant predictions detected; check feature pipeline")

    conf = _confidence(p)

    # Build response with key features
    residues = []
    for i, aa in enumerate(seq, start=1):
        item = {
            "residue_index": i,
            "aa": aa,
            "cl": float(p[i-1]),
            "confidence": float(conf[i-1]),
            "hydropathy": float(df.iloc[i-1]["hydropathy"]),
            "charge_pH7": float(df.iloc[i-1]["charge_pH7"]),
            "helix_prop": float(df.iloc[i-1]["helix_prop"]),
            "sheet_prop": float(df.iloc[i-1]["sheet_prop"]),
        }
        residues.append(item)

    # Add diagnostics
    diagnostics = {
        "constant_predictions": constant_preds,
        "feature_count": len(order),
        "used_feature_names": order[:8],  # small peek for debugging
        "has_feature_names_in": hasattr(model, "feature_names_in_")
    }

    return {
        "id": pid,
        "model": MODEL_NAME,
        "length": n,
        "residues": residues,
        "diagnostics": diagnostics
    }

@router.get("/ewcl/analyze-fasta/ewclv1-m/health")
def health_check():
    """Health check endpoint for EWCL v1-M model"""
    try:
        model_path = os.environ.get("EWCLV1_M_MODEL_PATH")
        model_exists = model_path and os.path.exists(model_path)
        
        if model_exists:
            try:
                model = _load_model()
                has_feature_names = hasattr(model, "feature_names_in_")
                return {
                    "model": "ewclv1-m",
                    "status": "healthy",
                    "model_path": model_path,
                    "model_loaded": True,
                    "has_feature_names": has_feature_names,
                    "feature_count": len(model.feature_names_in_) if has_feature_names else len(_FEATURE_ORDER),
                    "loader_used": "joblib.load (cached)",
                    "hardcoded_features": not has_feature_names,
                    "sklearn_warnings_suppressed": True
                }
            except Exception as e:
                return {
                    "model": "ewclv1-m", 
                    "status": "error",
                    "model_path": model_path,
                    "model_loaded": False,
                    "error": str(e),
                    "note": "Ensure model was saved with joblib"
                }
        else:
            return {
                "model": "ewclv1-m",
                "status": "error", 
                "model_path": model_path,
                "model_loaded": False,
                "error": "Model file not found or path not set"
            }
    except Exception as e:
        return {
            "model": "ewclv1-m",
            "status": "error",
            "error": str(e)
        }
