from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import os, io, numpy as np, pandas as pd
from Bio import SeqIO
from backend.models.singleton import get_model_manager, get_feature_extractor

# --- Use singleton model manager instead of local loading ---
_MODEL_NAME = "ewclv1-m"

# --- entropy overlays (for JSON response) ---
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

def _get_response_overlays(seq: str) -> Dict[str, np.ndarray]:
    """Compute simple overlays for the JSON response."""
    _AA = set("ARNDCQEGHILKMFPSTWYV")
    HYDRO = {
        'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,
        'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
    }
    CHARGE = {
        'D':-1,'E':-1,'K':+1,'R':+1,'H':0, 'A':0,'C':0,'Q':0,'N':0,'G':0,'I':0,'L':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0
    }
    aa = [c if c in _AA else 'X' for c in seq]
    hyd = np.array([HYDRO.get(a, 0.0) for a in aa], float)
    chg = np.array([CHARGE.get(a, 0.0) for a in aa], float)
    
    return {
        "hydro_entropy": _entropy01(hyd, 11),
        "charge_entropy": _entropy01(np.abs(chg), 11),
    }

router = APIRouter(prefix="/ewcl", tags=[_MODEL_NAME])

@router.post("/analyze-fasta/ewclv1-m")
async def analyze_fasta_ewclv1_m(file: UploadFile = File(...)):
    try:
        # Get singleton instances - models already loaded at startup
        model_manager = get_model_manager()
        feature_extractor = get_feature_extractor()
        
        if not model_manager.is_loaded(_MODEL_NAME):
            raise HTTPException(status_code=503, detail=f"Model {_MODEL_NAME} not loaded")
        
        if feature_extractor is None:
            raise HTTPException(status_code=503, detail="Feature extractor not available")
        
        raw = await file.read()
        try:
            record = next(SeqIO.parse(io.StringIO(raw.decode("utf-8")), "fasta"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid FASTA")
        
        seq_id = record.id
        seq = str(record.seq).upper()
        if not seq:
            raise HTTPException(status_code=400, detail="Empty sequence")

        print(f"[ewclv1-m] Processing sequence {seq_id} with {len(seq)} residues")

        # Use the real production feature extractor (already in memory)
        feature_df = feature_extractor.extract_sequence_features(seq, seq_id)
        print(f"[ewclv1-m] Extracted features: {feature_df.shape}")
        
        # Make prediction using singleton model (already in memory)
        cl = model_manager.predict_proba(_MODEL_NAME, feature_df)

        conf = 1.0 - np.abs(cl - 0.5) * 2.0
        conf = np.clip(conf, 0.0, 1.0)

        overlays = _get_response_overlays(seq)

        residues = []
        for i, a in enumerate(seq, start=1):
            residues.append({
                "residue_index": i, "aa": a,
                "cl": float(cl[i-1]), "confidence": float(conf[i-1]),
                "hydro_entropy": float(overlays["hydro_entropy"][i-1]),
                "charge_entropy": float(overlays["charge_entropy"][i-1]),
                "curvature": None, "flips": 0.0,
            })

        print(f"[ewclv1-m] Successfully processed {len(residues)} residues")
        return JSONResponse(content={
            "id": seq_id, "model": _MODEL_NAME, "length": len(seq), "residues": residues
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ewclv1-m] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{_MODEL_NAME} FASTA analysis failed: {e}")

@router.get("/analyze-fasta/ewclv1-m/health")
def health_check():
    try:
        model_manager = get_model_manager()
        feature_extractor = get_feature_extractor()
        
        return {
            "ok": True, 
            "model_name": _MODEL_NAME, 
            "loaded": model_manager.is_loaded(_MODEL_NAME), 
            "features": len(model_manager.get_feature_order(_MODEL_NAME)),
            "feature_extractor": feature_extractor is not None
        }
    except Exception as e:
        return {"ok": False, "model_name": _MODEL_NAME, "loaded": False, "error": str(e)}
