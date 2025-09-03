from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import os, io, numpy as np, pandas as pd
from pathlib import Path
from Bio import SeqIO
from backend.models.loader import load_model_forgiving

# Exact features for ewclv1-M.pkl (255 features) - extracted from model
EWCLV1_M_255_FEATURES = [
    "is_unknown_aa", "hydropathy", "polarity", "vdw_volume", "flexibility", "bulkiness", "helix_prop", "sheet_prop", "charge_pH7", "scd_local",
    "hydro_w5_mean", "hydro_w5_std", "hydro_w5_min", "hydro_w5_max", "polar_w5_mean", "polar_w5_std", "polar_w5_min", "polar_w5_max",
    "vdw_w5_mean", "vdw_w5_std", "vdw_w5_min", "vdw_w5_max", "flex_w5_mean", "flex_w5_std", "flex_w5_min", "flex_w5_max",
    "bulk_w5_mean", "bulk_w5_std", "bulk_w5_min", "bulk_w5_max", "helix_prop_w5_mean", "helix_prop_w5_std", "helix_prop_w5_min", "helix_prop_w5_max",
    "sheet_prop_w5_mean", "sheet_prop_w5_std", "sheet_prop_w5_min", "sheet_prop_w5_max", "charge_w5_mean", "charge_w5_std", "charge_w5_min", "charge_w5_max",
    "entropy_w5", "low_complex_w5", "comp_bias_w5", "uversky_dist_w5", "hydro_w11_mean", "hydro_w11_std", "hydro_w11_min", "hydro_w11_max",
    "polar_w11_mean", "polar_w11_std", "polar_w11_min", "polar_w11_max", "vdw_w11_mean", "vdw_w11_std", "vdw_w11_min", "vdw_w11_max",
    "flex_w11_mean", "flex_w11_std", "flex_w11_min", "flex_w11_max", "bulk_w11_mean", "bulk_w11_std", "bulk_w11_min", "bulk_w11_max",
    "helix_prop_w11_mean", "helix_prop_w11_std", "helix_prop_w11_min", "helix_prop_w11_max", "sheet_prop_w11_mean", "sheet_prop_w11_std", "sheet_prop_w11_min", "sheet_prop_w11_max",
    "charge_w11_mean", "charge_w11_std", "charge_w11_min", "charge_w11_max", "entropy_w11", "low_complex_w11", "comp_bias_w11", "uversky_dist_w11",
    "hydro_w25_mean", "hydro_w25_std", "hydro_w25_min", "hydro_w25_max", "polar_w25_mean", "polar_w25_std", "polar_w25_min", "polar_w25_max",
    "vdw_w25_mean", "vdw_w25_std", "vdw_w25_min", "vdw_w25_max", "flex_w25_mean", "flex_w25_std", "flex_w25_min", "flex_w25_max",
    "bulk_w25_mean", "bulk_w25_std", "bulk_w25_min", "bulk_w25_max", "helix_prop_w25_mean", "helix_prop_w25_std", "helix_prop_w25_min", "helix_prop_w25_max",
    "sheet_prop_w25_mean", "sheet_prop_w25_std", "sheet_prop_w25_min", "sheet_prop_w25_max", "charge_w25_mean", "charge_w25_std", "charge_w25_min", "charge_w25_max",
    "entropy_w25", "low_complex_w25", "comp_bias_w25", "uversky_dist_w25", "hydro_w50_mean", "hydro_w50_std", "hydro_w50_min", "hydro_w50_max",
    "polar_w50_mean", "polar_w50_std", "polar_w50_min", "polar_w50_max", "vdw_w50_mean", "vdw_w50_std", "vdw_w50_min", "vdw_w50_max",
    "flex_w50_mean", "flex_w50_std", "flex_w50_min", "flex_w50_max", "bulk_w50_mean", "bulk_w50_std", "bulk_w50_min", "bulk_w50_max",
    "helix_prop_w50_mean", "helix_prop_w50_std", "helix_prop_w50_min", "helix_prop_w50_max", "sheet_prop_w50_mean", "sheet_prop_w50_std", "sheet_prop_w50_min", "sheet_prop_w50_max",
    "charge_w50_mean", "charge_w50_std", "charge_w50_min", "charge_w50_max", "entropy_w50", "low_complex_w50", "comp_bias_w50", "uversky_dist_w50",
    "hydro_w100_mean", "hydro_w100_std", "hydro_w100_min", "hydro_w100_max", "polar_w100_mean", "polar_w100_std", "polar_w100_min", "polar_w100_max",
    "vdw_w100_mean", "vdw_w100_std", "vdw_w100_min", "vdw_w100_max", "flex_w100_mean", "flex_w100_std", "flex_w100_min", "flex_w100_max",
    "bulk_w100_mean", "bulk_w100_std", "bulk_w100_min", "bulk_w100_max", "helix_prop_w100_mean", "helix_prop_w100_std", "helix_prop_w100_min", "helix_prop_w100_max",
    "sheet_prop_w100_mean", "sheet_prop_w100_std", "sheet_prop_w100_min", "sheet_prop_w100_max", "charge_w100_mean", "charge_w100_std", "charge_w100_min", "charge_w100_max",
    "entropy_w100", "low_complex_w100", "comp_bias_w100", "uversky_dist_w100", "comp_D", "comp_Y", "comp_F", "comp_M", "comp_V", "comp_R",
    "comp_P", "comp_A", "comp_L", "comp_I", "comp_T", "comp_W", "comp_Q", "comp_N", "comp_K", "comp_E", "comp_G", "comp_S", "comp_H", "comp_C",
    "comp_frac_aromatic", "comp_frac_positive", "comp_frac_negative", "comp_frac_polar", "comp_frac_aliphatic", "comp_frac_proline", "comp_frac_glycine",
    "in_poly_P_run_ge3", "in_poly_E_run_ge3", "in_poly_K_run_ge3", "in_poly_Q_run_ge3", "in_poly_S_run_ge3", "in_poly_G_run_ge3", "in_poly_D_run_ge3", "in_poly_N_run_ge3",
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "pssm_entropy", "pssm_max_score", "pssm_variance", "pssm_native", "pssm_top1", "pssm_top2", "pssm_gap", "pssm_sum_hydrophobic", "pssm_sum_polar", "pssm_sum_charged"
]

_MODEL_NAME = "ewclv1m"
MODEL = None

def _get_model():
    global MODEL
    if MODEL is None:
        path = os.environ.get("EWCLV1_M_MODEL_PATH")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=503, detail="Model path missing or file not found")
        MODEL = load_model_forgiving(path)
    return MODEL

router = APIRouter(prefix="/ewcl", tags=[_MODEL_NAME])

# Mock feature extractor for EWCLv1-M (255 features)
def _mock_feature_extraction(seq: str) -> pd.DataFrame:
    """
    Mock feature extraction for EWCLv1-M that returns zero features.
    In production, this would be replaced with real EWCL feature extraction.
    """
    seq_len = len(seq)
    features = {}
    
    # Fill all 255 features with zeros/defaults
    for feat_name in EWCLV1_M_255_FEATURES:
        if "length" in feat_name or "count" in feat_name:
            features[feat_name] = float(seq_len)
        elif feat_name.startswith("comp_"):
            # Composition features - mock with small random values
            features[feat_name] = np.random.uniform(0.0, 0.1)
        elif feat_name.startswith("pssm_"):
            # PSSM features - mock with small values
            features[feat_name] = np.random.uniform(0.0, 0.5)
        else:
            features[feat_name] = 0.0
    
    # Create per-residue features (repeat for each position)
    rows = []
    for i in range(seq_len):
        row = features.copy()
        rows.append(row)
    
    return pd.DataFrame(rows, columns=EWCLV1_M_255_FEATURES)

@router.get("/analyze-fasta/ewclv1-m/health")
def health_check():
    """Health check for EWCLv1-M model."""
    try:
        model = _get_model()
        return {
            "ok": True,
            "model_name": _MODEL_NAME,
            "loaded": model is not None,
            "features": len(EWCLV1_M_255_FEATURES),
            "feature_extractor": True  # Mock extractor always available
        }
    except Exception as e:
        return {
            "ok": False,
            "model_name": _MODEL_NAME,
            "loaded": False,
            "error": str(e),
            "features": len(EWCLV1_M_255_FEATURES),
            "feature_extractor": True
        }

@router.post("/analyze-fasta/ewclv1-m")
async def analyze_fasta_ewclv1_m(file: UploadFile = File(...)):
    """Analyze FASTA sequence using EWCLv1-M model (255 features)."""
    model = _get_model()
    
    try:
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

        # Extract features using mock extractor
        feature_df = _mock_feature_extraction(seq)
        print(f"[ewclv1-m] Extracted features: {feature_df.shape}")
        
        # Make prediction using singleton model
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(feature_df)
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                cl = predictions[:, 1]  # Probability of positive class
           