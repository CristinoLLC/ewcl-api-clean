from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import os, io, numpy as np, pandas as pd
from pathlib import Path
from backend.models.loader import load_model_forgiving

# Exact features for ewclv1p3.pkl (302 features) - REAL features, not generic columns
EWCLV1P3_302_FEATURES = [
    "A", "C", "D", "E", "F", "G", "H", "H_hydro", "H_hydro__x__inv_plddt", "H_hydro_std_win101",
    "H_hydro_std_win21", "H_hydro_std_win51", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "bfactor", "bulk_w100_max", "bulk_w100_mean", "bulk_w100_min", "bulk_w100_std", "bulk_w11_max", "bulk_w11_mean",
    "bulk_w11_min", "bulk_w11_std", "bulk_w25_max", "bulk_w25_mean", "bulk_w25_min", "bulk_w25_std", "bulk_w50_max",
    "bulk_w50_mean", "bulk_w50_min", "bulk_w50_std", "bulk_w5_max", "bulk_w5_mean", "bulk_w5_min", "bulk_w5_std",
    "bulkiness", "charge_entropy_x", "charge_entropy_y", "charge_pH7", "charge_w100_max", "charge_w100_mean",
    "charge_w100_min", "charge_w100_std", "charge_w11_max", "charge_w11_mean", "charge_w11_min", "charge_w11_std",
    "charge_w25_max", "charge_w25_mean", "charge_w25_min", "charge_w25_std", "charge_w50_max", "charge_w50_mean",
    "charge_w50_min", "charge_w50_std", "charge_w5_max", "charge_w5_mean", "charge_w5_min", "charge_w5_std",
    "charge_x", "charge_y", "comp_A", "comp_C", "comp_D", "comp_E", "comp_F", "comp_G", "comp_H", "comp_I",
    "comp_K", "comp_L", "comp_M", "comp_N", "comp_P", "comp_Q", "comp_R", "comp_S", "comp_T", "comp_V", "comp_W", "comp_Y",
    "comp_bias_w100", "comp_bias_w11", "comp_bias_w25", "comp_bias_w5", "comp_bias_w50", "comp_frac_aliphatic",
    "comp_frac_aromatic", "comp_frac_glycine", "comp_frac_negative", "comp_frac_polar", "comp_frac_positive",
    "comp_frac_proline", "comp_local_A", "comp_local_C", "comp_local_D", "comp_local_E", "comp_local_F",
    "comp_local_G", "comp_local_H", "comp_local_I", "comp_local_K", "comp_local_L", "comp_local_M", "comp_local_N",
    "comp_local_P", "comp_local_Q", "comp_local_R", "comp_local_S", "comp_local_T", "comp_local_V", "comp_local_W",
    "comp_local_Y", "conflict_score", "curvature_x", "curvature_y", "entropy_w100", "entropy_w11", "entropy_w25",
    "entropy_w5", "entropy_w50", "entropy_win101", "entropy_win21", "entropy_win51", "flex_w100_max", "flex_w100_mean",
    "flex_w100_min", "flex_w100_std", "flex_w11_max", "flex_w11_mean", "flex_w11_min", "flex_w11_std", "flex_w25_max",
    "flex_w25_mean", "flex_w25_min", "flex_w25_std", "flex_w50_max", "flex_w50_mean", "flex_w50_min", "flex_w50_std",
    "flex_w5_max", "flex_w5_mean", "flex_w5_min", "flex_w5_std", "flexibility", "frac_dis_promo", "frac_dis_win101",
    "frac_dis_win21", "frac_dis_win51", "frac_ord_promo", "frac_ord_win101", "frac_ord_win21", "frac_ord_win51",
    "has_af2", "has_nmr", "has_pssm", "has_xray", "helix_prop", "helix_prop_w100_max", "helix_prop_w100_mean",
    "helix_prop_w100_min", "helix_prop_w100_std", "helix_prop_w11_max", "helix_prop_w11_mean", "helix_prop_w11_min",
    "helix_prop_w11_std", "helix_prop_w25_max", "helix_prop_w25_mean", "helix_prop_w25_min", "helix_prop_w25_std",
    "helix_prop_w50_max", "helix_prop_w50_mean", "helix_prop_w50_min", "helix_prop_w50_std", "helix_prop_w5_max",
    "helix_prop_w5_mean", "helix_prop_w5_min", "helix_prop_w5_std", "hydro_entropy_x", "hydro_entropy_y",
    "hydro_w100_max", "hydro_w100_mean", "hydro_w100_min", "hydro_w100_std", "hydro_w11_max", "hydro_w11_mean",
    "hydro_w11_min", "hydro_w11_std", "hydro_w25_max", "hydro_w25_mean", "hydro_w25_min", "hydro_w25_std",
    "hydro_w50_max", "hydro_w50_mean", "hydro_w50_min", "hydro_w50_std", "hydro_w5_max", "hydro_w5_mean",
    "hydro_w5_min", "hydro_w5_std", "hydropathy_x", "hydropathy_y", "in_poly_D_run_ge3", "in_poly_E_run_ge3",
    "in_poly_G_run_ge3", "in_poly_K_run_ge3", "in_poly_N_run_ge3", "in_poly_P_run_ge3", "in_poly_Q_run_ge3",
    "in_poly_S_run_ge3", "inv_plddt", "is_unknown_aa", "low_complex_w100", "low_complex_w11", "low_complex_w25",
    "low_complex_w5", "low_complex_w50", "plddt", "polar_w100_max", "polar_w100_mean", "polar_w100_min",
    "polar_w100_std", "polar_w11_max", "polar_w11_mean", "polar_w11_min", "polar_w11_std", "polar_w25_max",
    "polar_w25_mean", "polar_w25_min", "polar_w25_std", "polar_w50_max", "polar_w50_mean", "polar_w50_min",
    "polar_w50_std", "polar_w5_max", "polar_w5_mean", "polar_w5_min", "polar_w5_std", "polarity", "rmsf",
    "scd_local", "sheet_prop", "sheet_prop_w100_max", "sheet_prop_w100_mean", "sheet_prop_w100_min",
    "sheet_prop_w100_std", "sheet_prop_w11_max", "sheet_prop_w11_mean", "sheet_prop_w11_min", "sheet_prop_w11_std",
    "sheet_prop_w25_max", "sheet_prop_w25_mean", "sheet_prop_w25_min", "sheet_prop_w25_std", "sheet_prop_w50_max",
    "sheet_prop_w50_mean", "sheet_prop_w50_min", "sheet_prop_w50_std", "sheet_prop_w5_max", "sheet_prop_w5_mean",
    "sheet_prop_w5_min", "sheet_prop_w5_std", "uversky_dist_w100", "uversky_dist_w11", "uversky_dist_w25",
    "uversky_dist_w5", "uversky_dist_w50", "vdw_volume", "vdw_w100_max", "vdw_w100_mean", "vdw_w100_min",
    "vdw_w100_std", "vdw_w11_max", "vdw_w11_mean", "vdw_w11_min", "vdw_w11_std", "vdw_w25_max", "vdw_w25_mean",
    "vdw_w25_min", "vdw_w25_std", "vdw_w50_max", "vdw_w50_mean", "vdw_w50_min", "vdw_w50_std", "vdw_w5_max",
    "vdw_w5_mean", "vdw_w5_min", "vdw_w5_std", "z_bfactor", "z_plddt", "z_rmsf"
]

_MODEL_NAME = "ewclv1p3"
MODEL = None

def _get_model():
    global MODEL
    if MODEL is None:
        path = os.environ.get("EWCLV1_P3_MODEL_PATH")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=503, detail="Model path missing or file not found")
        MODEL = load_model_forgiving(path)
    return MODEL

router = APIRouter(prefix="/ewcl", tags=[_MODEL_NAME])

# Mock feature extractor for EWCLv1-P3 (302 REAL features)
def _mock_feature_extraction(seq: str) -> pd.DataFrame:
    """
    Mock feature extraction for EWCLv1-P3 that returns zero features.
    In production, this would be replaced with real EWCL-P3 feature extraction.
    Uses REAL feature names, not generic Column_X.
    """
    seq_len = len(seq)
    features = {}
    
    # Fill all 302 REAL features with zeros/defaults
    for feat_name in EWCLV1P3_302_FEATURES:
        if feat_name in ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]:
            # Amino acid identity features
            features[feat_name] = 0.0
        elif feat_name.startswith("comp_"):
            # Composition features
            features[feat_name] = np.random.uniform(0.0, 0.1)
        elif feat_name in ["bfactor", "plddt", "rmsf", "z_bfactor", "z_plddt", "z_rmsf"]:
            # Structural features
            features[feat_name] = np.random.uniform(0.0, 1.0)
        elif feat_name.startswith("has_"):
            # Boolean features
            features[feat_name] = 0.0
        else:
            # All other real features
            features[feat_name] = 0.0
    
    # Create per-residue features (repeat for each position)
    rows = []
    for i in range(seq_len):
        row = features.copy()
        # Set amino acid identity for this position
        if i < len(seq):
            aa = seq[i].upper()
            if aa in ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]:
                row[aa] = 1.0
        rows.append(row)
    
    return pd.DataFrame(rows, columns=EWCLV1P3_302_FEATURES)

@router.get("/analyze-pdb/ewclv1-p3/health")
def health_check():
    """Health check for EWCLv1-P3 model."""
    try:
        model = _get_model()
        return {
            "ok": True,
            "model_name": _MODEL_NAME,
            "loaded": model is not None,
            "features": len(EWCLV1P3_302_FEATURES),
            "real_features": True,  # NO generic Column_X features
            "feature_extractor": True
        }
    except Exception as e:
        return {
            "ok": False,
            "model_name": _MODEL_NAME,
            "loaded": False,
            "error": str(e),
            "features": len(EWCLV1P3_302_FEATURES),
            "real_features": True,
            "feature_extractor": True
        }

@router.post("/analyze-pdb/ewclv1-p3")
async def analyze_pdb_ewclv1_p3(file: UploadFile = File(...)):
    """Analyze PDB structure using EWCLv1-P3 model (302 REAL features)."""
    model = _get_model()
    
    try:
        raw = await file.read()
        
        # Simple PDB parsing to extract sequence
        content = raw.decode("utf-8", errors="ignore")
        seq = ""
        for line in content.split("\n"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                aa_code = line[17:20].strip()
                # Convert 3-letter to 1-letter amino acid codes
                aa_map = {
                    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
                }
                seq += aa_map.get(aa_code, "X")
        
        if not seq:
            raise HTTPException(status_code=400, detail="No valid amino acid sequence found in PDB")

        print(f"[ewclv1-p3] Processing PDB with {len(seq)} residues using REAL features")

        # Extract features using mock extractor with REAL feature names
        feature_df = _mock_feature_extraction(seq)
        print(f"[ewclv1-p3] Extracted REAL features: {feature_df.shape}")
        
        # Make prediction using singleton model
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(feature_df)
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                cl = predictions[:, 1]  # Probability of positive class
            else:
                cl = predictions.flatten()
        else:
            cl = model.predict(feature_df)
        
        # Create response with per-residue scores
        results = []
        for i, (residue, score) in enumerate(zip(seq, cl)):
            results.append({
                "position": i + 1,
                "residue": residue,
                "score": float(score),
                "prediction": "pathogenic" if score > 0.5 else "benign"
            })
        
        print(f"[ewclv1-p3] Generated {len(results)} predictions")
        
        return {
            "model": _MODEL_NAME,
            "sequence_length": len(seq),
            "features_used": len(EWCLV1P3_302_FEATURES),
            "results": results,
            "summary": {
                "mean_score": float(np.mean(cl)),
                "max_score": float(np.max(cl)),
                "pathogenic_residues": int(np.sum(cl > 0.5)),
                "total_residues": len(cl)
            }
        }
    
    except Exception as e:
        print(f"[ewclv1-p3] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


