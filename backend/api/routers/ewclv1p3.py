from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Tuple, Union, Any
import os, io, joblib, numpy as np, pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB import PDBParser
import sys
from pathlib import Path
import tempfile
import json

# --- Add backend_bundle and models to path to import parsers ---
_BACKEND_BUNDLE_PATH = str(Path(__file__).resolve().parents[3] / "backend_bundle")
if _BACKEND_BUNDLE_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_BUNDLE_PATH)

_MODELS_PATH = str(Path(__file__).resolve().parents[3] / "models")
if _MODELS_PATH not in sys.path:
    sys.path.insert(0, _MODELS_PATH)

try:
    from meta.ewcl_feature_extractor_v2 import EWCLFeatureExtractor
    _FEATURE_EXTRACTOR = EWCLFeatureExtractor()
    from enhanced_ewcl_af import compute_curvature_features
    print(f"[ewclv1-p3] Successfully imported parsers")
except ImportError as e:
    print(f"[ewclv1-p3] Could not import parsers: {e}")
    _FEATURE_EXTRACTOR = None
    compute_curvature_features = None

# --- Self-contained feature computation fallbacks ---
_AA_HYDRO = { # Kyte-Doolittle approximate
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
_AA_CHARGE = { # net charge at ~pH7 (simplified)
    'D':-1,'E':-1,'K':+1,'R':+1,'H':+0.1
}

def _curvature_from_ca(coords):
    """
    coords: (N,3) numpy array of CA coords (one chain, residue-ordered)
    returns: curvature (N,), flips (N,) boolean of turning sign changes
    """
    N = len(coords)
    kappa = np.zeros(N, dtype=float)
    flips = np.zeros(N, dtype=float)
    if N < 3:
        return kappa, flips
    v_prev = coords[1] - coords[0]
    v_prev /= (np.linalg.norm(v_prev) + 1e-9)
    for i in range(1, N-1):
        v = coords[i+1] - coords[i]
        v /= (np.linalg.norm(v) + 1e-9)
        cosang = float(np.clip(np.dot(v_prev, v), -1.0, 1.0))
        kappa[i] = 1.0 - cosang               # 0 (straight) → 2 (U-turn), typically 0..~1
        # flip if signed turn changes direction (use 2D projection sign heuristic)
        # robust proxy: sign change in finite difference of curvature
        flips[i] = 1.0 if (i >= 2 and (kappa[i]-kappa[i-1])*(kappa[i-1]-kappa[i-2]) < 0) else 0.0
        v_prev = v
    return kappa, flips

def _window_var_norm(arr, win=11):
    n = len(arr); out = np.zeros(n, float); h = win//2
    for i in range(n):
        s=max(0,i-h); e=min(n,i+h+1)
        seg = arr[s:e]
        out[i] = float(np.var(seg)) if seg.size else 0.0
    if out.max() > out.min():
        out = (out - out.min()) / (out.max()-out.min())
    return out

def _entropy_proxies(seq, win=11):
    hyd = np.array([_AA_HYDRO.get(a,0.0) for a in seq], float)
    chg = np.array([_AA_CHARGE.get(a,0.0) for a in seq], float)
    return _window_var_norm(hyd, win=win), _window_var_norm(chg, win=win)

# --- Load model once ---
_MODEL = None
_MODEL_NAME = "ewclv1-p3"
_FEATURE_ORDER = []

def _load_model():
    global _MODEL, _FEATURE_ORDER
    if _MODEL is not None:
        return
    
    model_path = os.environ.get("EWCLV1_P3_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"EWCLv1-P3 model not found. Set EWCLV1_P3_MODEL_PATH (got: {model_path})")
    
    print(f"[ewclv1-p3] Loading model from {model_path}")
    _MODEL = joblib.load(model_path)
    
    # Load the EXACT feature order from the JSON file that the model was trained on
    features_json_path = Path(_BACKEND_BUNDLE_PATH) / "meta" / "EWCLv1-P3_features.json"
    if features_json_path.exists():
        with open(features_json_path) as f:
            _FEATURE_ORDER = json.load(f)
        print(f"[ewclv1-p3] Loaded {len(_FEATURE_ORDER)} features from training JSON")
    else:
        raise RuntimeError(f"EWCLv1-P3 features JSON not found at {features_json_path}")

def _extract_curvature_features(pdb_bytes: bytes, pdb_id: str) -> pd.DataFrame:
    """Extract curvature features using the enhanced_ewcl_af extractor."""
    if compute_curvature_features is None:
        return pd.DataFrame()
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".pdb") as tmp:
            tmp.write(pdb_bytes.decode("utf-8"))
            tmp_path = tmp.name
        
        try:
            # Get the raw curvature output
            curv_obj = compute_curvature_features(tmp_path)
            print(f"[ewclv1-p3] Raw curvature object type: {type(curv_obj)}")
            
            if isinstance(curv_obj, dict):
                residues = curv_obj.get("residues", [])
                if residues:
                    df = pd.DataFrame(residues)
                    print(f"[ewclv1-p3] Curvature DataFrame columns: {list(df.columns)}")
                    
                    # Only use the columns that actually exist in the output
                    # Map position to residue_index for consistency
                    if "position" in df.columns:
                        df["residue_index"] = df["position"]
                    
                    return df
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"[ewclv1-p3] Curvature extraction failed: {e}")
        
    return pd.DataFrame()

def _parse_pdb_features(pdb_bytes: bytes, pdb_id: str) -> Tuple[pd.DataFrame, str, str]:
    """Parse PDB file and extract features using robust self-contained methods."""
    pdb_str = pdb_bytes.decode("utf-8")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_str))
    
    is_alphafold = "ALPHAFOLD" in pdb_str.upper()
    
    # Extract basic sequence and CA coordinates from PDB
    residues_data = []
    seq_list = []
    ca_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Standard residue
                    res_name = residue.get_resname()
                    res_seq = residue.get_id()[1]
                    
                    # Get CA coordinates
                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                        ca_coords.append(ca_coord)
                    else:
                        # If no CA, use average of all atoms
                        coords = [atom.get_coord() for atom in residue.get_atoms()]
                        ca_coord = np.mean(coords, axis=0) if coords else np.array([0.0, 0.0, 0.0])
                        ca_coords.append(ca_coord)
                    
                    bfactor = np.mean([atom.get_bfactor() for atom in residue.get_atoms()])
                    
                    # Convert 3-letter to 1-letter amino acid codes
                    aa_map = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    aa_1letter = aa_map.get(res_name, 'X')
                    
                    residues_data.append({
                        "residue_index": res_seq,
                        "aa": aa_1letter,
                        "bfactor": bfactor,
                        "plddt": bfactor if is_alphafold else 0.0,
                        "inv_plddt": (100.0 - bfactor) / 100.0 if is_alphafold else 0.0,
                        "has_af2": 1.0 if is_alphafold else 0.0,
                        "has_xray": 0.0 if is_alphafold else 1.0,
                        "has_nmr": 0.0,
                        "has_pssm": 0.0,
                        "ca_coord": ca_coord
                    })
                    seq_list.append(aa_1letter)
            
            # Process first chain with residues and break
            if seq_list:
                break
        break  # Process first model

    if not residues_data:
        raise ValueError("No standard residues found in PDB file.")

    sequence = "".join(seq_list)
    df = pd.DataFrame(residues_data)
    print(f"[ewclv1-p3] Parsed {len(seq_list)} residues from PDB")
    
    # Compute self-contained curvature and entropy features
    coords = np.stack([row['ca_coord'] for row in residues_data], axis=0)
    kappa, flips = _curvature_from_ca(coords)
    hyd_e, chg_e = _entropy_proxies(sequence, win=11)
    
    df['curvature'] = kappa
    df['flips'] = flips
    df['hydro_entropy'] = hyd_e
    df['charge_entropy'] = chg_e
    
    print(f"[ewclv1-p3] Computed self-contained features - curvature: {np.sum(kappa > 0)} non-zero, hydro_entropy: {np.sum(hyd_e > 0)} non-zero")
    
    # Extract sequence features using the REAL feature extractor if available
    if _FEATURE_EXTRACTOR:
        seq_df = _FEATURE_EXTRACTOR.extract_sequence_features(sequence, pdb_id)
        print(f"[ewclv1-p3] Extracted {seq_df.shape[1]} sequence features")
        
        # Merge with structural features
        merged_df = pd.merge(seq_df, df, on=["residue_index", "aa"], how="left", suffixes=('', '_struct'))
    else:
        # Fallback: use structural features only
        merged_df = df.copy()
        print(f"[ewclv1-p3] Using structural features only (no sequence extractor)")
    
    # If external curvature module exists and returns columns, prefer merging
    try:
        ext_df = _extract_curvature_features(pdb_bytes, pdb_id)
        if not ext_df.empty and "residue_index" in ext_df.columns:
            # Prefer non-null external values for curvature/entropy
            for c in ['curvature','hydro_entropy','charge_entropy','flips']:
                if c in ext_df.columns:
                    merged_df[c] = np.where(pd.notna(ext_df[c]), ext_df[c], merged_df[c])
            print(f"[ewclv1-p3] Merged external curvature features")
    except Exception as e:
        print(f"[warn] curvature module merge skipped: {e}")
    
    # Calculate Z-scores for structural features
    if "plddt" in merged_df.columns:
        merged_df["z_plddt"] = (merged_df["plddt"] - merged_df["plddt"].mean()) / (merged_df["plddt"].std() + 1e-9)
    if "bfactor" in merged_df.columns:
        merged_df["z_bfactor"] = (merged_df["bfactor"] - merged_df["bfactor"].mean()) / (merged_df["bfactor"].std() + 1e-9)
        # RMSF approximation
        merged_df["rmsf"] = merged_df["bfactor"] / 100.0
        merged_df["z_rmsf"] = (merged_df["rmsf"] - merged_df["rmsf"].mean()) / (merged_df["rmsf"].std() + 1e-9)
    
    # Add conflict score (placeholder)
    merged_df["conflict_score"] = 0.0
    
    # ONLY use features that are in the training set - fill missing ones with 0.0
    for feat in _FEATURE_ORDER:
        if feat not in merged_df.columns:
            merged_df[feat] = 0.0
    
    # Fill any NaN values
    merged_df = merged_df.fillna(0.0)
    
    protein_name = structure.header.get("name", pdb_id) or pdb_id
    print(f"[ewclv1-p3] Final feature matrix: {merged_df.shape}")
    
    # Debug: Show which key features have non-zero values
    key_features = ["curvature", "hydro_entropy", "charge_entropy", "flips", "bfactor", "plddt"]
    for feat in key_features:
        if feat in merged_df.columns:
            non_zero = (merged_df[feat] != 0.0).sum()
            print(f"[ewclv1-p3] '{feat}': {non_zero}/{len(merged_df)} non-zero values")
    
    return merged_df, sequence, protein_name

router = APIRouter(prefix="/ewcl", tags=[_MODEL_NAME])

@router.post("/analyze-pdb/ewclv1-p3")
async def analyze_pdb_ewclv1_p3(file: UploadFile = File(...)):
    try:
        if _FEATURE_EXTRACTOR is None:
            raise HTTPException(status_code=503, detail="Feature extractor not available")
        
        _load_model()
        raw = await file.read()
        
        # Parse PDB and extract features using ONLY real extractors
        feature_df, seq, pdb_name = _parse_pdb_features(raw, file.filename or "input.pdb")
        
        # Select features in EXACT training order from JSON
        X = feature_df[_FEATURE_ORDER].copy()
        print(f"[ewclv1-p3] Model input shape: {X.shape}")

        # Predict disorder
        if hasattr(_MODEL, "predict_proba"):
            y = _MODEL.predict_proba(X)
            cl = y[:, 1] if y.ndim == 2 and y.shape[1] > 1 else y.ravel()
        else:
            z = _MODEL.predict(X)
            cl = np.clip(z, 0.0, 1.0)

        # Calculate confidence
        conf = 1.0 - np.abs(cl - 0.5) * 2.0
        conf = np.clip(conf, 0.0, 1.0)

        # Build response with real extracted features
        residues = []
        for i in range(len(feature_df)):
            row = feature_df.iloc[i]
            
            # Map to the correct column names from the actual DataFrame
            curvature_val = None
            for col in ['curvature', 'bfactor_curv', 'curvature_x']:
                if col in row and pd.notna(row[col]):
                    curvature_val = float(row[col])
                    break
            
            hydro_entropy_val = None
            for col in ['hydro_entropy', 'bfactor_curv_entropy', 'hydro_entropy_x']:
                if col in row and pd.notna(row[col]):
                    hydro_entropy_val = float(row[col])
                    break
                    
            charge_entropy_val = None
            for col in ['charge_entropy', 'charge_entropy_x']:
                if col in row and pd.notna(row[col]):
                    charge_entropy_val = float(row[col])
                    break
                    
            flips_val = None
            for col in ['flips', 'bfactor_curv_flips', 'charge_entropy_y']:
                if col in row and pd.notna(row[col]):
                    flips_val = float(row[col])
                    break
            
            residues.append({
                "residue_index": int(row["residue_index"]),
                "aa": row["aa"],
                "cl": float(cl[i]),
                "confidence": float(conf[i]),
                # Use actual extracted features with robust column mapping
                "plddt": float(row.get("plddt", 0.0)) if pd.notna(row.get("plddt")) else None,
                "bfactor": float(row.get("bfactor", 0.0)) if pd.notna(row.get("bfactor")) else None,
                "curvature": curvature_val,
                "hydro_entropy": hydro_entropy_val,
                "charge_entropy": charge_entropy_val,
                "flips": flips_val if flips_val is not None else 0.0,
            })

        print(f"[ewclv1-p3] Successfully processed {len(residues)} residues")
        return JSONResponse(content={
            "id": pdb_name, 
            "model": _MODEL_NAME, 
            "length": len(seq), 
            "residues": residues
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ewclv1-p3] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{_MODEL_NAME} PDB analysis failed: {e}")

@router.get("/analyze-pdb/ewclv1-p3/health")
def health_check():
    try:
        _load_model()
        return {
            "ok": True, 
            "model_name": _MODEL_NAME, 
            "loaded": _MODEL is not None,
            "features": len(_FEATURE_ORDER),
            "feature_extractor": _FEATURE_EXTRACTOR is not None,
            "curvature_extractor": compute_curvature_features is not None
        }
    except Exception as e:
        return {"ok": False, "model_name": _MODEL_NAME, "loaded": False, "error": str(e)}


