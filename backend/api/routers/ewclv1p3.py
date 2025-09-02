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

# --- Add models to path for enhanced_ewcl_af if available ---
_MODELS_PATH = str(Path(__file__).resolve().parents[3] / "models")
if _MODELS_PATH not in sys.path:
    sys.path.insert(0, _MODELS_PATH)

# Try to import enhanced curvature features (optional)
try:
    from enhanced_ewcl_af import compute_curvature_features
    print(f"[ewclv1-p3] Successfully imported enhanced_ewcl_af")
except ImportError as e:
    print(f"[ewclv1-p3] enhanced_ewcl_af not available: {e}")
    compute_curvature_features = None

# --- Self-contained feature computation ---
_AA_HYDRO = { # Kyte-Doolittle hydropathy scale
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
_AA_CHARGE = { # net charge at ~pH7
    'D':-1,'E':-1,'K':+1,'R':+1,'H':+0.1
}

def _curvature_from_ca(coords):
    """Self-contained curvature calculation from CA coordinates."""
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
        kappa[i] = 1.0 - cosang  # 0 (straight) → 2 (U-turn)
        
        # Detect direction changes (flips)
        if i >= 2 and (kappa[i]-kappa[i-1])*(kappa[i-1]-kappa[i-2]) < 0:
            flips[i] = 1.0
        v_prev = v
    return kappa, flips

def _window_variance_normalized(arr, win=11):
    """Compute windowed variance and normalize."""
    n = len(arr)
    out = np.zeros(n, float)
    h = win//2
    
    for i in range(n):
        s = max(0, i-h)
        e = min(n, i+h+1)
        seg = arr[s:e]
        out[i] = float(np.var(seg)) if seg.size else 0.0
    
    # Normalize to [0,1]
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out

def _entropy_proxies(seq, win=11):
    """Compute hydropathy and charge entropy proxies."""
    hyd = np.array([_AA_HYDRO.get(a, 0.0) for a in seq], float)
    chg = np.array([_AA_CHARGE.get(a, 0.0) for a in seq], float)
    return _window_variance_normalized(hyd, win), _window_variance_normalized(chg, win)

# --- Model loading ---
_MODEL = None
_MODEL_NAME = "ewclv1-p3"

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return
    
    model_path = os.environ.get("EWCLV1_P3_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"EWCLv1-P3 model not found. Set EWCLV1_P3_MODEL_PATH (got: {model_path})")
    
    print(f"[ewclv1-p3] Loading model from {model_path}")
    _MODEL = joblib.load(model_path)

def _parse_pdb_features(pdb_bytes: bytes, pdb_id: str) -> Tuple[pd.DataFrame, str, str]:
    """Parse PDB file and extract features using self-contained methods."""
    pdb_str = pdb_bytes.decode("utf-8")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_str))
    
    is_alphafold = "ALPHAFOLD" in pdb_str.upper()
    
    # Extract sequence and coordinates
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
                        # Fallback to average of all atoms
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
    
    # Compute self-contained structural features
    coords = np.stack([row['ca_coord'] for row in residues_data], axis=0)
    kappa, flips = _curvature_from_ca(coords)
    hyd_entropy, chg_entropy = _entropy_proxies(sequence, win=11)
    
    df['curvature'] = kappa
    df['flips'] = flips
    df['hydro_entropy'] = hyd_entropy
    df['charge_entropy'] = chg_entropy
    
    # Additional features
    df['inv_plddt'] = (100.0 - df['plddt']) / 100.0 if is_alphafold else 0.0
    df['has_af2'] = 1.0 if is_alphafold else 0.0
    df['has_xray'] = 0.0 if is_alphafold else 1.0
    df['z_bfactor'] = (df['bfactor'] - df['bfactor'].mean()) / (df['bfactor'].std() + 1e-9)
    df['z_plddt'] = (df['plddt'] - df['plddt'].mean()) / (df['plddt'].std() + 1e-9)
    df['rmsf'] = df['bfactor'] / 100.0
    df['z_rmsf'] = (df['rmsf'] - df['rmsf'].mean()) / (df['rmsf'].std() + 1e-9)
    df['conflict_score'] = 0.0
    
    # Fill any NaN values
    df = df.fillna(0.0)
    
    protein_name = structure.header.get("name", pdb_id) or pdb_id
    print(f"[ewclv1-p3] Computed self-contained features: curvature, hydro_entropy, charge_entropy, flips")
    
    return df, sequence, protein_name

router = APIRouter(prefix="/ewcl", tags=[_MODEL_NAME])

@router.post("/analyze-pdb/ewclv1-p3")
async def analyze_pdb_ewclv1_p3(file: UploadFile = File(...)):
    try:
        _load_model()
        raw = await file.read()
        
        # Parse PDB and extract features
        feature_df, seq, pdb_name = _parse_pdb_features(raw, file.filename or "input.pdb")
        
        # Use a basic feature set that works with most models
        basic_features = ["bfactor", "curvature", "hydro_entropy", "charge_entropy", "flips", 
                         "z_bfactor", "z_plddt", "rmsf", "z_rmsf", "plddt", "inv_plddt",
                         "has_af2", "has_xray", "conflict_score"]
        
        # Only use features that exist in the DataFrame
        available_features = [f for f in basic_features if f in feature_df.columns]
        print(f"[ewclv1-p3] Using {len(available_features)} features: {available_features}")
        
        X = feature_df[available_features].copy()

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

        # Build response
        residues = []
        for i in range(len(feature_df)):
            row = feature_df.iloc[i]
            residues.append({
                "residue_index": int(row["residue_index"]),
                "aa": row["aa"],
                "cl": float(cl[i]),
                "confidence": float(conf[i]),
                "plddt": float(row.get("plddt", 0.0)),
                "bfactor": float(row.get("bfactor", 0.0)),
                "curvature": float(row.get("curvature", 0.0)),
                "hydro_entropy": float(row.get("hydro_entropy", 0.0)),
                "charge_entropy": float(row.get("charge_entropy", 0.0)),
                "flips": float(row.get("flips", 0.0)),
            })

        print(f"[ewclv1-p3] Successfully processed {len(residues)} residues")
        return JSONResponse(content={
            "id": pdb_name, 
            "model": _MODEL_NAME, 
            "length": len(seq), 
            "residues": residues
        })
        
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
            "self_contained": True
        }
    except Exception as e:
        return {"ok": False, "model_name": _MODEL_NAME, "loaded": False, "error": str(e)}


