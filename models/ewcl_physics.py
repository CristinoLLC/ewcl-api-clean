"""
Physics-based EWCL model - Enhanced with proper entropy calculations
No dependency on B-factor or pLDDT for CL computation
"""

import json, math, warnings
from typing import List, Dict
from Bio.PDB import PDBParser, is_aa
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import entropy

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
HYDRO_SCORES = {  # Kyte–Doolittle (normalised 0-1)
    "I": 1.00, "V": 0.86, "L": 0.82, "F": 0.80, "C": 0.77,
    "M": 0.74, "A": 0.62, "G": 0.48, "T": 0.40, "S": 0.38,
    "W": 0.37, "Y": 0.32, "P": 0.27, "H": 0.23, "E": 0.07,
    "Q": 0.04, "D": 0.02, "N": 0.00, "K": 0.00, "R": 0.00
}

CHARGE = {               # +1 / 0 / -1 at pH 7
    "K":  1, "R":  1, "H":  0.1,
    "D": -1, "E": -1
}

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def _norm(x: np.ndarray) -> np.ndarray:
    """0-1 min-max normalization (gracefully handles flat arrays)"""
    x = x.astype(float)
    ptp = np.ptp(x)
    return (x - x.min()) / (ptp + 1e-9)

def sliding_entropy(arr: np.ndarray, win=5) -> np.ndarray:
    """Shannon entropy over a sliding window"""
    out = np.zeros_like(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:  # Handle constant arrays
        return out
    
    for i in range(len(arr)):
        s = max(0, i - win//2)
        e = min(len(arr), i + win//2 + 1)
        window_data = arr[s:e]
        hist, _ = np.histogram(window_data, bins=10, range=(lo, hi), density=True)
        out[i] = entropy(hist + 1e-9)
    return out

def compute_curvature_features(ca_coords: np.ndarray) -> np.ndarray:
    """Compute curvature from CA coordinates"""
    if len(ca_coords) < 3:
        return np.zeros(len(ca_coords))
    
    # Second derivative approximation for curvature
    curv = np.zeros(len(ca_coords))
    for i in range(1, len(ca_coords) - 1):
        v1 = ca_coords[i] - ca_coords[i-1]
        v2 = ca_coords[i+1] - ca_coords[i]
        
        # Angle between consecutive segments
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1, 1)
            curv[i] = np.arccos(cos_angle)
    
    # Handle endpoints
    curv[0] = curv[1] if len(curv) > 1 else 0
    curv[-1] = curv[-2] if len(curv) > 1 else 0
    
    return curv

# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def compute_ewcl_from_pdb(pdb_path: str) -> List[Dict]:
    """
    Enhanced physics-based EWCL computation
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)

    # Gather residue data
    residues = []
    ca_coords = []
    
    for res in structure.get_residues():
        if not is_aa(res, standard=True) or "CA" not in res:
            continue
        residues.append(res)
        ca_coords.append(res["CA"].get_vector().get_array())

    if len(residues) == 0:
        return []

    ca_coords = np.array(ca_coords)

    # Extract sequence properties
    hydro = np.array([HYDRO_SCORES.get(res.get_resname()[0], 0.0) for res in residues])
    charge = np.array([CHARGE.get(res.get_resname()[0], 0) for res in residues])
    
    # Extract B-factors separately (for output only, not CL computation)
    bfactors = np.array([res["CA"].get_bfactor() for res in residues])

    # Physics-based feature calculations
    hydro_entropy = sliding_entropy(hydro, win=5)
    charge_entropy = sliding_entropy(charge, win=5)
    backbone_curvature = compute_curvature_features(ca_coords)

    # Normalize features
    hydro_ent_norm = _norm(hydro_entropy)
    charge_ent_norm = _norm(charge_entropy)
    curv_norm = _norm(backbone_curvature)

    # Physics-based collapse likelihood (independent of B-factor/pLDDT)
    cl = (0.4 * hydro_ent_norm + 
          0.4 * charge_ent_norm + 
          0.2 * curv_norm)

    # Build output
    results = []
    for idx, res in enumerate(residues):
        chain = res.get_parent().id
        res_id = res.id[1]
        b_factor_val = float(bfactors[idx])
        
        results.append({
            "chain": chain,
            "residue_id": int(res_id),
            "aa": res.get_resname()[0],
            "b_factor": round(b_factor_val, 2),
            "plddt": round(b_factor_val, 2),  # Copy B-factor as pLDDT for AlphaFold
            "entropy_hydropathy": round(float(hydro_entropy[idx]), 3),
            "entropy_charge": round(float(charge_entropy[idx]), 3),
            "curvature": round(float(backbone_curvature[idx]), 3),
            "cl": round(float(cl[idx]), 3)
        })
    
    return results
