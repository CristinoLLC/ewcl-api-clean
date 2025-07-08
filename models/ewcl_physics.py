"""
Physics-based EWCL model - Enhanced AlphaFold-aware version
Matches the local enhanced_ewcl_af.py implementation
"""

import os, json, re
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import entropy
from Bio.PDB import PDBParser, is_aa
from typing import List, Dict

# ──────────────── helper tables ────────────────
HYDROPATHY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}
CHARGE = {aa: 0 for aa in HYDROPATHY}
CHARGE.update({'ASP': -1, 'GLU': -1, 'ARG': 1, 'LYS': 1, 'HIS': 0.5})

# ──────────────── math helpers ────────────────
def _norm(x):
    """0-1 min-max normalization"""
    x = x.astype(float)
    ptp = np.ptp(x)
    return (x - x.min()) / (ptp + 1e-9)

def curvature(arr):
    """Second derivative approximation"""
    return np.gradient(np.gradient(arr.astype(float)))

def sliding_entropy(arr, win=5):
    """Shannon entropy over a sliding window"""
    out = np.zeros_like(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    for i in range(len(arr)):
        s, e = max(0, i - win // 2), min(len(arr), i + win // 2 + 1)
        hist, _ = np.histogram(arr[s:e], bins=10, range=(lo, hi), density=True)
        out[i] = entropy(hist + 1e-9)
    return out

def sign_flip_ratio(arr, win=5):
    """Fraction of sign flips in first derivative inside window"""
    flips = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        s, e = max(0, i - win // 2), min(len(arr), i + win // 2 + 1)
        diff_sign = np.sign(np.diff(arr[s:e]))
        flips[i] = np.sum(np.diff(diff_sign) != 0) / max(1, len(diff_sign) - 1)
    return flips

# ──────────────── main model ────────────────
def compute_ewcl_from_pdb(pdb_path: str, win_ent=5, win_curv=5) -> List[Dict]:
    """
    Enhanced physics-based EWCL computation matching local model
    """
    # Detect AlphaFold by checking header
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as fh:
        first_60 = ''.join([fh.readline() for _ in range(60)])
    is_af = bool(re.search(r'ALPHAFOLD', first_60, re.I)) \
            or os.path.basename(pdb_path).startswith("AF-")

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("X", pdb_path)[0]

    # Extract per-residue data
    bfac, aa, plddt = [], [], []
    for res in model.get_residues():
        if "CA" not in res:
            continue
        beta = res["CA"].get_bfactor()
        bfac.append(beta)
        plddt.append(beta if is_af else np.nan)
        aa.append(res.get_resname())

    if len(bfac) == 0:
        return []

    bfac = np.array(bfac)
    plddt = np.array(plddt)

    # Core features
    bfac_norm = _norm(bfac)
    hydro = np.array([HYDROPATHY.get(a, 0) for a in aa])
    charge = np.array([CHARGE.get(a, 0) for a in aa])
    
    # Entropy calculations
    hydro_ent = sliding_entropy(hydro, win_ent)
    charge_ent = sliding_entropy(charge, win_ent)
    hydro_ent_n = _norm(hydro_ent)
    charge_ent_n = _norm(charge_ent)

    # Curvature calculations with safety checks
    win_curv_adj = max(3, win_curv | 1)
    win_curv_adj = min(win_curv_adj, len(bfac) - (1 - len(bfac) % 2))

    curv_raw = curvature(bfac)
    curv_savgol = curvature(savgol_filter(bfac, win_curv_adj, 2))
    curv_mean = curvature(uniform_filter1d(bfac, size=win_curv_adj))
    curv_median = curvature(uniform_filter1d(bfac, size=win_curv_adj, mode='nearest'))
    curv_ent = sliding_entropy(curv_raw, win_ent)
    curv_flips = sign_flip_ratio(curv_raw, win_ent)
    curv_clip = np.clip((curv_raw - curv_raw.mean()) / (curv_raw.std() + 1e-6), -2, 2)

    # Final collapse likelihood calculation (matching local model)
    cl = (0.35 * hydro_ent_n + 0.35 * charge_ent_n + 0.30 * bfac_norm)

    # Build result records
    results = []
    for i in range(len(bfac)):
        results.append({
            "protein": os.path.splitext(os.path.basename(pdb_path))[0],
            "residue_id": i + 1,
            "aa": aa[i],
            "bfactor": float(bfac[i]),
            "plddt": None if np.isnan(plddt[i]) else float(plddt[i]),
            "cl": round(float(cl[i]), 3),
            "bfactor_norm": float(bfac_norm[i]),
            "hydro_entropy": float(hydro_ent[i]),
            "charge_entropy": float(charge_ent[i]),
            "bfactor_curv": float(curv_raw[i]),
            "bfactor_curv_entropy": float(curv_ent[i]),
            "bfactor_curv_flips": float(curv_flips[i]),
            "note": "Unstable" if cl[i] > 0.6 else "Stable"
        })
    
    return results

# Legacy compatibility - keeping old function names
def compute_curvature_features(pdb_path: str) -> List[Dict]:
    """Legacy wrapper for compatibility"""
    return compute_ewcl_from_pdb(pdb_path)
