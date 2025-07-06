"""
Physics-based EWCL model
────────────────────────
• No external ML checkpoints
• Pure NumPy + BioPython
• Deterministic & thread-safe
"""

import json, math, warnings
from typing import List, Dict
from Bio.PDB import PDBParser, is_aa
import numpy as np

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
def _entropy(arr: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of a 1-D array (continuous → hist)."""
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def _backbone_curvature(ca_coords: np.ndarray) -> np.ndarray:
    """Discrete curvature κ = |r''| using central finite difference."""
    n = len(ca_coords)
    curv = np.zeros(n)
    for i in range(1, n-1):
        r_prev, r, r_next = ca_coords[i-1], ca_coords[i], ca_coords[i+1]
        v1, v2 = r - r_prev, r_next - r
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 0:
            angle = np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            curv[i] = angle
    curv[0] = curv[1]
    curv[-1] = curv[-2]
    return curv

# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def compute_ewcl_from_pdb(pdb_path: str) -> List[Dict]:
    """
    Parameters
    ----------
    pdb_path : str
        Path to *.pdb* (AlphaFold or experimental)

    Returns
    -------
    List[Dict]
        One dict per residue with EWCL features + final 'cl' score
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)

    # ---- gather CA coords & basic features ----
    residues, ca_coords = [], []
    for res in structure.get_residues():
        if not is_aa(res, standard=True):
            continue
        if "CA" not in res:        # skip weird alt-locs
            continue
        residues.append(res)
        ca_coords.append(res["CA"].get_vector().get_array())

    ca_coords = np.array(ca_coords)
    curvature = _backbone_curvature(ca_coords)

    # ---- sliding-window entropies (hydropathy / charge) ----
    hydro = np.array([HYDRO_SCORES.get(res.get_resname()[0], 0.0)
                      for res in residues])
    charge = np.array([CHARGE.get(res.get_resname()[0], 0)
                       for res in residues])
    window = 21  # ~10 residues either side
    pad = window // 2

    hydro_entropy = np.array([
        _entropy(hydro[max(0, i-pad): i+pad+1]) for i in range(len(residues))
    ])
    charge_entropy = np.array([
        _entropy(charge[max(0, i-pad): i+pad+1], bins=5)
        for i in range(len(residues))
    ])

    # ---- collapse-likelihood (CL) heuristic ----
    # low entropy + high hydrophobicity + low curvature → high collapse
    norm_hydro   = (hydro - hydro.min()) / (hydro.ptp() + 1e-9)
    norm_curv    = (curvature - curvature.min()) / (curvature.ptp()+1e-9)
    norm_h_ent   = (hydro_entropy - hydro_entropy.min()) / (hydro_entropy.ptp()+1e-9)

    cl = np.clip( 1.2*norm_hydro - 0.6*norm_curv - 0.4*norm_h_ent , 0 , 1 )

    # ---- output ----
    results = []
    for idx, res in enumerate(residues):
        chain = res.get_parent().id
        res_id = res.id[1]
        results.append({
            "chain":   chain,
            "residue_id": int(res_id),
            "aa": res.get_resname()[0],
            "bfactor": float(res["CA"].get_bfactor()),
            "entropy_hydropathy": float(hydro_entropy[idx]),
            "entropy_charge": float(charge_entropy[idx]),
            "curvature": float(curvature[idx]),
            "cl": float(cl[idx])
        })
    return results
