#!/usr/bin/env python3
# ─────────────────── enhanced_ewcl_af.py ───────────────────
"""
Physics-based (no-neural) Collapse-Likelihood model – AlphaFold-aware.
• Same features as the original script.
• PLUS: detects AlphaFold PDBs and copies their per-residue pLDDT.
"""

import os, json, glob, re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import entropy, pearsonr
from Bio.PDB import PDBParser

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
    x = x.astype(float)
    ptp = np.ptp(x)
    return (x - x.min()) / (ptp + 1e-9)

def curvature(arr):
    return np.gradient(np.gradient(arr.astype(float)))

def sliding_entropy(arr, win=5):
    out = np.zeros_like(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    for i in range(len(arr)):
        s, e = max(0, i - win // 2), min(len(arr), i + win // 2 + 1)
        hist, _ = np.histogram(arr[s:e], bins=10, range=(lo, hi), density=True)
        out[i] = entropy(hist + 1e-9)
    return out

def sign_flip_ratio(arr, win=5):
    flips = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        s, e = max(0, i - win // 2), min(len(arr), i + win // 2 + 1)
        diff_sign = np.sign(np.diff(arr[s:e]))
        flips[i] = np.sum(np.diff(diff_sign) != 0) / max(1, len(diff_sign) - 1)
    return flips

# ──────────────── main model ────────────────
def compute_curvature_features(pdb_path, win_ent=5, win_curv=5):
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as fh:
        first_60 = ''.join([fh.readline() for _ in range(60)])
    is_af = bool(re.search(r'ALPHAFOLD', first_60, re.I)) \
            or os.path.basename(pdb_path).startswith("AF-")

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("X", pdb_path)[0]

    bfac_raw, aa, plddt_raw = [], [], []
    chain_ids, positions = [], []
    for res in model.get_residues():
        if "CA" not in res:
            continue
        beta = res["CA"].get_bfactor()
        bfac_raw.append(beta)
        # AlphaFold stores pLDDT in B-factor column
        plddt_raw.append(beta if is_af else None)
        aa.append(res.get_resname())
        chain_ids.append(res.get_parent().id)
        positions.append(res.id[1])

    bfac = np.array(bfac_raw)
    
    bfac_norm = _norm(bfac)
    hydro = np.array([HYDROPATHY.get(a, 0) for a in aa])
    charge = np.array([CHARGE.get(a, 0) for a in aa])
    hydro_ent = sliding_entropy(hydro, win_ent)
    charge_ent = sliding_entropy(charge, win_ent)
    hydro_ent_n = _norm(hydro_ent)
    charge_ent_n = _norm(charge_ent)

    win_curv_adj = max(3, win_curv | 1)
    win_curv_adj = min(win_curv_adj, len(bfac) - (1 - len(bfac) % 2))

    curv_raw = curvature(bfac)
    curv_savgol = curvature(savgol_filter(bfac, win_curv_adj, 2))
    curv_mean = curvature(uniform_filter1d(bfac, size=win_curv_adj))
    curv_median = curvature(uniform_filter1d(bfac, size=win_curv_adj, mode='nearest'))
    curv_ent = sliding_entropy(curv_raw, win_ent)
    curv_flips = sign_flip_ratio(curv_raw, win_ent)
    curv_clip = np.clip((curv_raw - curv_raw.mean()) / (curv_raw.std() + 1e-6), -2, 2)

    cl = (0.35 * hydro_ent_n + 0.35 * charge_ent_n + 0.30 * bfac_norm)

    recs = []
    basename = os.path.splitext(os.path.basename(pdb_path))[0]
    for i in range(len(bfac)):
        recs.append({
            "protein": basename,
            "residue_id": i + 1,
            "chain": chain_ids[i],
            "position": positions[i],
            "aa": aa[i],
            # keep raw values for correlation only
            "bfactor": None if is_af else float(bfac_raw[i]),
            "plddt": None if not is_af else float(plddt_raw[i]),
            # existing fields
            "cl": round(float(cl[i]), 3),
            "bfactor_norm": float(bfac_norm[i]),
            "hydro_entropy": float(hydro_ent[i]),
            "charge_entropy": float(charge_ent[i]),
            "bfactor_curv": float(curv_raw[i]),
            "bfactor_curv_entropy": float(curv_ent[i]),
            "bfactor_curv_flips": float(curv_flips[i]),
            "note": "Unstable" if cl[i] > 0.6 else "Stable"
        })
    return recs

# ──────────────── batch run ────────────────
def run_batch(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for pdb in glob.glob(os.path.join(in_dir, "*.[pe][nb][td]")):
        recs = compute_curvature_features(pdb)
        with open(os.path.join(out_dir, os.path.splitext(os.path.basename(pdb))[0] + ".json"), "w") as fh:
            json.dump(recs, fh, indent=2)
    print(f"✅  Saved enhanced JSONs ➜  {out_dir}")

def quick_corr(out_dir):
    rows = []
    for j in glob.glob(os.path.join(out_dir, "*.json")):
        with open(j) as fh:
            rows.extend(json.load(fh))
    df = pd.DataFrame(rows)
    r_b, p_b = pearsonr(df["cl"], df["bfactor"])
    msg = f"📈  r(CL, B-factor) = {r_b:+.3f}  (p={p_b:.1e})"
    if df["plddt"].notna().any():
        r_p, p_p = pearsonr(df.loc[df["plddt"].notna(), "cl"], df.loc[df["plddt"].notna(), "plddt"])
        msg += f"   |   r(CL, pLDDT) = {r_p:+.3f}  (p={p_p:.1e})"
    print(msg)

# ──────────────── main (Colab-safe) ────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Enhanced EWCL extractor (AlphaFold-aware)")
    ap.add_argument("--in_dir", default="pdbs_af", help="folder with PDB/ENT files")
    ap.add_argument("--out_dir", default="jsons_af", help="where JSONs are written")
    ap.add_argument("--corr", action="store_true", help="print quick Pearson r")
    args, _ = ap.parse_known_args()  # ✅ fixes Colab argparse crash

    run_batch(args.in_dir, args.out_dir)
    if args.corr:
        quick_corr(args.out_dir)