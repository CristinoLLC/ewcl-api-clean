#!/usr/bin/env python3
# ─────────────────── enhanced_ewcl_af.py ───────────────────
"""
Physics-based (no-neural) Collapse-Likelihood model – AlphaFold-aware.
Clean, surgical fix for consistent JSON output and proper B-factor handling.
"""

import os, re, json, numpy as np, pandas as pd
from Bio.PDB import PDBParser
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import entropy

HYDROPATHY = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,'CYS':2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,'HIS':-3.2,'ILE':4.5,'LEU':3.8,'LYS':-3.9,'MET':1.9,'PHE':2.8,'PRO':-1.6,'SER':-0.8,'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}
CHARGE     = {aa:0 for aa in HYDROPATHY}; CHARGE.update({'ASP':-1,'GLU':-1,'ARG':1,'LYS':1,'HIS':0.5})

def _norm(x):
    x = np.asarray(x, float)
    if len(x)==0: return x
    rng = x.max()-x.min()
    return (x - x.min())/(rng + 1e-9)

def _curvature(arr):  # simple second derivative
    arr = np.asarray(arr, float)
    return np.gradient(np.gradient(arr))

def _sliding_entropy(arr, win=5):
    arr = np.asarray(arr, float)
    out = np.zeros_like(arr, float)
    lo, hi = arr.min(), arr.max()
    for i in range(len(arr)):
        s, e = max(0, i-win//2), min(len(arr), i+win//2+1)
        hist, _ = np.histogram(arr[s:e], bins=10, range=(lo,hi), density=True)
        out[i] = entropy(hist + 1e-9)
    return out

def _sign_flip_ratio(arr, win=5):
    arr = np.asarray(arr, float)
    out = np.zeros_like(arr, float)
    for i in range(len(arr)):
        s, e = max(0, i-win//2), min(len(arr), i+win//2+1)
        d = np.sign(np.diff(arr[s:e]))
        out[i] = np.sum(np.diff(d) != 0) / max(1, len(d)-1)
    return out

def compute_curvature_features(pdb_path: str, bf_mode: str = "all", ent_win=5, curv_win=5):
    """
    bf_mode: "all" (mean over all atoms in residue) | "ca" (CA only)
    Returns a uniform object: { protein_id, summary, residues:[...] }
    """
    # detect AF by header or filename
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as fh:
        head = ''.join([fh.readline() for _ in range(60)])
    is_af = bool(re.search(r'ALPHAFOLD', head, re.I)) or os.path.basename(pdb_path).startswith("AF-")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)[0]
    protein_id = os.path.splitext(os.path.basename(pdb_path))[0]

    # collect per-residue features
    residues = []
    for res in structure.get_residues():
        # skip het/water
        hetflag, resseq, icode = res.id
        if hetflag != " ": 
            continue

        # atoms B list
        b_all = []
        for atom in res.get_atoms():
            try:
                b_all.append(float(atom.get_bfactor()))
            except Exception:
                pass

        if not b_all:
            continue

        bfactor_all = float(np.mean(b_all))
        bfactor_ca  = None
        if "CA" in res:
            bfactor_ca = float(res["CA"].get_bfactor())

        # AlphaFold: B-factor column carries pLDDT
        plddt = bfactor_ca if (is_af and bfactor_ca is not None) else (bfactor_all if is_af else None)

        residues.append(dict(
            chain=res.get_parent().id,
            position=resseq,
            aa=res.get_resname(),
            bfactor_all=bfactor_all,
            bfactor_ca=bfactor_ca,
            plddt=plddt
        ))

    if not residues:
        return dict(protein_id=protein_id, summary=dict(total_residues=0), residues=[])

    df = pd.DataFrame(residues).reset_index(drop=True)

    # choose which bfactor to expose
    if bf_mode == "ca":
        df["bfactor"] = df["bfactor_ca"]
        bsrc = "CA"
    else:
        df["bfactor"] = df["bfactor_all"]
        bsrc = "ALL"

    # physics features (use exposed bfactor stream)
    b = df["bfactor"].fillna(0).to_numpy(float)
    b_norm = _norm(b)
    hydro = np.array([HYDROPATHY.get(aa,0) for aa in df["aa"]])
    charge= np.array([CHARGE.get(aa,0) for aa in df["aa"]])
    hydro_ent  = _sliding_entropy(hydro, ent_win); hydro_ent_n  = _norm(hydro_ent)
    charge_ent = _sliding_entropy(charge, ent_win); charge_ent_n = _norm(charge_ent)

    # curvature family on b-factor stream
    # keep window odd and sane
    w = max(3, curv_win | 1); w = min(w, len(b) - (1 - len(b)%2))
    curv_raw   = _curvature(b)
    curv_savg  = _curvature(savgol_filter(b, w, 2)) if len(b) >= w else curv_raw
    curv_mean  = _curvature(uniform_filter1d(b, size=w))
    curv_median= _curvature(uniform_filter1d(b, size=w, mode='nearest'))
    curv_ent   = _sliding_entropy(curv_raw, ent_win)
    curv_flips = _sign_flip_ratio(curv_raw, ent_win)

    # physics-only CL (no ML, no target leakage)
    cl = (0.35*hydro_ent_n + 0.35*charge_ent_n + 0.30*b_norm)

    df["cl"] = np.round(cl, 3)
    df["bfactor_norm"] = b_norm
    df["hydro_entropy"] = hydro_ent
    df["charge_entropy"] = charge_ent
    df["bfactor_curv"] = curv_raw
    df["bfactor_curv_entropy"] = curv_ent
    df["bfactor_curv_flips"] = curv_flips
    df["note"] = np.where(df["cl"]>0.6, "Unstable", "Stable")

    # output as uniform object
    out = dict(
        protein_id=protein_id,
        summary=dict(
            total_residues=int(len(df)),
            is_alphafold=bool(is_af),
            bfactor_source=bsrc
        ),
        residues=df[[
            "chain","position","aa","cl","bfactor","plddt",
            "bfactor_norm","hydro_entropy","charge_entropy",
            "bfactor_curv","bfactor_curv_entropy","bfactor_curv_flips","note"
        ]].to_dict("records")
    )
    return out

# ──────────────── batch run ────────────────
def run_batch(in_dir, out_dir, bf_mode="all"):
    os.makedirs(out_dir, exist_ok=True)
    for pdb in glob.glob(os.path.join(in_dir, "*.[pe][nb][td]")):
        recs = compute_curvature_features(pdb, bf_mode=bf_mode)
        with open(os.path.join(out_dir, os.path.splitext(os.path.basename(pdb))[0] + ".json"), "w") as fh:
            json.dump(recs, fh, indent=2)
    print(f"✅  Saved enhanced JSONs ➜  {out_dir}")

def quick_corr(out_dir):
    rows = []
    for j in glob.glob(os.path.join(out_dir, "*.json")):
        with open(j) as fh:
            rows.extend(json.load(fh)["residues"])
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
    ap.add_argument("--bf_mode", default="all", choices=["all", "ca"], help="B-factor mode: all atoms or CA only")
    ap.add_argument("--corr", action="store_true", help="print quick Pearson r")
    args, _ = ap.parse_known_args()  # ✅ fixes Colab argparse crash

    run_batch(args.in_dir, args.out_dir, bf_mode=args.bf_mode)
    if args.corr:
        quick_corr(args.out_dir)