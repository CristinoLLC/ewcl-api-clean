from __future__ import annotations
import pandas as pd
import numpy as np
import io
from Bio.PDB import PDBParser
from scipy.stats import entropy as shannon_entropy

# Standard 3-letter to 1-letter amino acid mapping
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Extended mapping for modified residues
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P",
    "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", 
    "CME": "C", "KCX": "K", "MLZ": "K", "FME": "M",
    "UNK": "X", "ASX": "B", "GLX": "Z"
}


def _window_entropy(values: np.ndarray, win: int = 9, bins: int = 10) -> np.ndarray:
    values = np.asarray(values, float)
    n = len(values)
    out = np.zeros(n)
    half = win // 2
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return out
    edges = np.linspace(lo, hi + 1e-9, bins + 1)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        hist, _ = np.histogram(values[s:e], bins=edges, density=True)
        out[i] = shannon_entropy(hist + 1e-12)
    rng = out.max() - out.min()
    if rng > 1e-12:
        out = (out - out.min()) / (rng + 1e-12)
    return out


def parser_pdb_p3(pdb_bytes: bytes) -> pd.DataFrame:
    """Parse PDB and extract per-residue features for EWCLv1p3.
    Returns a DataFrame with columns: chain, residue_index, aa, support, support_norm,
    support_type in {plddt,bfactor,nmr}, curvature, hydro_entropy, charge_entropy, flips.
    """
    text = pdb_bytes.decode("utf-8", errors="ignore")
    up = text.upper()
    is_af = "ALPHAFOLD" in up
    is_nmr = " NMR " in up or "EXPDTA    NMR" in up

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", io.StringIO(text))

    rows = []
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" not in res:
                    continue
                try:
                    ca = res["CA"]
                    b = float(ca.get_bfactor())
                    # Convert 3-letter residue name to 1-letter amino acid code
                    resname = res.get_resname().strip().upper()
                    aa_single = AA3_TO_1.get(resname, "X")
                    
                    rows.append({
                        "chain": chain.id,
                        "residue_index": int(res.id[1]),
                        "aa": aa_single,  # Now returns single-letter codes
                        "resname": resname,  # Keep original for debugging
                        "support": b,
                        "is_af": is_af,
                        "is_nmr": is_nmr,
                    })
                except Exception:
                    continue
        break  # first model only

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No CA atoms found.")

    if df["is_af"].iloc[0]:
        df["support_norm"] = np.clip(df["support"] / 100.0, 0.0, 1.0)
        df["support_type"] = "plddt"
    elif df["is_nmr"].iloc[0]:
        df["support_norm"] = 0.0  # NMR: no per-residue support metric
        df["support_type"] = "nmr"
    else:
        lo, hi = np.percentile(df["support"], [5, 95])
        df["support_norm"] = np.clip((df["support"] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        df["support_type"] = "bfactor"

    diffs = np.abs(np.diff(df["support_norm"], prepend=df["support_norm"].iloc[0]))
    rng = diffs.max() - diffs.min()
    df["curvature"] = 0.0 if rng < 1e-12 else (diffs - diffs.min()) / (rng + 1e-12)

    df["hydro_entropy"] = _window_entropy(df["support_norm"].values)
    df["charge_entropy"] = _window_entropy(df["curvature"].values)

    flips = [0]
    for a, b in zip(df["support_norm"].values[:-1], df["support_norm"].values[1:]):
        flips.append(1 if (a * b) < 0 else 0)
    df["flips"] = flips

    return df
