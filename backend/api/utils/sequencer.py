from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

AA_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

AA_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.5, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0
}

def _window_entropy(values: np.ndarray, win: int = 9, bins: int = 10) -> np.ndarray:
    values = np.asarray(values, float)
    n = len(values)
    out = np.zeros(n)
    half = win // 2
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return out
    edges = np.linspace(lo, hi + 1e-9, bins + 1)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        hist, _ = np.histogram(values[s:e], bins=edges, density=True)
        out[i] = shannon_entropy(hist + 1e-12)
    rng = out.max() - out.min()
    if rng > 1e-12:
        out = (out - out.min()) / rng
    return out

def parser_ewclv1(seq: str) -> pd.DataFrame:
    rows = []
    for i, aa in enumerate(seq, start=1):
        hydro = AA_HYDROPATHY.get(aa.upper(), 0.0)
        charge = AA_CHARGE.get(aa.upper(), 0.0)
        rows.append({"residue_index": i, "aa": aa.upper(), "hydropathy": hydro, "charge": charge})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["hydro_entropy"] = _window_entropy(df["hydropathy"].values)
    df["charge_entropy"] = _window_entropy(df["charge"].values)
    diffs = np.abs(np.diff(df["hydropathy"], prepend=df["hydropathy"].iloc[0]))
    rng = diffs.max() - diffs.min()
    df["curvature"] = 0.0 if rng < 1e-12 else (diffs - diffs.min()) / (rng + 1e-12)
    flips = [0]
    for a, b in zip(df["hydropathy"].values[:-1], df["hydropathy"].values[1:]):
        flips.append(1 if (a * b) < 0 else 0)
    df["flips"] = flips
    return df

def parser_ewclv1m(seq: str) -> pd.DataFrame:
    return parser_ewclv1(seq)


