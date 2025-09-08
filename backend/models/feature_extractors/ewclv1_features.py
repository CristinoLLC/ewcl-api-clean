# backend/models/feature_extractors/ewclv1_features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd

# ---------- constants ----------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

# Kyte-Doolittle hydropathy
KD = {
    'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
    'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,
    'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}

# Polarity (Grantham; lower=nonpolar, higher=polar)
POLARITY = {
    'G':0,'A':0,'V':0,'L':0,'I':0,'F':0,'W':0,'Y':0,
    'S':1,'T':1,'C':1,'M':1,'N':1,'Q':1,'D':1,'E':1,'K':1,'R':1,'H':1,'P':0
}

# VdW volume (approx, A^3)
VDW = {
    'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,
    'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,
    'T':116.1,'W':227.8,'Y':193.6,'V':140.0
}

# Flexibility (Bhaskaran & Ponnuswamy)
FLEX = {
    'A':0.357,'R':0.529,'N':0.463,'D':0.511,'C':0.346,'Q':0.493,'E':0.497,'G':0.544,
    'H':0.323,'I':0.462,'L':0.365,'K':0.466,'M':0.295,'F':0.314,'P':0.509,'S':0.507,
    'T':0.444,'W':0.305,'Y':0.420,'V':0.386
}

# Bulkiness (Zimmerman)
BULK = {
    'A':11.5,'R':14.28,'N':12.82,'D':11.68,'C':13.46,'Q':14.45,'E':13.57,'G':3.4,
    'H':13.69,'I':21.4,'L':21.4,'K':15.71,'M':16.25,'F':19.8,'P':17.43,'S':9.47,
    'T':15.77,'W':21.67,'Y':18.03,'V':21.57
}

# Chou-Fasman helix & sheet propensity (normalized)
HELIX = {'A':1.45,'R':1.00,'N':0.67,'D':1.01,'C':0.77,'Q':1.11,'E':1.51,'G':0.57,
         'H':1.00,'I':1.08,'L':1.34,'K':1.07,'M':1.20,'F':1.12,'P':0.57,'S':0.77,
         'T':0.83,'W':1.14,'Y':0.61,'V':1.06}
SHEET = {'A':0.97,'R':0.90,'N':0.89,'D':0.54,'C':1.30,'Q':1.10,'E':0.37,'G':0.75,
         'H':0.87,'I':1.60,'L':1.22,'K':0.74,'M':1.67,'F':1.28,'P':0.55,'S':0.75,
         'T':1.19,'W':1.19,'Y':1.29,'V':1.70}

# Charge at pH 7 (very coarse: +1 K/R, -1 D/E, ~0 others; H as +0.1)
CHARGE7 = {**{a:0.0 for a in AA}, 'K':+1.0,'R':+1.0,'D':-1.0,'E':-1.0,'H':+0.1}

WINDOWS = [5, 11, 25, 50, 100]

# AA to index mapping for aa_encoded feature
AA_TO_IDX = {a: i for i, a in enumerate("ARNDCQEGHILKMFPSTWYV")}

# ---------- helpers ----------
def _seq_to_vec(seq: str, table: Dict[str, float]) -> np.ndarray:
    return np.array([table.get(a, 0.0) for a in seq], dtype=float)

def _rolling_stats(x: np.ndarray, w: int) -> Dict[str, np.ndarray]:
    # Centered window; pad at ends
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    out_mean, out_std, out_min, out_max = [], [], [], []
    for i in range(len(x)):
        win = xpad[i:i+w]
        out_mean.append(win.mean())
        out_std.append(win.std())
        out_min.append(win.min())
        out_max.append(win.max())
    return {
        "mean": np.array(out_mean),
        "std":  np.array(out_std),
        "min":  np.array(out_min),
        "max":  np.array(out_max),
    }

def _window_entropy(seq: str, w: int) -> np.ndarray:
    # Shannon entropy of AA distribution in window
    pad = w // 2
    spad = seq[0]*pad + seq + seq[-1]*pad
    out = np.zeros(len(seq))
    for i in range(len(seq)):
        win = spad[i:i+w]
        counts = np.array([win.count(a) for a in AA], dtype=float)
        p = counts / counts.sum()
        # avoid log(0)
        nz = p > 0
        H = -(p[nz]*np.log(p[nz])).sum()
        out[i] = H
    return out

def _low_complex_from_entropy(H: np.ndarray, thresh: float = 1.5) -> np.ndarray:
    # Flag low complexity by entropy threshold
    return (H < thresh).astype(float)

def _composition_bias(seq: str, w: int) -> np.ndarray:
    # (max_fraction - uniform_fraction)
    pad = w // 2
    spad = seq[0]*pad + seq + seq[-1]*pad
    out = np.zeros(len(seq))
    for i in range(len(seq)):
        win = spad[i:i+w]
        counts = np.array([win.count(a) for a in AA], dtype=float)
        frac = counts / counts.sum()
        out[i] = frac.max() - (1.0/20.0)
    return out

def _uversky_distance(hydro_mean: np.ndarray, charge_mean: np.ndarray) -> np.ndarray:
    # Uversky boundary (approx): H = 1.151*|Q| + 0.693; distance = H_mean - (1.151*|Q_mean| + 0.693)
    return hydro_mean - (1.151*np.abs(charge_mean) + 0.693)

def _poly_run_flags(seq: str, aa: str, run_len: int = 3) -> np.ndarray:
    out = np.zeros(len(seq))
    cur = 0
    for i, c in enumerate(seq):
        cur = cur + 1 if c == aa else 0
        if cur >= run_len:
            out[i] = 1.0
    return out

def _scd_local(charge_vec: np.ndarray, w: int = 25) -> np.ndarray:
    """SCD (Sequence Charge Decoration) - local charge decoration metric"""
    pad = w // 2
    xpad = np.pad(charge_vec, (pad, pad), mode='edge')
    out = np.zeros(len(charge_vec))
    # Sawle & Ghosh-like: sum_{i<j} q_i q_j |i-j|^0.5 within window, normalized
    for i in range(len(charge_vec)):
        win = xpad[i:i+w]
        s = 0.0
        for a in range(len(win)):
            qa = win[a]
            if qa == 0.0: 
                continue
            for b in range(a+1, len(win)):
                qb = win[b]
                if qb == 0.0:
                    continue
                s += qa * qb * ((b - a) ** 0.5)
        # normalize by number of pairs to keep scale stable
        denom = (len(win)*(len(win)-1))/2 or 1.0
        out[i] = s / denom
    return out

@dataclass
class FeatureBlock:
    base_df: pd.DataFrame
    pssm_df: pd.DataFrame
    all_df: pd.DataFrame
    has_pssm: bool

# ---------- main API ----------
def build_ewclv1_features(seq: str,
                          pssm: pd.DataFrame | None = None,
                          expand_aa_onehot: bool = False) -> FeatureBlock:
    """
    Build the feature matrix for EWCL v1. If `pssm` is None, PSSM fields will be zeroed and has_pssm_data=0.
    PSSM expected columns: 'A','R','N','D',...,'Y','V' plus we compute pssm_entropy, pssm_max_score, pssm_variance.
    """
    seq = seq.strip().upper()
    n = len(seq)
    idx = np.arange(1, n+1)

    # scalar per-residue tracks using EXACT names from schema
    hyd = _seq_to_vec(seq, KD)
    pol = _seq_to_vec(seq, POLARITY) 
    vdw = _seq_to_vec(seq, VDW)
    flex = _seq_to_vec(seq, FLEX)
    bulk = _seq_to_vec(seq, BULK)
    helx = _seq_to_vec(seq, HELIX)
    beta = _seq_to_vec(seq, SHEET)
    chg  = _seq_to_vec(seq, CHARGE7)

    # rolling stats for each track & each window - use EXACT prefixes from schema
    cols = {}
    def emit_track(prefix: str, x: np.ndarray):
        # Only emit windowed stats, not the base track itself
        for w in WINDOWS:
            stats = _rolling_stats(x, w)
            cols[f"{prefix}_w{w}_mean"] = stats["mean"]
            cols[f"{prefix}_w{w}_std"]  = stats["std"]
            cols[f"{prefix}_w{w}_min"]  = stats["min"]
            cols[f"{prefix}_w{w}_max"]  = stats["max"]

    # Use EXACT names expected by model - generate windowed stats only
    emit_track("hydro", hyd)        # window stats will be hydro_w5_mean, etc.
    emit_track("polar", pol)        # window stats will be polar_w5_mean, etc. (NOT polarity_)
    emit_track("vdw", vdw)          # window stats will be vdw_w5_mean, etc.
    emit_track("flex", flex)        # window stats will be flex_w5_mean, etc.
    emit_track("bulk", bulk)        # window stats will be bulk_w5_mean, etc.
    emit_track("helix_prop", helx)  # window stats will be helix_prop_w5_mean, etc.
    emit_track("sheet_prop", beta)  # window stats will be sheet_prop_w5_mean, etc.
    emit_track("charge", chg)       # window stats will be charge_w5_mean, etc.

    # Base single-residue features with EXACT schema names
    cols["hydropathy"] = hyd        # NOT hydro
    cols["polarity"] = pol          # keep as polarity (base feature)
    cols["vdw_volume"] = vdw        # NOT vdw
    cols["flexibility"] = flex      # NOT flex
    cols["bulkiness"] = bulk        # NOT bulk
    cols["helix_prop"] = helx       # keep as helix_prop
    cols["sheet_prop"] = beta       # keep as sheet_prop
    cols["charge_pH7"] = chg        # NOT charge

    # windowed entropies + low complexity + comp bias + Uversky distance
    for w in WINDOWS:
        Hw = _window_entropy(seq, w)
        cols[f"entropy_w{w}"]    = Hw
        cols[f"low_complex_w{w}"] = _low_complex_from_entropy(Hw)
        cols[f"comp_bias_w{w}"]   = _composition_bias(seq, w)
        cols[f"uversky_dist_w{w}"] = _uversky_distance(cols[f"hydro_w{w}_mean"], cols[f"charge_w{w}_mean"])

    # simple composition fractions (global per window is complex; we add global fractions once)
    counts = np.array([seq.count(a) for a in AA], dtype=float)
    frac = counts / max(1, counts.sum())
    comp_cols = {f"comp_{a}": np.full(n, frac[i]) for i, a in enumerate(AA)}
    cols.update(comp_cols)

    cols["comp_frac_aromatic"] = np.full(n, frac[AA.index('F')] + frac[AA.index('W')] + frac[AA.index('Y')])
    cols["comp_frac_positive"] = np.full(n, frac[AA.index('K')] + frac[AA.index('R')] + frac[AA.index('H')])
    cols["comp_frac_negative"] = np.full(n, frac[AA.index('D')] + frac[AA.index('E')])
    cols["comp_frac_polar"]    = np.full(n, sum(frac[AA.index(x)] for x in ['S','T','N','Q','C','Y','H','K','R','D','E']))
    cols["comp_frac_aliphatic"]= np.full(n, frac[AA.index('A')] + frac[AA.index('V')] + frac[AA.index('L')] + frac[AA.index('I')])
    cols["comp_frac_proline"]  = np.full(n, frac[AA.index('P')])
    cols["comp_frac_glycine"]  = np.full(n, frac[AA.index('G')])

    # poly runs (flags)
    for a in ['P','E','K','Q','S','G','D','N']:
        cols[f"in_poly_{a}_run_ge3"] = _poly_run_flags(seq, a, run_len=3)

    # unknown AA flag (should be 0 for canonical AA)
    cols["is_unknown_aa"] = np.array([0.0 if c in AA_SET else 1.0 for c in seq], dtype=float)

    # compute scd_local (local charge decoration; window=25)
    cols["scd_local"] = _scd_local(cols["charge_pH7"])

    # NOTE: aa_encoded is in base_features but not in all_features in the schema
    # Only add it if the model actually expects it
    # cols["aa_encoded"] = np.array([float(AA_TO_IDX.get(a, -1)) for a in seq])

    base_df = pd.DataFrame(cols, index=idx)

    # ---- PSSM block with EXACT names ----
    has_pssm = pssm is not None and all(k in pssm.columns for k in list(AA))
    if not has_pssm:
        # Zero-fill PSSM features to maintain exact 249 feature count
        pssm_cols = {a: np.zeros(n) for a in AA}  # A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
        pssm_cols["pssm_entropy"] = np.zeros(n)
        pssm_cols["pssm_max_score"] = np.zeros(n)
        pssm_cols["pssm_variance"] = np.zeros(n)
        pssm_cols["has_pssm_data"] = np.zeros(n)
        pssm_df = pd.DataFrame(pssm_cols, index=idx)
    else:
        P = pssm[AA].to_numpy(dtype=float)
        row_sum = P.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        probs = P / row_sum
        H = -(np.where(probs>0, probs*np.log(probs), 0.0)).sum(axis=1)
        pssm_df = pd.DataFrame({a: P[:,i] for i,a in enumerate(AA)}, index=idx)
        pssm_df["pssm_entropy"] = H
        pssm_df["pssm_max_score"] = P.max(axis=1)
        pssm_df["pssm_variance"] = P.var(axis=1)
        pssm_df["has_pssm_data"] = 1.0

    # align indices and concat
    all_df = pd.concat([base_df, pssm_df], axis=1)
    return FeatureBlock(base_df=base_df, pssm_df=pssm_df, all_df=all_df, has_pssm=has_pssm)