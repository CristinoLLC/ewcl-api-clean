from __future__ import annotations
import json
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Simple AA properties (hydropathy, charge)
AA_PROPERTIES: Dict[str, Tuple[float, float]] = {
    'A': (1.8, 0), 'R': (-4.5, 1), 'N': (-3.5, 0), 'D': (-3.5, -1),
    'C': (2.5, 0), 'E': (-3.5, -1), 'Q': (-3.5, 0), 'G': (-0.4, 0),
    'H': (-3.2, 0), 'I': (4.5, 0), 'L': (3.8, 0), 'K': (-3.9, 1),
    'M': (1.9, 0), 'F': (2.8, 0), 'P': (-1.6, 0), 'S': (-0.8, 0),
    'T': (-0.7, 0), 'W': (-0.9, 0), 'Y': (-1.3, 0), 'V': (4.2, 0)
}

# EWCLv1-C feature order (hardcoded from the training data)
EWCLV1_C_FEATURES = [
    "position",
    "sequence_length", 
    "position_ratio",
    "delta_hydropathy",
    "delta_charge",
    "delta_entropy_w5",
    "delta_entropy_w11",
    "has_embeddings",
    "delta_helix_prop",
    "delta_sheet_prop",
    "delta_entropy_w25",
    "ewcl_hydropathy",
    "ewcl_charge_pH7",
    "ewcl_entropy_w5",
    "ewcl_entropy_w11",
    "emb_0", "emb_1", "emb_2", "emb_3", "emb_4", "emb_5", "emb_6", "emb_7",
    "emb_8", "emb_9", "emb_10", "emb_11", "emb_12", "emb_13", "emb_14", "emb_15",
    "emb_16", "emb_17", "emb_18", "emb_19", "emb_20", "emb_21", "emb_22", "emb_23",
    "emb_24", "emb_25", "emb_26", "emb_27", "emb_28", "emb_29", "emb_30", "emb_31"
]

def parse_fasta(fasta_str: str) -> str:
    lines = fasta_str.strip().splitlines()
    seq_parts: List[str] = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith(">"):
            continue
        seq_parts.append(l)
    return ("".join(seq_parts)).upper()


def parse_vcf(vcf_str: str) -> List[Dict]:
    variants: List[Dict] = []
    for line in vcf_str.splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        pos = int(parts[1])
        ref = parts[3].strip()[:1]
        alt = parts[4].strip()[:1]
        variants.append({"pos": pos, "ref": ref, "alt": alt})
    return variants


def parse_json_variants(json_str: str) -> Tuple[str, List[Dict]]:
    data = json.loads(json_str)
    seq = str(data["sequence"]).upper()
    variants = data.get("variants", [])
    return seq, variants


def make_feature_vector(seq: str, pos: int, ref: str, alt: str, emb: np.ndarray | None = None) -> Dict[str, float]:
    L = len(seq)
    ratio = (pos / L) if L > 0 else 0.0
    hyd_ref, ch_ref = AA_PROPERTIES.get(ref.upper(), (0.0, 0.0))
    hyd_alt, ch_alt = AA_PROPERTIES.get(alt.upper(), (0.0, 0.0))
    emb = emb if emb is not None else np.zeros(32, dtype=float)

    feats: Dict[str, float] = {
        "position": float(pos),
        "sequence_length": float(L),
        "position_ratio": float(ratio),
        "delta_hydropathy": float(hyd_alt - hyd_ref),
        "delta_charge": float(ch_alt - ch_ref),
        # TODO: hook real secondary/entropy signals; placeholders for now
        "delta_helix_prop": 0.0,
        "delta_sheet_prop": 0.0,
        "delta_entropy_w5": 0.0,
        "delta_entropy_w11": 0.0,
        "delta_entropy_w25": 0.0,
        "has_embeddings": 0.0 if emb is None else 1.0,
    }
    for i in range(32):
        feats[f"emb_{i}"] = float(emb[i])
    return feats


def build_features(seq: str, variants: List[Dict], feature_order: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for v in variants:
        pos = int(v["pos"]) if "pos" in v else int(v.get("position"))
        ref = str(v.get("ref") or v.get("wt") or v.get("ref_aa") or "X")[:1]
        alt = str(v.get("alt") or v.get("mut") or v.get("alt_aa") or "X")[:1]
        fv = make_feature_vector(seq, pos, ref, alt, emb=np.zeros(32, dtype=float))
        rows.append(fv)
    df = pd.DataFrame(rows)
    # Ensure exact order (missing columns filled with 0)
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0
    df = df.loc[:, feature_order]
    return df


