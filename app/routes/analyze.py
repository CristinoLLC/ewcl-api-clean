from fastapi import APIRouter, UploadFile, File, HTTPException
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
import io
import math

router = APIRouter()


def parse_pdb(file_bytes: bytes) -> pd.DataFrame:
    """Parse PDB into residue-level table (chain, position, aa, support)."""
    parser = PDBParser(QUIET=True)
    handle = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))
    structure = parser.get_structure("model", handle)

    rows = []
    for model in structure:
        for chain in model:
            # Precompute valid chain B-factors for sanitization fallback
            chain_bfactors = [a.get_bfactor() for a in chain.get_atoms()]
            valid_chain_bfactors = [float(v) for v in chain_bfactors if np.isfinite(v) and float(v) >= 0.0]
            median_chain_b = float(np.median(valid_chain_bfactors)) if valid_chain_bfactors else 0.0

            for res in chain:
                if "CA" not in res:
                    continue
                atom = res["CA"]
                raw_b = atom.get_bfactor()
                if not np.isfinite(raw_b) or float(raw_b) < 0.0:
                    bfactor = median_chain_b
                    sanitized = True
                else:
                    bfactor = float(raw_b)
                    sanitized = False

                rows.append({
                    "chain": chain.id,
                    "position": res.id[1],
                    "aa": res.resname,
                    "support": bfactor,
                    "sanitized": sanitized,
                })
    return pd.DataFrame(rows)


def compute_ewcl(df: pd.DataFrame, source_type: str, w: int = 7, alpha: float = 0.5, beta: float = 0.5) -> pd.DataFrame:
    """
    Compute EWCL features.
    - AlphaFold: support = pLDDT → inv_conf + entropy → cl_norm
    - X-ray: support = B-factor → normalized directly
    """
    df = df.copy()
    # Force numeric dtype for support to avoid object arrays
    df["support"] = pd.to_numeric(df["support"], errors="coerce").astype(float)
    df["support_type"] = source_type

    if source_type == "plddt":
        df["support_norm"] = df["support"].astype(float) / 100.0
        df["inv_conf"] = 1.0 - df["support_norm"]

        arr = df["inv_conf"].to_numpy()
        ent = np.zeros_like(arr, dtype=float)
        half = w // 2
        for i in range(len(arr)):
            lo, hi = max(0, i - half), min(len(arr), i + half + 1)
            window = arr[lo:hi]
            hist, _ = np.histogram(window, bins=10, range=(0, 1), density=True)
            hist = hist + 1e-6
            ent[i] = shannon_entropy(hist, base=2)
        df["entropy"] = ent

        # EWCL collapse likelihood: inv_conf + entropy (per-chain normalized)
        df["cl_raw"] = df["inv_conf"] + df["entropy"]
        df["cl_norm"] = df.groupby("chain")["cl_raw"].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        # Reverse collapse likelihood: proxy for disorder
        df["rev_cl"] = 1.0 - df["cl_norm"]
        # Ensure sanitized field exists and is boolean False for AF
        if "sanitized" not in df.columns:
            df["sanitized"] = False
        else:
            df["sanitized"] = False

    elif source_type == "bfactor":
        # Sanitized pass-through for X-ray: per-chain max normalization
        chain_max = df.groupby("chain")["support"].transform("max").astype(float)
        df["support_norm"] = df["support"].astype(float) / (chain_max + 1e-6)
        df["inv_conf"] = None
        df["entropy"] = None
        df["cl_raw"] = None
        df["cl_norm"] = None
        df["rev_cl"] = None
        # sanitized field is set in parse step; ensure it exists
        if "sanitized" not in df.columns:
            df["sanitized"] = False

    return df


@router.post("/api/analyze/main")
async def analyze_main(file: UploadFile = File(...)):
    """
    EWCL proxy analyzer:
      - Detect AlphaFold vs X-ray by support range
      - Returns per-residue features: support, support_type, support_norm, inv_conf, entropy, cl_raw, cl_norm
    """
    try:
        content = await file.read()
        df = parse_pdb(content)
        if df.empty:
            raise HTTPException(status_code=400, detail="No residues found in PDB")

        source_type = "plddt" if float(df["support"].max()) <= 100.0 else "bfactor"
        df = compute_ewcl(df, source_type)

        return {"residues": df.to_dict(orient="records"), "n": int(len(df)), "source_type": source_type}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


