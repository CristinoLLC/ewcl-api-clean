from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io, hashlib, numpy as np, pandas as pd
from typing import List, Dict
from Bio.PDB import PDBParser
from Bio import SeqIO
from scipy.stats import entropy as shannon_entropy

# ─────────────────────────────────────────
# 1) Physics extractor (CA-only)
# ─────────────────────────────────────────
# Uses your existing module. If you moved it, update the import path.
from models.enhanced_ewcl_af import compute_curvature_features  # ← keep

def _md5_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def add_rev_cl(df: pd.DataFrame, cl_col: str = "cl", out_col: str = "rev_cl") -> pd.DataFrame:
    """Append reversed collapse likelihood: rev_cl = 1 - cl, clipped to [0,1]."""
    if cl_col not in df.columns:
        df[out_col] = np.nan
        return df
    x = pd.to_numeric(df[cl_col], errors="coerce").astype(float)
    df[out_col] = np.clip(1.0 - x, 0.0, 1.0)
    return df

def run_physics(pdb_bytes: bytes, bf_mode: str = "ca") -> dict:
    """Wrap your physics extractor into a uniform dict."""
    import tempfile, os, json
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        tmp.write(pdb_bytes)
        tmp_path = tmp.name
    try:
        obj = compute_curvature_features(tmp_path, bf_mode=bf_mode)
        # Expecting obj like {"protein_id":..., "residues":[...]}
        if not isinstance(obj, dict) or "residues" not in obj:
            raise ValueError("physics extractor returned unexpected structure")
        return obj
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ─────────────────────────────────────────
# 2) Proxy model (self-contained)
#    Locks: alpha=0.75, beta=0.35, win=7 (CA-only)
# ─────────────────────────────────────────
def _parse_pdb_ca_support(pdb_bytes: bytes) -> pd.DataFrame:
    """CA-only support vector. AF: pLDDT in B-factor; X-ray: B-factor."""
    text = pdb_bytes.decode("utf-8", errors="ignore")
    is_af = ("ALPHAFOLD" in text.upper())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", io.StringIO(text))

    rows = []
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" not in res:
                    continue
                ca = res["CA"]
                b  = float(ca.get_bfactor())
                rows.append({
                    "chain": chain.id,
                    "position": int(res.id[1]),
                    "aa": res.get_resname(),
                    "support": b,    # AF: pLDDT; X-ray: B
                    "is_af": is_af
                })
        break  # first model only

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No CA atoms found.")
    if df["is_af"].iloc[0]:
        df["support_norm"] = np.clip(df["support"] / 100.0, 0.0, 1.0)
        support_type = "plddt"
    else:
        lo, hi = np.percentile(df["support"], [5,95])
        df["support_norm"] = np.clip((df["support"] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        support_type = "bfactor"

    df["uncertainty_norm"] = 1.0 - df["support_norm"] if support_type=="plddt" else df["support_norm"]
    df["support_type"] = support_type
    return df

def _local_entropy(x: np.ndarray, win: int = 7, bins: int = 10) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x); out = np.zeros(n, float)
    half = win // 2
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(n):
        s = max(0, i-half); e = min(n, i+half+1)
        hist, _ = np.histogram(x[s:e], bins=edges, density=True)
        out[i] = shannon_entropy(hist + 1e-12)
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out

def compute_proxy_from_pdb_bytes(pdb_bytes: bytes,
                                 alpha: float = 0.7,
                                 beta: float = 0.3,
                                 win: int = 15) -> pd.DataFrame:
    df = _parse_pdb_ca_support(pdb_bytes)
    # Base inverse confidence proxy
    df["inv_conf"] = df["uncertainty_norm"].astype(float)

    # Entropy-aware proxy only for AlphaFold inputs
    if df["support_type"].iloc[0] == "plddt":
        he = _local_entropy(df["support_norm"].values, win=win)
        df["proxy_entropy"] = np.clip(alpha * df["inv_conf"].values + beta * he, 0.0, 1.0)
        cl = df["proxy_entropy"].values
    else:
        # X-ray: leave as inverse confidence only
        cl = df["inv_conf"].values

    df["cl_proxy"] = np.asarray(cl, dtype=float)
    df["rev_cl_proxy"] = 1.0 - df["cl_proxy"]
    return df

# --- New EWCL helpers for /api/analyze/main ---
def shannon_entropy(window: np.ndarray) -> float:
    hist, _ = np.histogram(window, bins=10, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log(hist)).sum()) if len(hist) else 0.0

def ewcl_from_alphafold(df: pd.DataFrame, win: int = 15) -> pd.DataFrame:
    vals = df["inv_conf"].to_numpy(dtype=float)
    ent = np.zeros_like(vals, dtype=float)
    half = win // 2
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        ent[i] = shannon_entropy(vals[lo:hi])
    df["entropy"] = ent
    df["cl_raw"] = df["inv_conf"] * df["entropy"]
    lo, hi = float(df["cl_raw"].min()), float(df["cl_raw"].max())
    df["cl_norm"] = (df["cl_raw"] - lo) / (hi - lo + 1e-6)
    return df

def ewcl_from_xray(df: pd.DataFrame) -> pd.DataFrame:
    vals = df["support"].to_numpy(dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    norm = (vals - lo) / (hi - lo + 1e-6)
    df["inv_conf"] = norm
    df["cl_norm"] = norm
    return df

# ─────────────────────────────────────────
# 3) FastAPI app + routes
# ─────────────────────────────────────────
app = FastAPI(
    title="EWCL V5 Flip-Aware Backend",
    version="2025.08",
    description="Physics-based EWCL + EWCL V5 (Flip-aware) endpoints."
)

# CORS (add your frontends here)
origins = [
    "https://ewclx.com",
    "https://www.ewclx.com",
    "https://vercel.link",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "EWCL API alive",
        "endpoints": {
            "GET /": "this message",
            "GET /health": "status + versions",
            "POST /analyze-pdb": "Physics EWCL (CA-only)",
            "POST /api/analyze/raw": "Physics-only table",
            "POST /api/analyze/main": "EWCL Proxy (AF entropy-window; X-ray B-factor)"
        }
    }

@app.get("/health")
def health():
    import sys, platform
    return {
        "status": "ok",
        "python": sys.version,
        "platform": platform.platform(),
        "version": "2025.08",
        "models": {"physics": True, "proxy": True, "ml": False}
    }

@app.post("/analyze-pdb")
async def analyze_pdb(file: UploadFile = File(...)):
    try:
        pdb_bytes = await file.read()
        obj = run_physics(pdb_bytes, bf_mode="ca")  # returns {"protein_id", "residues":[...]}
        protein_id = obj.get("protein_id", file.filename or "protein")
        df = pd.DataFrame(obj.get("residues", []))
        if not df.empty:
            df = add_rev_cl(df, "cl", "rev_cl")
            result_cols = list(df.columns)
            if "rev_cl" not in result_cols:
                result_cols.append("rev_cl")
            obj["residues"] = df[result_cols].to_dict("records")
        obj.setdefault("protein_id", protein_id)
        return JSONResponse(content=obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"physics failed: {e}")

@app.post("/api/analyze/raw")
async def analyze_raw(file: UploadFile = File(...)):
    """Physics-only EWCL with rev_cl, returns list of residue dicts."""
    try:
        pdb_bytes = await file.read()
        obj = run_physics(pdb_bytes, bf_mode="ca")
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj.get("residues", obj))
        if df.empty:
            return []
        df = add_rev_cl(df, "cl", "rev_cl")
        result_cols = [c for c in [
            "chain","position","aa","cl","rev_cl","bfactor","plddt",
            "bfactor_norm","hydro_entropy","charge_entropy",
            "bfactor_curv","bfactor_curv_entropy","bfactor_curv_flips"
        ] if c in df.columns]
        if not result_cols:
            result_cols = list(df.columns)
            if "rev_cl" not in result_cols:
                result_cols.append("rev_cl")
        return df[result_cols].to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"physics(raw) failed: {e}")

from app.routes.analyze import router as analyze_router
app.include_router(analyze_router)

# --- EWCL V5 Flip-aware ML Endpoint (CSV features) ---
from inference import predict_protein
import pandas as pd

@app.post("/analyze-ewcl-flip/")
async def analyze_ewcl_flip(file: UploadFile = File(...)):
    """
    Analyze protein features with EWCL V5 (Flip-aware Hybrid ML model).
    Input: CSV with precomputed features (columns must match model's X_cols)
    Output: JSON with protein-level + residue-level predictions
    """
    try:
        df = pd.read_csv(file.file)
        protein_id = str(df.get("uniprot", [file.filename or "protein"]) [0])
        return predict_protein(df, protein_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"EWCL V5 analysis failed: {e}")
