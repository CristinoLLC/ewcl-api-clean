from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, hashlib, io, pandas as pd, numpy as np
from typing import List, Dict
from Bio.PDB import PDBParser
from scipy.stats import entropy as shannon_entropy

# --- physics (keep your existing file) ---
from models.enhanced_ewcl_af import compute_curvature_features  # physics extractor

# -----------------------
# helpers (no ML here)
# -----------------------
def _md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def run_physics(pdb_bytes: bytes, bf_mode: str = "ca") -> dict:
    """Run physics extractor on a temp file (CA-only B-factor policy)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp.write(pdb_bytes); tmp.close()
    return compute_curvature_features(tmp.name, bf_mode=bf_mode)

# === CORE: parse PDB (CA-only) and extract raw 'support' ===
def _parse_pdb_ca_support(pdb_bytes: bytes) -> pd.DataFrame:
    """
    Returns a DataFrame with CA-only per-residue raw support and identity.
    AF: support= pLDDT (stored in B-factor); X-ray: support = B-factor.
    """
    text = pdb_bytes.decode("utf-8", errors="ignore")
    is_af = ("ALPHAFOLD" in text.upper()) or any(line.startswith("COMPND   2 MODEL:") and "ALPHAFOLD" in line.upper()
                                                for line in text.splitlines()[:120])

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", io.StringIO(text))

    rows = []
    for model in structure:
        # use first model only for residue-level summaries
        for chain in model:
            for res in chain:
                if "CA" not in res:
                    continue
                ca = res["CA"]
                b  = float(ca.get_bfactor())
                rows.append({
                    "chain": chain.id,
                    "position": int(res.id[1]),
                    "icode": res.id[2] if isinstance(res.id, tuple) and len(res.id) > 2 else " ",
                    "aa": res.get_resname(),
                    "support": b,            # AF: pLDDT in [0,100] ; X-ray: B-factor
                    "is_af": is_af
                })
        break  # only first model

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No CA atoms found in PDB.")

    # Normalize support to [0,1] per protein for consistent later math:
    # AF: pLDDT_norm = pLDDT / 100
    # X-ray: B_norm = min-max per protein (robust to outliers via 5th–95th percentiles)
    if df["is_af"].iloc[0]:
        df["support_norm"] = np.clip(df["support"] / 100.0, 0.0, 1.0)
        support_type = "plddt"
    else:
        lo = np.percentile(df["support"].values, 5)
        hi = np.percentile(df["support"].values, 95)
        df["support_norm"] = np.clip((df["support"] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        support_type = "bfactor"

    # Define an "uncertainty" channel (higher = more disorder) used by proxy EWCL:
    # AF: uncertainty = 1 - pLDDT_norm ; X-ray: uncertainty = B_norm
    if support_type == "plddt":
        df["uncertainty_norm"] = 1.0 - df["support_norm"]
    else:
        df["uncertainty_norm"] = df["support_norm"]

    df["support_type"] = support_type
    return df

# === HELPER: sliding-window entropy over a 1D array in [0,1] ===
def _local_entropy(x: np.ndarray, win: int = 7, bins: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=float)
    half = win // 2
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        hist, _ = np.histogram(x[s:e], bins=edges, density=True)
        out[i] = shannon_entropy(hist + 1e-12)  # avoid log(0)
    # normalize entropy to [0,1]
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out

# === PROXY DEFAULTS (lock these) ===
PROXY_ALPHA = 0.75   # nonlinearity on uncertainty
PROXY_BETA  = 0.35   # weight on local entropy
PROXY_WIN   = 7      # sliding window size for entropy

# === PROXY: compute EWCL-Proxy from uncertainty + entropy (CA-only) ===
def compute_proxy_from_pdb_bytes(pdb_bytes: bytes,
                                 alpha: float = PROXY_ALPHA,
                                 beta: float  = PROXY_BETA,
                                 win: int     = PROXY_WIN) -> pd.DataFrame:
    """
    EWCL-Proxy: a reweighting of local 'uncertainty' (disorder proxy) with a local
    entropy term. Higher cl_proxy means higher collapse likelihood (instability).
      - alpha controls how strongly uncertainty drives the score
      - beta  controls entropy contribution
    """
    df = _parse_pdb_ca_support(pdb_bytes)
    u  = df["uncertainty_norm"].values  # [0,1], higher = more disorder
    he = _local_entropy(df["support_norm"].values, win=win)

    # Combine channels — stays in [0,1] and non-linear:
    # (1) uncertainty^alpha increases contrast at the high-uncertainty end
    # (2) + beta * entropy highlights locally heterogeneous regions
    cl = np.clip((u ** alpha) + beta * he, 0.0, 1.0)

    df["cl_proxy"] = cl
    df["rev_cl_proxy"] = 1.0 - cl  # convenience for users who want the inverse
    return df

def safe_float(value):
    """Safely convert a value to a float, returning 0.0 on failure."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="EWCL Physics + Proxy API",
    version="2025.0.2",
    description="Model-free API: physics-only EWCL + proxy EWCL (entropy-weighted pLDDT/B).",
)

# CORS (adjust domains as you need)
origins = [
    "https://ewclx.com",
    "https://www.ewclx.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://v0-next-webapp-with-7ide41v6k-lucas-cristino.vercel.app",
    "https://v0-next-webapp-with-mol-git-analysis-page-backup-lucas-cristino.vercel.app",
    "https://v0-next-webapp-with-mol.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

api = APIRouter(prefix="/api")

# -----------------------
# Endpoints (model-free)
# -----------------------
@api.post("/analyze/raw")
async def analyze_raw(file: UploadFile = File(...)):
    """Physics-only EWCL analysis (CA-only)."""
    try:
        obj = run_physics(await file.read(), bf_mode="ca")
        return JSONResponse(content=obj)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/analyze/proxy")
async def analyze_proxy(file: UploadFile = File(...)):
    """
    Proxy EWCL: entropy-weighted re-interpretation of pLDDT/B (CA-only).
    Returns an envelope with residue-level fields, plus support passthrough for QC.
    """
    try:
        pdb_bytes = await file.read()
        pid = (file.filename or "input").replace(".pdb", "")
        md5 = _md5_bytes(pdb_bytes)

        df = compute_proxy_from_pdb_bytes(pdb_bytes)
        if df.empty:
            return JSONResponse(content={
                "protein_id": pid, "model_type": "proxy",
                "version": "2025.08", "input_md5": md5, "residues": []})

        support_source = df["support_type"].iloc[0]

        residues = []
        for _, r in df.iterrows():
            residues.append({
                "chain": str(r["chain"]),
                "position": int(r["position"]),
                "aa": str(r["aa"]),
                "support": float(r["support"]),              # AF: pLDDT; X-ray: B-factor
                "support_type": support_source,              # "plddt" | "bfactor"
                "support_norm": float(r["support_norm"]),    # [0,1]
                "uncertainty_norm": float(r["uncertainty_norm"]),
                "cl": float(r["cl_proxy"]),                  # EWCL-Proxy score (0..1)
                "rev_cl": float(r["rev_cl_proxy"]),          # convenience
            })

        envelope = {
            "protein_id": pid,
            "model_type": "proxy",
            "support_source": support_source,
            "version": "2025.08",
            "input_md5": md5,
            "residues": residues,
        }
        return JSONResponse(content=envelope)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Proxy analysis failed: {e}")

@app.post("/analyze-rev-ewcl")
async def analyze_reverse_ewcl(file: UploadFile = File(...)):
    """Reverse physics CL for instability highlighting: rev_cl = 1 - cl."""
    try:
        obj = run_physics(await file.read(), bf_mode="ca")
        df = pd.DataFrame(obj.get("residues", []))
        if df.empty:
            raise ValueError("No residues found in PDB.")
        
        # Apply reversed collapse logic using safe_float helper
        df["cl"] = df["cl"].apply(safe_float)
        df["rev_cl"] = 1.0 - df["cl"]

        out = []
        for _, r in df.iterrows():
            rev_cl_val = safe_float(r.get("rev_cl"))
            
            # Add instability classification (exploratory thresholds)
            if rev_cl_val > 0.7:
                instability_level = "highly_unstable"
            elif rev_cl_val > 0.5:
                instability_level = "moderately_unstable"
            else:
                instability_level = "stable"
            
            out.append({
                "chain": str(r.get("chain", "A")),
                "position": int(safe_float(r.get("position", 0))),
                "aa": str(r.get("aa", "")),
                "cl": round(safe_float(r.get("cl")), 3),
                "rev_cl": round(rev_cl_val, 3),
                "instability_level": instability_level,
                "bfactor": round(safe_float(r.get("bfactor")), 3),
                "plddt": round(safe_float(r.get("plddt", r.get("bfactor"))), 3),
                "bfactor_norm": round(safe_float(r.get("bfactor_norm")), 3),
                "hydro_entropy": round(safe_float(r.get("hydro_entropy")), 3),
                "charge_entropy": round(safe_float(r.get("charge_entropy")), 3),
                "bfactor_curv": round(safe_float(r.get("bfactor_curv")), 3),
                "bfactor_curv_entropy": round(safe_float(r.get("bfactor_curv_entropy")), 3),
                "bfactor_curv_flips": round(safe_float(r.get("bfactor_curv_flips")), 3),
                "note": "Unstable" if rev_cl_val > 0.6 else "Stable"
            })

        return JSONResponse(content={
            "status": "ok",
            "data": out,
            "summary": {
                "total_residues": len(out),
                "highly_unstable": sum(1 for r in out if r["instability_level"] == "highly_unstable"),
                "moderately_unstable": sum(1 for r in out if r["instability_level"] == "moderately_unstable"),
                "stable": len(out) - sum(1 for r in out if r["instability_level"] == "highly_unstable") - sum(1 for r in out if r["instability_level"] == "moderately_unstable"),
                "analysis_type": "reverse_ewcl",
                "description": "Entropy-based collapse likelihood inversion for instability analysis"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-pdb")
async def analyze_pdb_direct(file: UploadFile = File(...)):
    """Unified physics endpoint kept for backward compatibility."""
    try:
        obj = run_physics(await file.read(), bf_mode="ca")
        return JSONResponse(content=obj)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "EWCL Physics + Proxy API v2025.0.2", 
        "message": "Model-free API is running successfully",
        "physics_only": True,
        "proxy_defaults": {"alpha": PROXY_ALPHA, "beta": PROXY_BETA, "win": PROXY_WIN, "bf_mode": "ca"},
        "endpoints": {
            "GET /": "Health check",
            "GET /health": "Detailed health status",
            "POST /analyze-pdb": "Direct unified physics analysis endpoint",
            "POST /analyze-rev-ewcl": "Entropy-based collapse likelihood inversion for instability analysis",
            "POST /api/analyze/raw": "Physics-only EWCL (CA-only)",
            "POST /api/analyze/proxy": "EWCL-Proxy: entropy-weighted pLDDT/B-factor analysis (CA-only)",
        },
        "models_loaded": {
            "regressor": False,
            "high_model": False,
            "high_scaler": False,
            "halluc_model": False,
            "disprot_model": False,
        },
        "note": "Model-free API providing physics-based and proxy EWCL analysis only."
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2025.0.2",
        "models_loaded": False,
        "proxy_defaults": {"alpha": PROXY_ALPHA, "beta": PROXY_BETA, "win": PROXY_WIN, "bf_mode": "ca"},
        "endpoints": [
            "POST /api/analyze/raw",
            "POST /api/analyze/proxy", 
            "POST /analyze-rev-ewcl",
            "POST /analyze-pdb",
            "GET  /health",
        ],
        "physics_extractor": "READY",
        "proxy_calculator": "READY",
        "note": "Model-free physics + proxy API"
    }

# mount router
app.include_router(api)

# legacy shims (optional)
@app.post("/raw-physics/")
async def raw_physics_legacy(pdb: UploadFile = File(...)):  # maps to /api/analyze/raw
    return await analyze_raw(pdb)

@app.post("/analyze-ewcl/")
async def analyze_ewcl_legacy(pdb: UploadFile = File(...)):  # old name, same physics
    return await analyze_pdb_direct(pdb)

# disprot fallback
@app.post("/disprot-predict")
async def disprot_predict_fallback(file: UploadFile = File(...)):
    """Enhanced DisProt prediction with physics-based fallback"""
    try:
        print(f"📥 DisProt prediction for: {file.filename}")
        obj = run_physics(await file.read(), bf_mode="ca")
        df = pd.DataFrame(obj.get("residues", []))
        
        result = []
        for i, row in df.iterrows():
            cl = safe_float(row.get("cl", 0.0))
            rev_cl = 1.0 - cl
            entropy = safe_float(row.get("hydro_entropy", 0.0))
            curvature = abs(safe_float(row.get("bfactor_curv", 0.0)))
            
            # Enhanced physics-based disorder prediction
            base_disorder = (entropy * 0.4) + (rev_cl * 0.4) + (curvature * 0.2)
            
            # Apply sigmoid-like transformation for better range
            import math
            try:
                disprot_prob = 1 / (1 + math.exp(-5 * (base_disorder - 0.5)))
                disprot_prob = min(max(disprot_prob, 0.0), 1.0)
            except (OverflowError, ValueError):
                disprot_prob = 0.5
            
            result.append({
                "chain": str(row.get("chain", "A")),
                "position": int(safe_float(row.get("position", i + 1))),
                "aa": str(row.get("aa", "")),
                "rev_cl": float(rev_cl),
                "entropy": float(entropy),
                "disprot_prob": float(disprot_prob),
                "hallucination_score": 0.0  # No hallucination detection in model-free version
            })
        
        disorder_count = sum(1 for r in result if r["disprot_prob"] > 0.7)
        print(f"✅ DisProt prediction (physics-based): {disorder_count}/{len(result)} disordered")
        
        return JSONResponse(content={"results": result})
        
    except Exception as e:
        print(f"❌ DisProt prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"DisProt prediction failed: {str(e)}")
