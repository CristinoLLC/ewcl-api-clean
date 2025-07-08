"""
EWCL Collapse-Likelihood API with four endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, joblib, pandas as pd, numpy as np

# ───────────────────────────────────────────
# 1)  load physics extractor
# ───────────────────────────────────────────
from models.enhanced_ewcl_af import compute_curvature_features  # physics extractor

def run_physics(pdb_bytes: bytes) -> pd.DataFrame:
    """Run the physics-only EWCL extractor on raw PDB bytes."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp.write(pdb_bytes)
    tmp.close()

    rows = compute_curvature_features(tmp.name)  # returns list[dict]
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No CA atoms found or extractor failed")

    return df

# ───────────────────────────────────────────
# 2)  load ML models & scalers
# ───────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

try:
    REGRESSOR = joblib.load(MODEL_DIR / "ewcl_regressor_model.pkl")
    HIGH_MODEL = joblib.load(MODEL_DIR / "ewcl_residue_local_high_model.pkl")
    HIGH_SCALER = joblib.load(MODEL_DIR / "ewcl_residue_local_high_scaler.pkl")
    HALLUC_MODEL = joblib.load(MODEL_DIR / "hallucination_detector_model.pkl")
except FileNotFoundError as e:
    print(f"Warning: Model file not found: {e}")
    REGRESSOR = HIGH_MODEL = HIGH_SCALER = HALLUC_MODEL = None

REG_FEATS = [
    "bfactor",
    "plddt",
    "bfactor_norm",
    "hydro_entropy",
    "charge_entropy",
    "bfactor_curv",
    "bfactor_curv_entropy",
    "bfactor_curv_flips"
]

HIGH_FEATS = [
    "bfactor",
    "plddt",
    "bfactor_norm",
    "hydro_entropy",
    "charge_entropy",
    "bfactor_curv",
    "bfactor_curv_entropy",
    "bfactor_curv_flips"
]

HAL_FEATS = [
    "cl_diff",
    "cl_diff_slope",
    "cl_diff_curv",
    "cl_diff_flips",
    "bfactor",
    "bfactor_norm",
    "hydro_entropy",
    "charge_entropy",
    "bfactor_curv",
    "bfactor_curv_entropy",
    "bfactor_curv_flips"
]

# ───────────────────────────────────────────
# 3)  FastAPI app
# ───────────────────────────────────────────
api = FastAPI(
    title="EWCL Collapse-Likelihood API",
    version="2025.0.1",
    description="Physics-based + ML refined EWCL with hallucination flags",
)

# CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.ewclx.com",
        "https://ewclx.com", 
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/")
def health_check():
    return {
        "status": "EWCL API v2025.0.1", 
        "endpoints": {
            "raw-physics": "Physics-only EWCL (no ML)",
            "analyze-ewcl": "Physics + main regressor",
            "refined-ewcl": "High-confidence refiner",
            "detect-hallucination": "Hallucination detection"
        },
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "high_model": HIGH_MODEL is not None,
            "high_scaler": HIGH_SCALER is not None,
            "halluc_model": HALLUC_MODEL is not None
        }
    }

@api.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "refiner": HIGH_MODEL is not None,
            "hallucination": HALLUC_MODEL is not None,
            "scaler": HIGH_SCALER is not None
        },
        "version": "2025.0.1"
    }

# ─────────────  ENDPOINT 1  ─────────────
@api.post("/raw-physics/")
async def raw_physics(pdb: UploadFile = File(...)):
    """
    Return physics-only EWCL (`cl`) with chain, position, and aa.
    """
    try:
        df = run_physics(await pdb.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(df[["chain", "position", "aa", "cl"]].to_dict("records"))

# ─────────────  ENDPOINT 2  ─────────────
@api.post("/analyze-ewcl/")
async def analyze_ewcl(pdb: UploadFile = File(...)):
    """
    Physics + main regressor (`cl_pred`).
    Uses: ewcl_regressor_model.pkl with 8 features
    """
    try:
        df = run_physics(await pdb.read())
        if REGRESSOR is None:
            raise HTTPException(status_code=503, detail="Regressor model not available")
        
        # Ensure plddt column exists for regressor features
        if "plddt" not in df.columns:
            df["plddt"] = df["bfactor"]  # Copy for consistency with training
        
        # Use exact features the model was trained on
        features = ['bfactor', 'plddt', 'bfactor_norm', 'hydro_entropy', 'charge_entropy',
                   'bfactor_curv', 'bfactor_curv_entropy', 'bfactor_curv_flips']
        
        X = df[features]
        df["cl_pred"] = REGRESSOR.predict(X)
        
        return JSONResponse(
            df[["chain", "position", "aa", "cl", "cl_pred"]].to_dict("records")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────  ENDPOINT 3  ─────────────
@api.post("/refined-ewcl/")
async def refined_ewcl(pdb: UploadFile = File(...)):
    """
    Extra-trust refiner (`cl_refined`) using the high-correlation model.
    """
    try:
        df = run_physics(await pdb.read())
        if HIGH_MODEL is None or HIGH_SCALER is None:
            raise HTTPException(status_code=503, detail="High refinement models not available")
        
        # Add plddt column for high model features
        df["plddt"] = df["bfactor"]  # Copy for consistency with training
        
        X = df[HIGH_FEATS]
        X_scaled = HIGH_SCALER.transform(X)
        df["cl_refined"] = HIGH_MODEL.predict(X_scaled)
        
        return JSONResponse(
            df[["chain", "position", "aa", "cl", "cl_refined"]].to_dict("records")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────  ENDPOINT 4  ─────────────
@api.post("/detect-hallucination/")
async def detect_hallucination(pdb: UploadFile = File(...)):
    """
    Flags hallucinated residues from combined physics + ML features.
    """
    try:
        df = run_physics(await pdb.read())
        if REGRESSOR is None or HALLUC_MODEL is None:
            raise HTTPException(status_code=503, detail="Hallucination detection models not available")
        
        # Add plddt column for regressor features
        df["plddt"] = df["bfactor"]  # Copy for consistency with training
        
        df["cl_pred"] = REGRESSOR.predict(df[REG_FEATS])

        #  Compute diff + derivatives for hallucination features
        df["cl_diff"] = (df["cl_pred"] - df["cl"]).abs()
        df["cl_diff_slope"] = np.gradient(df["cl_diff"])
        df["cl_diff_curv"]  = np.gradient(df["cl_diff_slope"])
        df["cl_diff_flips"] = (
            pd.Series(np.sign(df["cl_diff_slope"])).diff().abs().fillna(0)
        )

        #  Classify using correct feature set
        X = df[HAL_FEATS]
        df["hallucination"] = HALLUC_MODEL.predict(X)
        df["halluc_score"]  = HALLUC_MODEL.predict_proba(X)[:, 1]

        return JSONResponse(
            df[["chain", "position", "aa", "cl", "cl_pred",
                "hallucination", "halluc_score"]].to_dict("records")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
