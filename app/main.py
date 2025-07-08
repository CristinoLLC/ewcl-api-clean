"""
FastAPI entry-point with three EWCL endpoints
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.physics import run_physics
from app.predictors import add_main_prediction, add_high_refinement, add_hallucination

api = FastAPI(title="EWCL Collapse-Likelihood API", version="1.0")

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
    return {"status": "EWCL API", "endpoints": 3}

# ─────────── 1) raw + main regressor ───────────
@api.post("/analyze-ewcl/")
async def analyze_ewcl(pdb: UploadFile = File(...)):
    """Raw physics + main regressor prediction"""
    raw_bytes = await pdb.read()
    df = run_physics(raw_bytes)
    df = add_main_prediction(df)
    return df[["residue_id", "chain", "position", "aa", "cl", "cl_pred"]].to_dict(orient="records")

# ─────────── 2) refined (high model + scaler) ───────────
@api.post("/refined-ewcl/")
async def refined_ewcl(pdb: UploadFile = File(...)):
    """High-correlation refined prediction"""
    raw_bytes = await pdb.read()
    df = run_physics(raw_bytes)
    df = add_high_refinement(df)
    return df[["residue_id", "chain", "position", "aa", "cl", "cl_refined"]].to_dict(orient="records")

# ─────────── 3) hallucination detection ───────────
@api.post("/detect-hallucination/")
async def detect_hallucination(pdb: UploadFile = File(...)):
    """Hallucination detection using mismatch patterns"""
    raw_bytes = await pdb.read()
    df = run_physics(raw_bytes)
    df = add_main_prediction(df)    # need cl_pred for cl_diff
    df = add_hallucination(df)
    return df[["residue_id", "chain", "position", "aa", "cl", "cl_pred", "hallucination", "halluc_score"]].to_dict(orient="records")
