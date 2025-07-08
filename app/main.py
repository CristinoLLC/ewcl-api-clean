"""
FastAPI entry-point for EWCL Collapse Service
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.physics_ewcl import compute_ewcl_from_pdb
from app.ml_predict import get_refined_cl, detect_hallucinations

api = FastAPI(title="EWCL Collapse Service", version="1.0")

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
    return {"status": "EWCL Collapse Service", "version": "1.0"}

@api.post("/analyze-pdb")
async def analyze_pdb(pdb: UploadFile = File(...)):
    """Core endpoint: returns raw CL, refined CL, hallucination flags."""
    raw_bytes = await pdb.read()
    cl_df = compute_ewcl_from_pdb(raw_bytes)

    # ===== ML passes =====
    cl_df = get_refined_cl(cl_df)
    cl_df = detect_hallucinations(cl_df)

    # return as JSON (records-orient)
    return cl_df.to_dict(orient="records")
