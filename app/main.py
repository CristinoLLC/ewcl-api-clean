"""
Clean FastAPI with three EWCL endpoints only
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
    return {"status": "EWCL API", "endpoints": ["analyze-ewcl", "detect-hallucination", "refined-ewcl"]}

@api.post("/analyze-ewcl/")
async def analyze_ewcl(pdb: UploadFile = File(...)):
    """
    Predicts collapse likelihood from physics-based features
    Uses: ewcl_regressor_model.pkl
    """
    raw_bytes = await pdb.read()
    df = run_physics(raw_bytes)
    df = add_main_prediction(df)
    
    # Clean output format
    output_cols = ["position", "cl", "chain", "aa"]
    if "cl_pred" in df.columns:
        output_cols.append("cl_pred")
    
    return df[output_cols].to_dict(orient="records")

@api.post("/detect-hallucination/")
async def detect_hallucination(pdb: UploadFile = File(...)):
    """
    Flags residues where cl_model diverges from expected physics
    Uses: hallucination_detector_model.pkl
    """
    raw_bytes = await pdb.read()
    df = run_physics(raw_bytes)
    df = add_main_prediction(df)
    df = add_hallucination(df)
    
    # Format output for hallucination detection
    result = []
    for _, row in df.iterrows():
        result.append({
            "position": int(row["position"]),
            "hallucination": int(row["hallucination"]),
            "probability": round(float(row["halluc_score"]), 3),
            "chain": row["chain"],
            "aa": row["aa"]
        })
    
    return result

@api.post("/refined-ewcl/")
async def refined_ewcl(pdb: UploadFile = File(...)):
    """
    More confident scoring for highly correlated residues
    Uses: ewcl_residue_local_high_model.pkl + ewcl_residue_local_high_scaler.pkl
    """
    raw_bytes = await pdb.read()
    df = run_physics(raw_bytes)
    df = add_high_refinement(df)
    
    # Format output for refined predictions
    result = []
    for _, row in df.iterrows():
        result.append({
            "position": int(row["position"]),
            "refined_cl": round(float(row["cl_refined"]), 3),
            "chain": row["chain"],
            "aa": row["aa"]
        })
    
    return result
