from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.ewcl_physics import compute_ewcl_from_pdb
import tempfile
import os
import json

app = FastAPI(title="EWCL Physics API", version="2.0")

app.add_middleware(
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

def detect_structure_type(pdb_content):
    """Detect if PDB is from AlphaFold or X-ray"""
    lines = pdb_content.split('\n')
    for line in lines:
        if "AlphaFold" in line or "ALPHAFOLD" in line:
            return "alphafold"
        if "EXPDTA" in line and "X-RAY" in line:
            return "xray"
    return "unknown"

@app.get("/")
def health_check():
    return {"status": "EWCL Physics API is live", "version": "2.0"}

@app.post("/analyze-pdb")
async def analyze_pdb(file: UploadFile = File(...)):
    """Analyze PDB file and return physics-based EWCL predictions"""
    try:
        pdb_content = (await file.read()).decode('utf-8')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        structure_type = detect_structure_type(pdb_content)
        result = compute_ewcl_from_pdb(tmp_path)
        
        response = {
            "model_type": structure_type,
            "metric_used": "pLDDT" if structure_type == "alphafold" else "B-factor",
            "residues": result,
            "n_residues": len(result)
        }
        
        os.remove(tmp_path)
        return JSONResponse(content=response)
        
    except Exception as e:
        if 'tmp_path' in locals():
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"EWCL analysis failed: {str(e)}")

@app.post("/correlate")
async def correlate_analysis(file: UploadFile = File(...)):
    """Correlate EWCL with pLDDT or B-factor"""
    try:
        pdb_content = (await file.read()).decode('utf-8')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        structure_type = detect_structure_type(pdb_content)
        ewcl_results = compute_ewcl_from_pdb(tmp_path)
        
        from scipy.stats import pearsonr, spearmanr
        
        cl_scores = [r["cl"] for r in ewcl_results]
        bfactor_scores = [r["bfactor"] for r in ewcl_results]
        
        pearson_r, pearson_p = pearsonr(cl_scores, bfactor_scores)
        spearman_rho, spearman_p = spearmanr(cl_scores, bfactor_scores)
        
        response = {
            "model_type": structure_type,
            "metric_used": "pLDDT" if structure_type == "alphafold" else "B-factor",
            "correlation": {
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p": round(float(pearson_p), 4),
                "spearman_rho": round
