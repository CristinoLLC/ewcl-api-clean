from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.ewcl_physics import compute_ewcl_from_pdb
from models.hallucination_detect import compute_hallucination
import tempfile
import os
import numpy as np

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

@app.post("/analyze-hallucination")
async def analyze_hallucination(file: UploadFile = File(...)):
    """Analyze PDB file for hallucination detection using EWCL vs confidence mismatch"""
    try:
        # Read file content
        pdb_content = (await file.read()).decode('utf-8')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        # Detect structure type
        structure_type = detect_structure_type(pdb_content)
        
        # Run physics-based EWCL model (reuse existing logic)
        ewcl_results = compute_ewcl_from_pdb(tmp_path)
        
        # Compute hallucination scores
        hallucination_results = compute_hallucination(ewcl_results)
        
        # Prepare metadata
        metadata = {
            "model_type": structure_type,
            "metric_used": "pLDDT" if structure_type == "alphafold" else "B-factor",
            "filename": file.filename,
            "n_residues": len(hallucination_results),
            "hallucinated_count": sum(1 for r in hallucination_results if r.get('hallucinated', False)),
            "avg_hallucination_score": round(np.mean([r['hallucination_score'] for r in hallucination_results]), 3)
        }
        
        # Format output focusing on hallucination metrics
        hallucination_data = []
        for r in hallucination_results:
            hallucination_data.append({
                "residue_id": r["residue_id"],
                "chain": r["chain"],
                "aa": r["aa"],
                "ewcl": round(r["cl"], 3),
                "plddt_normalized": round(r["plddt_normalized"], 3),
                "curvature": round(r["curvature"], 3),
                "mismatch": round(r["mismatch"], 3),
                "hallucination_score": round(r["hallucination_score"], 3),
                "hallucinated": r["hallucinated"]
            })
        
        # Cleanup
        os.remove(tmp_path)
        
        return JSONResponse(content={
            "hallucination": hallucination_data,
            "metadata": metadata
        })
        
    except Exception as e:
        if 'tmp_path' in locals():
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Hallucination analysis failed: {str(e)}")
