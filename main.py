"""
Fresh EWCL API - Completely Clean
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.ewcl_core import analyze_pdb
import tempfile
import os
import numpy as np

app = FastAPI(title="Fresh EWCL API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Fresh EWCL API v1.0", "status": "clean"}

@app.post("/analyze")
async def analyze_structure(file: UploadFile = File(...)):
    """Analyze PDB structure for collapse likelihood"""
    
    if not file.filename.endswith('.pdb'):
        raise HTTPException(status_code=400, detail="Only PDB files allowed")
    
    try:
        # Save uploaded file
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='wb') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Run analysis
        results = analyze_pdb(tmp_path)
        
        if not results:
            raise HTTPException(status_code=400, detail="No residues found")
        
        # Calculate stats
        cl_values = [r['cl'] for r in results]
        has_plddt = any(r['plddt'] is not None for r in results)
        
        response = {
            "filename": file.filename,
            "type": "alphafold" if has_plddt else "experimental",
            "residue_count": len(results),
            "avg_cl": round(np.mean(cl_values), 3),
            "max_cl": round(np.max(cl_values), 3),
            "unstable_residues": sum(1 for r in results if r['cl'] > 0.6),
            "residues": results
        }
        
        # Cleanup
        os.unlink(tmp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
