from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

class PDBJSONInput(BaseModel):
    pdb_data: str
    normalize: bool = True

def run_ewcl_analysis(pdb_str: str, normalize: bool = True) -> dict:
    """
    Core EWCL analysis function that processes PDB string and returns results
    """
    # Validate PDB string
    if len(pdb_str) < 100:
        raise HTTPException(status_code=400, detail="Invalid PDB: File too short (less than 100 characters)")
    
    if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
        raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
    
    try:
        structure = parser.get_structure("protein", io.StringIO(pdb_str))
        
        plddt_scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        plddt_scores.append(residue["CA"].get_bfactor())
        
        if not plddt_scores:
            raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")
        
        # === Compute raw CL scores ===
        cl_scores = cl_model.score(np.array(plddt_scores))
        
        # === Normalize to [0, 1] per-protein if requested ===
        if normalize:
            min_score = cl_scores.min()
            max_score = cl_scores.max()
            cl_scores_norm = (cl_scores - min_score) / (max_score - min_score + 1e-8)  # avoid div by 0
        else:
            cl_scores_norm = cl_scores  # return raw values unchanged
        
        results = [
            {
                "residue_id": i + 1,
                "cl": round(score, 6),  # normalized or raw based on toggle
                "plddt": round(plddt_scores[i], 6),
                "b_factor": round(plddt_scores[i], 6)
            }
            for i, score in enumerate(cl_scores_norm)
        ]
        
        return {
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "normalized": normalize,
            "generated": datetime.utcnow().isoformat() + "Z",
            "n_residues": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDB: Failed to parse - {str(e)}")

@router.post("/analyze-pdb")
async def analyze_pdb(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range")
):
    """
    Upload PDB file for EWCL analysis (backward-compatible multipart endpoint)
    """
    try:
        pdb_bytes = await file.read()
        pdb_str = pdb_bytes.decode()
        return run_ewcl_analysis(pdb_str, normalize)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid PDB: File encoding error")

@router.post("/analyze-pdb-json")
async def analyze_pdb_json(input_data: PDBJSONInput):
    """
    Analyze PDB from JSON body (alternate route for JSON input)
    """
    return run_ewcl_analysis(input_data.pdb_data, input_data.normalize)
