from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
import numpy as np
import io

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

@router.post("/generate-cl-json")
async def generate_cl_json(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range")
):
    try:
        pdb_bytes = await file.read()
        pdb_str = pdb_bytes.decode()
        
        # Validate PDB string
        if len(pdb_str) < 100:
            raise HTTPException(status_code=400, detail="Invalid PDB: File too short")
        
        if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
            raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
        
        structure = parser.get_structure("u", io.StringIO(pdb_str))
        
        plddt_scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        bfactor = residue["CA"].get_bfactor()
                        plddt_scores.append(bfactor)

        # === Compute raw CL scores ===
        cl_scores = cl_model.score(np.array(plddt_scores))
        
        # === Normalize to [0, 1] per-protein if requested ===
        if normalize:
            min_score = cl_scores.min()
            max_score = cl_scores.max()
            cl_scores_norm = (cl_scores - min_score) / (max_score - min_score + 1e-8)
        else:
            cl_scores_norm = cl_scores

        response = {
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "normalized": normalize,
            "generated": datetime.utcnow().isoformat() + "Z",
            "scores": [
                {
                    "residue_id": i + 1,
                    "cl": round(score, 6),  # normalized or raw based on toggle
                    "plddt": round(plddt_scores[i], 6),
                    "b_factor": round(plddt_scores[i], 6)
                }
                for i, score in enumerate(cl_scores_norm)
            ]
        }

        return response

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid PDB: File encoding error")
    except Exception as e:
        if "Invalid PDB" in str(e):
            raise
        raise HTTPException(status_code=400, detail=f"Invalid PDB: {str(e)}")
