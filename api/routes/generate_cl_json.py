from fastapi import APIRouter, UploadFile, File, HTTPException
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
import numpy as np
import io

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

@router.post("/generate-cl-json")
async def generate_cl_json(file: UploadFile = File(...)):
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

        cl_scores = cl_model.score(np.array(plddt_scores))

        response = {
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "generated": datetime.utcnow().isoformat() + "Z",
            "scores": [
                {
                    "residue_id": i + 1,
                    "cl": round(score, 4),
                    "plddt": plddt_scores[i],
                    "b_factor": plddt_scores[i]
                }
                for i, score in enumerate(cl_scores)
            ]
        }

        return response

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid PDB: File encoding error")
    except Exception as e:
        if "Invalid PDB" in str(e):
            raise
        raise HTTPException(status_code=400, detail=f"Invalid PDB: {str(e)}")
