from fastapi import APIRouter, UploadFile, File
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
    pdb_bytes = await file.read()
    structure = parser.get_structure("u", io.StringIO(pdb_bytes.decode()))

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
            {"residue_id": i + 1, "cl": round(score, 4)}
            for i, score in enumerate(cl_scores)
        ]
    }

    return response
