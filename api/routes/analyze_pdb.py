import os, shutil, json
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, UploadFile, File
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
import pandas as pd
import numpy as np
import io
import logging

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

@router.post("/analyze-pdb")
async def analyze_pdb(file: UploadFile = File(...)):
    pdb_bytes = await file.read()
    structure = parser.get_structure("protein", io.StringIO(pdb_bytes.decode()))

    plddt_scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    plddt_scores.append(residue["CA"].get_bfactor())

    cl_scores = cl_model.score(np.array(plddt_scores))

    results = [
        {"residue_id": i + 1, "cl_score": round(score, 4)}
        for i, score in enumerate(cl_scores)
    ]

    return {
        "model": "CollapseLikelihood",
        "lambda": cl_model.lambda_,
        "generated": datetime.utcnow().isoformat() + "Z",
        "n_residues": len(results),
        "results": results
    }
