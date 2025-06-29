from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.ewcl_real_model import compute_ewcl_df
from models.qwip3d import run_qwip_on_pdb, compute_qwip3d
from utils.io import save_uploaded_file, cleanup_temp_file
import pandas as pd
import json
import os
import tempfile
from Bio.PDB import PDBParser
import io

app = FastAPI(title="EWCL API", version="2.0")

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

@app.get("/")
def root():
    return {"message": "EWCL API v2.0 - Clean & Simple"}

@app.post("/analyze-pdb")
async def analyze_pdb(file: UploadFile = File(...)):
    """Analyze PDB file and return EWCL predictions"""
    tmp_path = await save_uploaded_file(file)
    
    try:
        df = compute_ewcl_df(tmp_path)
        records = df.to_dict(orient="records")
        
        return {
            "residues": records, 
            "n_residues": len(records),
            "filename": file.filename
        }
    finally:
        cleanup_temp_file(tmp_path)

@app.post("/analyze-qwip3d")
async def analyze_qwip3d(file: UploadFile = File(...)):
    """Analyze PDB file and return QWIP3D predictions with enhanced output format"""
    if not file.filename.endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Upload a .pdb file")

    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Get detailed QWIP3D results
        result = run_qwip_on_pdb(tmp_path)
        
        # Also extract chain info for enhanced response
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", io.StringIO(contents.decode()))
        
        # Extract first model, first chain
        model = structure[0]
        chain = next(model.get_chains())
        
        # Extract residue IDs and QWIP scores from result
        residue_ids = [item["residue_id"] for item in result]
        qwip_scores = [item["qwip_3d"] for item in result]
        
        if len(residue_ids) < 3:
            return {"error": "Not enough residues with Cα atoms for QWIP 3D."}
        
        return {
            "protein": file.filename.replace('.pdb', ''),
            "chain": chain.id,
            "residue_ids": residue_ids,
            "qwip_3d": qwip_scores,
            "n_residues": len(residue_ids)
        }
        
    finally:
        os.remove(tmp_path)
