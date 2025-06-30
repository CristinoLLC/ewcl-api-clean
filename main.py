from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.ewcl_real_model import compute_ewcl_df
from models.qwip3d import run_qwip_on_pdb, compute_qwip3d
from utils.io import save_uploaded_file, cleanup_temp_file
from utils.qwip3d_disorder import predict_disorder
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

def normalize_scores(raw_scores):
    """Normalize scores to 0-1 range for frontend coloring"""
    min_val = min(raw_scores)
    max_val = max(raw_scores)
    if max_val == min_val:
        return [0.0 for _ in raw_scores]  # avoid div by zero
    return [(x - min_val) / (max_val - min_val) for x in raw_scores]

@app.post("/analyze-qwip3d")
async def analyze_qwip3d(file: UploadFile = File(...)):
    """Analyze PDB file and return QWIP3D predictions with B-factor and pLDDT if available"""
    if not file.filename.endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Upload a .pdb file")

    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Get detailed QWIP3D results
        result = run_qwip_on_pdb(tmp_path)
        
        # Also extract chain info and additional data for enhanced response
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", io.StringIO(contents.decode()))
        
        # Extract first model, first chain
        model = structure[0]
        chain = next(model.get_chains())
        
        # Extract residue IDs, QWIP scores, B-factors, and pLDDT
        residue_ids = [item["residue_id"] for item in result]
        qwip_scores = [item["qwip_3d"] for item in result]
        
        # Extract B-factors and pLDDT from PDB structure
        bfactors = []
        plddts = []
        
        for residue in chain:
            if "CA" in residue:
                try:
                    bfactor = residue["CA"].get_bfactor()
                    bfactors.append(round(bfactor, 2))
                    
                    # pLDDT is often stored in B-factor field for AlphaFold structures
                    # Check if B-factor values are in pLDDT range (0-100)
                    if 0 <= bfactor <= 100:
                        plddts.append(round(bfactor, 2))
                    else:
                        plddts.append(None)
                except:
                    bfactors.append(None)
                    plddts.append(None)
        
        if len(residue_ids) < 3:
            return {"error": "Not enough residues with Cα atoms for QWIP 3D."}
        
        # Build enhanced response with normalized scores for frontend coloring
        response = {
            "protein": file.filename.replace('.pdb', ''),
            "chain": chain.id,
            "residue_ids": residue_ids,
            "qwip_3d": [round(score, 3) for score in qwip_scores],
            "qwip_3d_normalized": [round(score, 3) for score in normalize_scores(qwip_scores)],
            "n_residues": len(residue_ids)
        }
        
        # Add B-factors if available
        if bfactors and any(b is not None for b in bfactors):
            response["bfactor"] = bfactors
        
        # Add pLDDT if available (and different from B-factor)
        if plddts and any(p is not None for p in plddts):
            response["plddt"] = plddts
        
        return response
        
    finally:
        os.remove(tmp_path)

@app.post("/predict-disorder")
async def predict_disorder_api(file: UploadFile = File(...)):
    """Predict disorder probability from CSV with QWIP3D features"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode()))
        
        # Make predictions
        preds = predict_disorder(df, proba=True)
        
        return {
            "predictions": preds.tolist(),
            "n_residues": len(preds),
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
