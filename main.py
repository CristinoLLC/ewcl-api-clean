from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models.ewcl_real_model import compute_ewcl_df
from utils.io import save_uploaded_file, cleanup_temp_file
import pandas as pd
import json
import os

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
