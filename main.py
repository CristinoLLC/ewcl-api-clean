from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.ewcl_physics import compute_ewcl_from_pdb
import tempfile
import os
import json

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

DISPROT_DIR = "disprot_data"  # DisProt annotations directory

def detect_structure_type(pdb_content):
    """Detect if PDB is from AlphaFold or X-ray"""
    lines = pdb_content.split('\n')
    for line in lines:
        if "AlphaFold" in line or "ALPHAFOLD" in line:
            return "alphafold"
        if "EXPDTA" in line and "X-RAY" in line:
            return "xray"
    return "unknown"

def extract_uniprot_id(pdb_content: str) -> str:
    """Extract UniProt ID from PDB content"""
    lines = pdb_content.split('\n')
    for line in lines:
        if "DBREF" in line or "UNP" in line or "UniProtKB" in line:
            parts = line.split()
            for part in parts:
                if part.startswith("P") or part.startswith("Q"):
                    return part.strip()
    
    # Try to extract from AlphaFold filename pattern
    for line in lines:
        if "AF-" in line and "-F1-model" in line:
            # Extract P/Q ID from AF-P37840-F1-model_v4 pattern
            start = line.find("AF-") + 3
            end = line.find("-", start)
            if end > start:
                potential_id = line[start:end]
                if potential_id.startswith("P") or potential_id.startswith("Q"):
                    return potential_id
    
    return None

def load_disprot_annotations(uniprot_id: str):
    """Load DisProt annotations from JSON file"""
    path = os.path.join(DISPROT_DIR, f"{uniprot_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

@app.get("/")
def health_check():
    return {"status": "EWCL Physics API is live", "version": "2.0"}

@app.post("/analyze-pdb")
async def analyze_pdb(file: UploadFile = File(...)):
    """Analyze PDB file and return physics-based EWCL predictions"""
    try:
        # Read file content
        pdb_content = (await file.read()).decode('utf-8')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        # Detect structure type
        structure_type = detect_structure_type(pdb_content)
        
        # Run physics-based EWCL model
        result = compute_ewcl_from_pdb(tmp_path)
        
        # Add metadata
        response = {
            "model_type": structure_type,
            "metric_used": "pLDDT" if structure_type == "alphafold" else "B-factor",
            "residues": result,
            "n_residues": len(result)
        }
        
        # Cleanup
        os.remove(tmp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        if 'tmp_path' in locals():
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"EWCL analysis failed: {str(e)}")

@app.post("/get-disprot-annotations")
async def get_disprot_annotations(file: UploadFile = File(...)):
    """Extract UniProt ID and return DisProt disorder annotations if available"""
    try:
        # Read PDB content
        pdb_content = (await file.read()).decode('utf-8')
        
        # Extract UniProt ID
        uniprot_id = extract_uniprot_id(pdb_content)
        
        if not uniprot_id:
            return JSONResponse(content={
                "error": "UniProt ID not found in PDB file",
                "has_disprot": False
            }, status_code=400)

        # Load DisProt annotations
        annotations = load_disprot_annotations(uniprot_id)
        
        if not annotations:
            return JSONResponse(content={
                "uniprot_id": uniprot_id,
                "has_disprot": False,
                "disprot_regions": [],
                "message": f"No DisProt annotations available for {uniprot_id}"
            })

        return JSONResponse(content={
            "uniprot_id": uniprot_id,
            "has_disprot": True,
            "disprot_regions": annotations.get("regions", []),
            "protein_name": annotations.get("name", "Unknown"),
            "total_regions": len(annotations.get("regions", []))
        })

    except Exception as e:
        return JSONResponse(content={
            "error": f"Failed to process DisProt annotations: {str(e)}",
            "has_disprot": False
        }, status_code=500)

@app.post("/analyze-pdb-with-disprot")
async def analyze_pdb_with_disprot(file: UploadFile = File(...)):
    """Combined EWCL analysis with DisProt overlay if available"""
    try:
        # Read file content
        pdb_content = (await file.read()).decode('utf-8')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        # Run EWCL analysis
        structure_type = detect_structure_type(pdb_content)
        ewcl_results = compute_ewcl_from_pdb(tmp_path)
        
        # Try to get DisProt annotations
        uniprot_id = extract_uniprot_id(pdb_content)
        disprot_data = {}
        
        if uniprot_id:
            annotations = load_disprot_annotations(uniprot_id)
            if annotations:
                disprot_data = {
                    "uniprot_id": uniprot_id,
                    "has_disprot": True,
                    "disprot_regions": annotations.get("regions", [])
                }
            else:
                disprot_data = {
                    "uniprot_id": uniprot_id,
                    "has_disprot": False,
                    "disprot_regions": []
                }
        else:
            disprot_data = {
                "uniprot_id": None,
                "has_disprot": False,
                "disprot_regions": []
            }
        
        # Combined response
        response = {
            "model_type": structure_type,
            "metric_used": "pLDDT" if structure_type == "alphafold" else "B-factor",
            "residues": ewcl_results,
            "n_residues": len(ewcl_results),
            "disprot_annotations": disprot_data
        }
        
        # Cleanup
        os.remove(tmp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        if 'tmp_path' in locals():
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/correlate")
async def correlate_plddt(file: UploadFile = File(...)):
    """Correlate EWCL with pLDDT (AlphaFold) or B-factor (X-ray)"""
    try:
        # Read file content
        pdb_content = (await file.read()).decode('utf-8')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        # Detect structure type
        structure_type = detect_structure_type(pdb_content)
        
        # Run EWCL analysis
        ewcl_results = compute_ewcl_from_pdb(tmp_path)
        
        # Calculate correlation
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        
        cl_scores = [r["cl"] for r in ewcl_results]
        bfactor_scores = [r["bfactor"] for r in ewcl_results]
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(cl_scores, bfactor_scores)
        spearman_rho, spearman_p = spearmanr(cl_scores, bfactor_scores)
        
        response = {
            "model_type": structure_type,
            "metric_used": "pLDDT" if structure_type == "alphafold" else "B-factor",
            "correlation": {
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p": round(float(pearson_p), 4),
                "spearman_rho": round
