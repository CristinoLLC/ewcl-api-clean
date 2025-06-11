from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from entropy_collapse_model import infer_entropy_from_pdb
import shutil
import os
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Protein Collapse Analysis API",
    version="1.0.0",
    description="Minimal API for protein entropy collapse analysis with normal and reverse modes"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    if not file.filename.endswith('.pdb'):
        raise HTTPException(status_code=400, detail="Only .pdb files are supported")
    
    temp_path = f"/tmp/{uuid.uuid4()}.pdb"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path

@app.get("/health")
def health_check():
    """Health check endpoint to confirm server is alive"""
    logger.info("Health check requested")
    return {"status": "ok", "message": "Protein Collapse Analysis API is running"}

@app.post("/analyze")
async def analyze_protein(file: UploadFile = File(...)):
    """Normal mode analysis for structured proteins"""
    logger.info(f"Processing file in normal mode: {file.filename}")
    
    pdb_path = save_uploaded_file(file)
    try:
        results = infer_entropy_from_pdb(pdb_path)
        os.remove(pdb_path)
        
        logger.info(f"✅ Successfully analyzed {file.filename} in normal mode: {len(results)} residues")
        return {
            "status": "ok",
            "mode": "normal",
            "results": results
        }
    except Exception as e:
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-rev")
async def analyze_protein_reverse(file: UploadFile = File(...)):
    """Reverse mode analysis for disordered proteins (DisProt)"""
    logger.info(f"Processing file in reverse mode: {file.filename}")
    
    pdb_path = save_uploaded_file(file)
    try:
        results = infer_entropy_from_pdb(pdb_path, reverse=True)
        os.remove(pdb_path)
        
        logger.info(f"✅ Successfully analyzed {file.filename} in reverse mode: {len(results)} residues")
        return {
            "status": "ok",
            "mode": "reverse",
            "reverse": True,
            "results": results
        }
    except Exception as e:
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
        logger.error(f"❌ Reverse analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
