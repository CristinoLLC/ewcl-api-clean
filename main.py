from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Protein Collapse Analysis API",
    version="1.0.0",
    description="Minimal API for protein entropy collapse analysis"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Health check endpoint to confirm server is alive"""
    logger.info("Health check requested")
    return {"status": "ok", "message": "Protein Collapse Analysis API is running"}

@app.post("/analyze-file")
async def analyze_pdb_file(file: UploadFile = File(...)):
    """
    Analyze a PDB file for protein entropy collapse
    
    Args:
        file: Uploaded PDB file
        
    Returns:
        JSON list of residue analysis: [{residue_id, aa, ewcl_score}, ...]
    """
    if not file.filename.endswith('.pdb'):
        logger.error(f"Invalid file type: {file.filename}")
        return JSONResponse(
            status_code=400, 
            content={"error": "Only .pdb files are supported"}
        )
    
    # Create temporary file
    temp_path = f"/tmp/{file.filename}"
    logger.info(f"Processing file: {file.filename}")
    
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Import here to avoid startup issues if model is missing
        from entropy_collapse_model import infer_entropy_from_pdb
        
        # Analyze the PDB file
        results = infer_entropy_from_pdb(temp_path)
        logger.info(f"Successfully analyzed {file.filename}: {len(results)} residues")
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return JSONResponse(content=results)
        
    except ImportError as e:
        logger.error(f"Model import error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Entropy collapse model not available"}
        )
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {e}")
        # Clean up temporary file even on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500, 
            content={"error": f"Analysis failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
