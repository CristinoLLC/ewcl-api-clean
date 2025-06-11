from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# DisProt protein IDs for reverse mode validation
DISPROT_IDS = set([
    "P37840", "P62326", "P0DTC2", "P04637", "P02666", "P10636", 
    "P35408", "P68871", "P21359", "Q15796"  # Add more as needed
])

def is_disprot_protein(uniprot_id: str) -> bool:
    """Check if protein ID is in DisProt database"""
    return uniprot_id.upper() in DISPROT_IDS

@app.get("/health")
def health_check():
    """Health check endpoint to confirm server is alive"""
    logger.info("Health check requested")
    return {"status": "ok", "message": "Protein Collapse Analysis API is running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Normal mode analysis for structured proteins
    Collapse = high entropy (ewcl_score > 0.7)
    """
    if not file.filename.endswith('.pdb'):
        logger.error(f"Invalid file type: {file.filename}")
        return JSONResponse(
            status_code=400, 
            content={"error": "Only .pdb files are supported"}
        )
    
    temp_path = f"/tmp/{uuid.uuid4()}.pdb"
    logger.info(f"Processing file in normal mode: {file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        from entropy_collapse_model import infer_entropy_from_pdb
        
        results = infer_entropy_from_pdb(temp_path)
        
        # Process results for normal mode
        for r in results:
            score = r["ewcl_score"]
            r["ewcl_score"] = round(score, 6)
            r["collapse"] = int(score > 0.7)  # Normal mode: collapse = high entropy
        
        os.remove(temp_path)
        logger.info(f"Successfully analyzed {file.filename} in normal mode: {len(results)} residues")
        
        return {
            "status": "ok", 
            "mode": "normal",
            "reverse": False, 
            "results": results
        }
        
    except ImportError as e:
        logger.error(f"Model import error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Entropy collapse model not available"}
        )
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500, 
            content={"error": f"Analysis failed: {str(e)}"}
        )

@app.post("/analyze-rev")
async def analyze_reverse(file: UploadFile = File(...)):
    """
    Reverse mode analysis for disordered proteins (DisProt)
    Collapse = low entropy (1 - ewcl_score < 0.3)
    """
    if not file.filename.endswith('.pdb'):
        logger.error(f"Invalid file type: {file.filename}")
        return JSONResponse(
            status_code=400, 
            content={"error": "Only .pdb files are supported"}
        )
    
    temp_path = f"/tmp/{uuid.uuid4()}.pdb"
    logger.info(f"Processing file in reverse mode: {file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        from entropy_collapse_model import infer_entropy_from_pdb
        
        results = infer_entropy_from_pdb(temp_path)
        
        # Extract protein ID from filename for DisProt validation
        protein_id = file.filename.split('.')[0].upper()
        is_disprot = is_disprot_protein(protein_id)
        
        # Process results for reverse mode
        for r in results:
            score = 1.0 - r["ewcl_score"]  # Invert the signal
            r["ewcl_score"] = round(score, 6)
            r["collapse"] = int(score < 0.3)  # Reverse mode: collapse = low entropy
        
        os.remove(temp_path)
        logger.info(f"Successfully analyzed {file.filename} in reverse mode: {len(results)} residues")
        
        response = {
            "status": "ok", 
            "mode": "reverse",
            "reverse": True, 
            "results": results
        }
        
        # Add warning if protein not in DisProt
        if not is_disprot:
            response["warning"] = "Protein ID not found in DisProt list. Reverse mode results may be unreliable."
            logger.warning(f"Reverse mode used for non-DisProt protein: {protein_id}")
        
        return response
        
    except ImportError as e:
        logger.error(f"Model import error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Entropy collapse model not available"}
        )
    except Exception as e:
        logger.error(f"Reverse analysis failed for {file.filename}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500, 
            content={"error": f"Analysis failed: {str(e)}"}
        )

# Keep the original analyze-file endpoint for backward compatibility
@app.post("/analyze-file")
async def analyze_pdb_file(file: UploadFile = File(...)):
    """
    Legacy endpoint - redirects to normal analyze mode
    """
    logger.info(f"Legacy endpoint called, redirecting to normal mode")
    return await analyze(file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
