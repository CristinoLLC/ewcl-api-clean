import os
import joblib
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein

# 🚀 Initialize FastAPI app
app = FastAPI(
    title="EWCL API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ✅ CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Frontend running locally on port 5173
        "http://localhost:3000",  # Frontend running locally on port 3000
        "https://v0-next-webapp-with-mol-git-main-lucas-cristino.vercel.app",  # Existing frontend
        "https://www.ewclx.com",  # Production domain
        "https://ewclx.com"       # Alternate production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ✅ Request/Response models
class EWCLRequest(BaseModel):
    filename: str

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Analysis endpoint with optional residue limit
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...), top_n: int = 100):
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        pdb_data = contents.decode("utf-8")

        # Run EWCL scoring
        result = ewcl_score_protein(pdb_data)

        # Optional filtering (return top-N highest entropy residues)
        if top_n > 0 and "ewcl_scores" in result:
            scores = result["ewcl_scores"]
            filtered = sorted(scores, key=lambda r: r["score"], reverse=True)[:top_n]
            result["filtered_scores"] = filtered

        logger.info(f"Successfully processed file: {file.filename}")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)