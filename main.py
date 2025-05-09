import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein


app = FastAPI(
    title="EWCL API",
    version="1.0.0",
    docs_url="/docs",         # Swagger UI
    redoc_url="/redoc",       # ReDoc UI
    openapi_url="/openapi.json"
)

# Updated origins list
origins = [
    "http://localhost:3000",
    "https://v0-ewcl-platform.vercel.app",
    "https://ewclx.com",
    "https://www.ewclx.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Request/Response models
class SequenceRequest(BaseModel):
    sequence: str

class EWCLRequest(BaseModel):
    structure: str               # PDB or sequence content
    entropyMethod: str = "shannon"
    weightingFactor: float = 1.0
    temperature: float = 298.0

# First, add this response model
class EWCLResponse(BaseModel):
    scores: dict
    summary: dict
    status: str

# Then add these endpoints
@app.post("/analyze", response_model=EWCLResponse)
async def analyze_structure(req: EWCLRequest):
    try:
        ewcl_map = ewcl_score_protein(req.structure)
        scores_list = list(ewcl_map.values())
        avg_score = float(np.mean(scores_list)) if scores_list else 0
        min_score = min(scores_list) if scores_list else 0
        max_score = max(scores_list) if scores_list else 0

        return {
            "scores": ewcl_map,
            "summary": {
                "method": req.entropyMethod,
                "mean_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "weightingFactor": req.weightingFactor,
                "temperature": req.temperature
            },
            "status": "success"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "scores": {},
                "summary": {},
                "status": "error",
                "error": str(e)
            }
        )

@app.get("/health")
def health():
    return {"ok": True}