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