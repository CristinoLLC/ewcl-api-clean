import os
import joblib
import numpy as np
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
        "https://v0-next-webapp-with-mol-git-main-lucas-cristino.vercel.app/",  # Existing frontend
        "https://www.ewclx.com",  # Production domain
        "https://ewclx.com"       # Alternate production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Request/Response models
class EWCLRequest(BaseModel):
    pass  # You can define fields like sequence: str or structure: dict here later

# Other routes and logic...

# ✅ Health check route
@app.get("/health")
async def health_check():
    return {"status": "ok"}