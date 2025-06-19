from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from models.collapse_likelihood import CollapseLikelihood
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

from api.routes.analyze_pdb import router as analyze_router
from api.routes.generate_cl_json import router as cl_json_router

app = FastAPI()

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load the physics-based Collapse Likelihood model
try:
    with open("models/collapse_likelihood.pkl", "rb") as f:
        cl_model = pickle.load(f)
    logging.info(f"✅ Loaded Collapse Likelihood model with λ = {cl_model['lambda']}")
except Exception as e:
    logging.error(f"❌ Could not load CL model: {e}")

# ✅ Allow CORS from localhost and EWCL production domains
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ewclx.com",
    "https://www.ewclx.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze_router)
app.include_router(cl_json_router)

@app.get("/")
def root():
    return {"status": "EWCL API running", "message": "CORS enabled for localhost and ewclx.com"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Protein Collapse Analysis API is running"}

class CLInput(BaseModel):
    plddt: List[float]

@app.post("/predict-cl")
async def predict_cl(input: CLInput):
    λ = cl_model["lambda"]
    plddt = np.array(input.plddt)
    cl = np.exp(-λ * (1 - plddt / 100))
    return {"cl": cl.tolist()}
