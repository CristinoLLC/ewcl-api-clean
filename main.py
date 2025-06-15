from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from api.routes.analyze import analyze_file
from api.routes.analyze_final import analyze_final

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Check if models directory exists
MODELS_DIR = os.path.abspath("models")
if os.path.exists(MODELS_DIR):
    logging.info(f"✅ Models directory detected at {MODELS_DIR}")
else:
    logging.error(f"❌ Models directory not found at {MODELS_DIR}")

# ✅ Allow CORS from specific origins
origins = [
    "http://localhost:3000",
    "https://ewclx.com",
    "https://www.ewclx.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "EWCL API running", "message": "CORS enabled for all origins"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Protein Collapse Analysis API is running"}

@app.post("/analyze")
async def analyze_file_endpoint(file: UploadFile = File(...)):
    return await analyze_file(file)

@app.post("/analyze-final")
async def analyze_final_endpoint(file: UploadFile = File(...)):
    return await analyze_final(file)
