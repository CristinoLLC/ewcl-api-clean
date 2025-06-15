from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging

from routes.analyze import analyze_file
from routes.analyze_rev import analyze_reverse
from routes.analyze_regressor import analyze_regression
from routes.analyze_final import analyze_final

app = FastAPI()
logging.basicConfig(level=logging.INFO)

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

# Route using the new analyze_file handler
@app.post("/analyze")
async def analyze_file_endpoint(file: UploadFile = File(...)):
    return await analyze_file(file)

# Route using the new analyze_reverse handler
@app.post("/analyze-rev")
async def analyze_rev_endpoint(file: UploadFile = File(...)):
    return await analyze_reverse(file)

@app.post("/analyze-regression")
async def analyze_regression_endpoint(file: UploadFile = File(...)):
    return await analyze_regression(file)

@app.post("/analyze-final")
async def analyze_final_endpoint(file: UploadFile = File(...)):
    return await analyze_final(file)
