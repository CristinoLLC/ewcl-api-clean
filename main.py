from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routes.analyze import analyze_file
from api.routes.analyze_rev import analyze_reverse

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
