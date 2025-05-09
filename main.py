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
        "http://localhost:3000",
        "https://v0-ewcl-platform.vercel.app",
        "https://www.ewclx.com",
        "https://ewclx.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Request/Response models
class EWCLRequest(BaseModel):
    structure: str
    entropyMethod: str = "shannon"
    weightingFactor: float = 1.0
    temperature: float = 298.0

class SequenceRequest(BaseModel):
    sequence: str

class EWCLResponse(BaseModel):
    scores: dict
    summary: dict
    status: str = "success"

# ✅ Load ML model
model_path = os.path.join("models", "ewcl_model_final.pkl")
model = None
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")

# ✅ Health routes
@app.get("/health")
def health():
    return {"ok": True}

@app.head("/health")
def health_head():
    return JSONResponse(status_code=200, content=None)

# ✅ EWCL analysis endpoint
@app.post("/analyze", response_model=EWCLResponse)
async def analyze_structure(req: EWCLRequest):
    try:
        ewcl_map = ewcl_score_protein(req.structure)
        scores = list(ewcl_map.values())
        avg = float(np.mean(scores)) if scores else 0
        std = float(np.std(scores)) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        length = len(scores)

        collapse_risk = None
        if model:
            X = np.array([[avg, std, min_s, max_s, length]])
            collapse_risk = float(model.predict(X)[0])

        return {
            "scores": ewcl_map,
            "summary": {
                "method": req.entropyMethod,
                "mean_score": avg,
                "min_score": min_s,
                "max_score": max_s,
                "collapse_risk": collapse_risk,
                "weightingFactor": req.weightingFactor,
                "temperature": req.temperature
            },
            "status": "success"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"scores": {}, "summary": {}, "status": "error", "error": str(e)}
        )

# ✅ File upload version
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        sequence = (await file.read()).decode("utf-8").strip()
        if not sequence:
            return {"error": "Empty file", "status": "error"}

        ewcl_map = ewcl_score_protein(sequence)
        scores = list(ewcl_map.values())
        avg = float(np.mean(scores)) if scores else 0
        std = float(np.std(scores)) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        length = len(scores)

        collapse_risk = None
        if model:
            X = np.array([[avg, std, min_s, max_s, length]])
            collapse_risk = float(model.predict(X)[0])

        return {
            "filename": file.filename,
            "scores": ewcl_map,
            "summary": {
                "method": "shannon",
                "mean_score": avg,
                "min_score": min_s,
                "max_score": max_s,
                "collapse_risk": collapse_risk
            },
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}