import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein


app = FastAPI(
    title="EWCL API",
    version="1.0.0",
    docs_url="/docs",         # Swagger UI
    redoc_url="/redoc",       # ReDoc UI
    openapi_url="/openapi.json"
)

# Updated CORS setup with specific origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://v0-next-webapp-with-mol.vercel.app",
        "https://ewcl-platform.onrender.com",
        "https://ewclbio.com",
        "https://ewclx.com",             # ✅ newly added
        "https://www.ewclx.com"          # ✅ newly added
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request/Response models
class SequenceRequest(BaseModel):
    sequence: str

class EWCLRequest(BaseModel):
    structure: str               # PDB or sequence content
    entropyMethod: str = "shannon"
    weightingFactor: float = 1.0
    temperature: float = 298.0

class EWCLResponse(BaseModel):
    scores: dict
    summary: dict
    status: str = "success"

# ✅ Load the final EWCL model
model_path = os.path.join("models", "ewcl_model_final.pkl")
model = None
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

@app.get("/health")
def health():
    return JSONResponse(status_code=200, content={"ok": True})

@app.head("/health")
def health_head():
    return JSONResponse(status_code=200, content=None)

@app.post("/")
def fallback_root():
    return JSONResponse(status_code=405, content={"error": "Use /analyze or /predict"})

# ✅ Enhanced main endpoint with Pydantic models
@app.post("/analyze", response_model=EWCLResponse)
async def analyze_structure(req: EWCLRequest):
    try:
        # Process the sequence/structure
        ewcl_map = ewcl_score_protein(req.structure)
        
        # Calculate summary statistics
        scores_list = list(ewcl_map.values())
        avg_score = float(np.mean(scores_list)) if scores_list else 0
        min_score = min(scores_list) if scores_list else 0
        max_score = max(scores_list) if scores_list else 0
        
        # Run AI prediction
        collapse_risk = None
        if model:
            # Calculate standard deviation as a feature
            std_score = float(np.std(scores_list)) if scores_list else 0
            
            # Create feature array with all required features
            X = np.array([[avg_score, std_score, min_score, max_score, len(scores_list)]])
            collapse_risk = float(model.predict(X)[0])
        
        return {
            "scores": ewcl_map,
            "summary": {
                "method": req.entropyMethod,
                "mean_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "collapse_risk": collapse_risk,
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

# ✅ File upload version of analyze
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Decode sequence
        try:
            sequence = contents.decode("utf-8").strip()
            if not sequence:
                return {"error": "Empty sequence", "status": "error"}
        except UnicodeDecodeError:
            return {"error": "Invalid file format", "status": "error"}

        # Run EWCL scoring
        ewcl_map = ewcl_score_protein(sequence)
        scores_list = list(ewcl_map.values())
        avg_score = float(np.mean(scores_list)) if scores_list else 0
        min_score = min(scores_list) if scores_list else 0
        max_score = max(scores_list) if scores_list else 0

        # Run AI prediction
        collapse_risk = None
        if model:
            # Calculate standard deviation as a feature
            std_score = float(np.std(scores_list)) if scores_list else 0
            
            # Create feature array with all required features
            X = np.array([[avg_score, std_score, min_score, max_score, len(scores_list)]])
            collapse_risk = float(model.predict(X)[0])

        return {
            "filename": file.filename,
            "scores": ewcl_map,
            "summary": {
                "method": "shannon",
                "mean_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "collapse_risk": collapse_risk
            },
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Legacy endpoints for backward compatibility
@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        # Make sure all required features are included
        X = np.array([[
            data["score"], 
            data["stdEntropy"] if "stdEntropy" in data else data["avgEntropy"],  
            data["minEntropy"], 
            data["maxEntropy"],
            data["sequenceLength"] if "sequenceLength" in data else 0
        ]])
        prediction = model.predict(X)[0]
        return {"collapseRisk": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/runeucl")
def run_ewcl(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return {"ewcl_map": result}
    except Exception as e:
        return {"error": str(e)}