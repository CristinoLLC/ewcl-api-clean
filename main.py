import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class SequenceRequest(BaseModel):
    sequence: str

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

# New primary prediction endpoint
@app.post("/predict")
async def predict_entropy_score(file: UploadFile = File(...)):
    try:
        content = await file.read()
        with open("temp.pdb", "wb") as f:
            f.write(content)

        # Extract features from PDB file
        # run_entropy_pipeline removed, update required
        return {"error": "run_entropy_pipeline function is no longer available"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Legacy endpoints for backward compatibility
@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        X = np.array([[data["score"], data["avgEntropy"], data["minEntropy"], data["maxEntropy"]]])
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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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
        score = np.mean(list(ewcl_map.values()))
        min_entropy = min(ewcl_map.values())
        max_entropy = max(ewcl_map.values())

        # Run AI prediction
        if model:
            X = np.array([[score, np.mean(list(ewcl_map.values())), min_entropy, max_entropy]])
            collapse_risk = float(model.predict(X)[0])
        else:
            collapse_risk = None

        return {
            "filename": file.filename,
            "ewcl_map": ewcl_map,
            "collapse_risk": collapse_risk,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}