from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein

app = FastAPI()

# Allow all origins (for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model
model_path = os.path.join("models", "ewcl_model.pkl")
try:
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# 🔹 AI prediction endpoint
@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        X = np.array([[data["score"], data["avgEntropy"], data["minEntropy"], data["maxEntropy"]]])
        prediction = model.predict(X)[0]
        return {"collapseRisk": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

# 🔹 Real EWCL entropy map endpoint
class SequenceRequest(BaseModel):
    sequence: str

@app.post("/runeucl")
async def run_ewcl(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return result
    except Exception as e:
        return {"error": str(e)}