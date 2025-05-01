from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define the input model
class SequenceRequest(BaseModel):
    sequence: str

model_path = os.path.join("models", "ewcl_model.pkl")
model = None

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load model: {e}")

@app.get("/")
def root():
    return {"message": "EWCL API is running"}

@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        X = np.array([[data["score"], data["avgEntropy"], data["minEntropy"], data["maxEntropy"]]])
        prediction = model.predict(X)[0]
        return {"collapseRisk": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

# ✅ Fix: Attach the SequenceRequest as input body
@app.post("/runeucl")
def run_ewcl(req: SequenceRequest):
    try:
        result = ewcl_score_protein(req.sequence)
        return {"ewcl_map": result}
    except Exception as e:
        return {"error": str(e)}