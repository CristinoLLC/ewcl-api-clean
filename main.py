from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join("models", "ewcl_model.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

@app.post("/runaiinference")
async def run_inference(request: Request):
    data = await request.json()
    try:
        X = np.array([[data["score"], data["avgEntropy"], data["minEntropy"], data["maxEntropy"]]])
        prediction = model.predict(X)[0]
        return {"collapseRisk": float(prediction)}
    except Exception as e:
        return {"error": str(e)}
