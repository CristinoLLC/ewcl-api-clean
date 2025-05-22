import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import numpy as np
from joblib import load
from predict import predict_hallucination
from ewcl_core import (
    compute_ewcl_scores_from_pdb,
    compute_ewcl_scores_from_alphafold_json,
    compute_ewcl_scores_from_sequence
)

from pydantic import BaseModel


# Pydantic model for hallucination prediction endpoint
class CollapseFeatures(BaseModel):
    mean_ewcl: float
    std_ewcl: float
    collapse_likelihood: float
    mean_plddt: float
    std_plddt: float
    mean_bfactor: float
    std_bfactor: float

app = FastAPI(
    title="EWCL API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.on_event("startup")
async def debug_routes():
    print("Available routes:")
    for route in app.routes:
        print(route.path)

from fastapi import Request

# Custom CORS middleware using regex_origin
@app.middleware("http")
async def custom_cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    response = await call_next(request)
    if origin and regex_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Vary"] = "Origin"
    return response

@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".pdb"):
            pdb_text = contents.decode("utf-8")
            scores = compute_ewcl_scores_from_pdb(pdb_text)
        elif filename.endswith(".json"):
            scores = compute_ewcl_scores_from_alphafold_json(contents)
        elif filename.endswith(".fasta") or filename.endswith(".fa"):
            fasta = contents.decode("utf-8")
            scores = compute_ewcl_scores_from_sequence(fasta)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format"}
            )

        mean_ewcl = round(np.mean(list(scores.values())), 4)
        std_ewcl = round(np.std(list(scores.values())), 4)
        max_ewcl = round(np.max(list(scores.values())), 4)

        ai_label = predict_hallucination({
            "mean_ewcl": mean_ewcl,
            "std_ewcl": std_ewcl,
            "collapse_likelihood": mean_ewcl,
            "mean_plddt": 0.0,
            "std_plddt": 0.0,
            "mean_bfactor": 0.0,
            "std_bfactor": 0.0
        })

        return {
            "scores": scores,
            "ai_label": ai_label,
            "mean_ewcl": mean_ewcl,
            "std_ewcl": std_ewcl,
            "max_ewcl": max_ewcl
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Utility function for regex-based CORS origin checking (not used by default)
import re
def regex_origin(origin: str) -> bool:
    allowed_patterns = [
        r"https://ewclx\.com$",
        r"https://www\.ewclx\.com$",
        r"https://.*\.vercel\.app$",
        r"https://.*\.v0\.dev$",
        r"http://localhost:3000$",
        r"http://127\.0\.0\.1:3000$"
    ]
    return any(re.match(pattern, origin) for pattern in allowed_patterns)

try:
    refold_model = load("models/refolding_classifier_model.pkl")
    print("✅ Refolding model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading refolding model: {e}")
    refold_model = None

# New endpoint for hallucination prediction
@app.post("/predict-hallucination")
def predict(data: CollapseFeatures):
    result = predict_hallucination(data.dict())
    return {"hallucination_risk": result}

@app.post("/predict-refolding")
def predict_refolding(data: CollapseFeatures):
    print("✅ /predict-refolding endpoint hit")
    if refold_model is None:
        return JSONResponse(status_code=500, content={"error": "Refolding model not loaded"})
    X = np.array([[data.mean_ewcl, data.std_ewcl, data.collapse_likelihood, data.mean_plddt, data.std_plddt, data.mean_bfactor, data.std_bfactor]])
    result = refold_model.predict(X)[0]
    return {"refolding_risk": result}


# New endpoint: validate EWCL vs DisProt
import json
from fastapi import UploadFile, File

@app.post("/validate-ewcl-vs-disprot")
async def validate_ewcl_vs_disprot(ewcl: UploadFile = File(...), disprot: UploadFile = File(...)):
    ewcl_data = json.loads((await ewcl.read()).decode())
    disprot_data = json.loads((await disprot.read()).decode())

    # Align keys
    common_keys = set(ewcl_data.keys()).intersection(disprot_data.keys())
    if not common_keys:
        return JSONResponse(status_code=400, content={"error": "No overlapping residues found"})

    y_scores = [ewcl_data[k] for k in common_keys]
    y_labels = [disprot_data[k] for k in common_keys]

    try:
        auc = roc_auc_score(y_labels, y_scores)
        precision = precision_score(y_labels, [int(s > 0.5) for s in y_scores])
        recall = recall_score(y_labels, [int(s > 0.5) for s in y_scores])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "common_residues": len(common_keys),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }
