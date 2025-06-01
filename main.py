import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import numpy as np
from predict import predict_hallucination
from ewcl_core import (
    compute_ewcl_scores_from_pdb,
    compute_ewcl_scores_from_alphafold_json,
    compute_ewcl_scores_from_sequence
)
import json
from pathlib import Path

# Load mutation reference once
MUTATION_REF_PATH = Path("api/data/ewcl_mutation_reference.json")
if MUTATION_REF_PATH.exists():
    with open(MUTATION_REF_PATH) as f:
        mutation_reference = json.load(f)
else:
    mutation_reference = {}

from pydantic import BaseModel

class SequenceInput(BaseModel):
    sequence: str


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

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app",
    "https://*.v0.dev",
    "https://ewclx.com",
    "https://www.ewclx.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import Body

@app.post("/analyze")
async def analyze(data: Optional[SequenceInput] = None, file: Optional[UploadFile] = File(None)):
    try:
        if file:
            contents = await file.read()
            filename = file.filename.lower()

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
        elif data and data.sequence:
            scores = compute_ewcl_scores_from_sequence(data.sequence)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No input provided"}
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

        pdb_id = "INPUT"
        if file:
            pdb_id = file.filename.split('.')[0].upper()

        mutation_map = mutation_reference.get(pdb_id, {})

        annotated_scores = []
        for res_id, score in scores.items():
            res_id_int = int(res_id)
            mutation_info = mutation_map.get(str(res_id_int)) or mutation_map.get(res_id_int)
            annotated_scores.append({
                "residue_id": res_id_int,
                "ewcl_score": score,
                "mutation": bool(mutation_info),
                "mutation_info": mutation_info if mutation_info else None
            })

        return {
            "scores": annotated_scores,
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

# Refolding model temporarily disabled
refold_model = None

# Commented out model loading code as models are now handled in predict.py
# try:
#     refold_model = load("models/refolding_classifier_model.pkl")
#     print("✅ Refolding model loaded successfully.")
# except Exception as e:
#     print(f"❌ Error loading refolding model: {e}")
#     refold_model = None

# from joblib import load
# halluc_model = load("models/hallucination_safe_model_v5000.pkl")  # Or similar

# New endpoint for hallucination prediction
@app.post("/predict-hallucination")
def predict(data: CollapseFeatures):
    return {"message": "Hallucination prediction temporarily disabled"}


@app.post("/predict-refolding")
def predict_refolding(data: CollapseFeatures):
    return {"message": "Refolding prediction temporarily disabled"}


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
