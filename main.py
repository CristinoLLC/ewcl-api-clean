import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from predict import predict_hallucination
from ewcl_core import (
    compute_ewcl_scores_from_pdb,
    compute_ewcl_scores_from_alphafold_json,
    compute_ewcl_scores_from_sequence
)
import json
from pathlib import Path
from pydantic import BaseModel
from validation import validate_ewcl_scores

# Load mutation reference once
MUTATION_REF_PATH = Path("api/data/ewcl_mutation_reference.json")
if MUTATION_REF_PATH.exists():
    with open(MUTATION_REF_PATH) as f:
        mutation_reference = json.load(f)
else:
    mutation_reference = {}

# Constants
DISPROT_LABELS_DIR = "api/data/labels"

class SequenceInput(BaseModel):
    sequence: str

class ValidationRequest(BaseModel):
    ewcl_scores: Dict[int, float]
    disprot_labels: Dict[int, int]
    threshold: float = 0.297

app = FastAPI(
    title="EWCL API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ewclx.com",
        "https://www.ewclx.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def debug_routes():
    print("Available routes:")
    for route in app.routes:
        print(route.path)

@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

def extract_plddt_scores(json_data: bytes) -> Dict[str, float]:
    """Extract pLDDT scores from AlphaFold JSON output"""
    try:
        data = json.loads(json_data)
        if "plddt" in data:
            return {str(i+1): float(score) for i, score in enumerate(data["plddt"])}
        return {}
    except Exception:
        return {}

def extract_bfactors(pdb_text: str) -> Dict[str, float]:
    """Extract B-factors from PDB file"""
    bfactors = {}
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM  ') or line.startswith('HETATM'):
            try:
                res_id = line[22:26].strip()
                bfactor = float(line[60:66].strip())
                bfactors[res_id] = bfactor
            except (ValueError, IndexError):
                continue
    return bfactors

def compute_correlation_metrics(scores1: list, scores2: list) -> dict:
    """Compute Pearson and Spearman correlations between two score lists"""
    try:
        pearson_corr, pearson_p = pearsonr(scores1, scores2)
        spearman_corr, spearman_p = spearmanr(scores1, scores2)
        return {
            "pearson_correlation": round(float(pearson_corr), 4),
            "pearson_p_value": round(float(pearson_p), 4),
            "spearman_correlation": round(float(spearman_corr), 4),
            "spearman_p_value": round(float(spearman_p), 4)
        }
    except Exception:
        return {}

def evaluate_disprot(ewcl_scores: dict, disprot_labels: dict, threshold: float = 0.297):
    """Evaluate EWCL scores against DisProt labels using specified threshold"""
    matching_keys = set(ewcl_scores.keys()) & set(disprot_labels.keys())
    if not matching_keys:
        raise ValueError("No matching residue IDs found between EWCL and DisProt data")
        
    y_true = [disprot_labels[k] for k in matching_keys]
    y_scores = [ewcl_scores[k] for k in matching_keys]
    y_pred = [1 if score > threshold else 0 for score in y_scores]
    
    return {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_scores), 4),
        "matching_residues": len(matching_keys)
    }

def try_run_disprot_validation(protein_id: str, ewcl_scores: Dict[int, float]):
    label_path = os.path.join(DISPROT_LABELS_DIR, f"{protein_id}.json")
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        disprot_labels = json.load(f)
        # Convert string keys to int if needed
        disprot_labels = {int(k): v for k, v in disprot_labels.items()}

    validation_result = validate_ewcl_scores(
        ewcl_scores,
        disprot_labels,
        threshold=0.297
    )
    return validation_result

@app.post("/analyze")
async def analyze(
    data: Optional[SequenceInput] = Body(None),
    file: Optional[UploadFile] = File(None)
):
    print("DEBUG DATA:", data)
    try:
        plddt_scores = {}
        bfactors = {}
        
        if file:
            contents = await file.read()
            filename = file.filename.lower()

            if filename.endswith(".pdb"):
                pdb_text = contents.decode("utf-8")
                scores = compute_ewcl_scores_from_pdb(pdb_text)
                bfactors = extract_bfactors(pdb_text)
            elif filename.endswith(".json"):
                scores = compute_ewcl_scores_from_alphafold_json(contents)
                plddt_scores = extract_plddt_scores(contents)
            elif filename.endswith(".fasta") or filename.endswith(".fa"):
                fasta = contents.decode("utf-8")
                scores = compute_ewcl_scores_from_sequence(fasta)
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Unsupported file format"}
                )
        elif data and data.sequence:
            print("🧪 Received sequence input with length:", len(data.sequence))
            print("✅ Using sequence-based computation")
            scores = compute_ewcl_scores_from_sequence(data.sequence)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No input provided"}
            )

        # Calculate basic EWCL statistics
        ewcl_values = list(scores.values())
        mean_ewcl = round(np.mean(ewcl_values), 4)
        std_ewcl = round(np.std(ewcl_values), 4)
        max_ewcl = round(np.max(ewcl_values), 4)

        # Calculate correlations if additional data exists
        correlation_summary = {}
        
        # B-factor correlation
        if bfactors:
            common_residues = set(scores.keys()) & set(bfactors.keys())
            if common_residues:
                ewcl_scores = [scores[res] for res in common_residues]
                bfactor_scores = [bfactors[res] for res in common_residues]
                bfactor_corr = compute_correlation_metrics(ewcl_scores, bfactor_scores)
                if bfactor_corr:
                    correlation_summary["bfactor"] = bfactor_corr
                mean_bfactor = np.mean(bfactor_scores)
                std_bfactor = np.std(bfactor_scores)
            else:
                mean_bfactor = 0.0
                std_bfactor = 0.0
        else:
            mean_bfactor = 0.0
            std_bfactor = 0.0

        # pLDDT correlation
        if plddt_scores:
            common_residues = set(scores.keys()) & set(plddt_scores.keys())
            if common_residues:
                ewcl_scores = [scores[res] for res in common_residues]
                plddt_values = [plddt_scores[res] for res in common_residues]
                plddt_corr = compute_correlation_metrics(ewcl_scores, plddt_values)
                if plddt_corr:
                    correlation_summary["plddt"] = plddt_corr
                mean_plddt = np.mean(plddt_values)
                std_plddt = np.std(plddt_values)
            else:
                mean_plddt = 0.0
                std_plddt = 0.0
        else:
            mean_plddt = 0.0
            std_plddt = 0.0

        ai_label = predict_hallucination({
            "mean_ewcl": mean_ewcl,
            "std_ewcl": std_ewcl,
            "collapse_likelihood": mean_ewcl,
            "mean_plddt": mean_plddt,
            "std_plddt": std_plddt,
            "mean_bfactor": mean_bfactor,
            "std_bfactor": std_bfactor
        })

        pdb_id = "INPUT"
        if file:
            pdb_id = file.filename.split('.')[0].upper()

        mutation_map = mutation_reference.get(pdb_id, {})

        annotated_scores = []
        for res_id, score in scores.items():
            res_id_int = int(res_id)
            mutation_info = mutation_map.get(str(res_id_int)) or mutation_map.get(res_id_int)
            score_info = {
                "residue_id": res_id_int,
                "ewcl_score": score,
                "mutation": bool(mutation_info),
                "mutation_info": mutation_info if mutation_info else None
            }
            
            # Add additional scores if available
            if res_id in plddt_scores:
                score_info["plddt"] = plddt_scores[res_id]
            if res_id in bfactors:
                score_info["bfactor"] = bfactors[res_id]
                
            annotated_scores.append(score_info)

        response = {
            "scores": annotated_scores,
            "ai_label": ai_label,
            "mean_ewcl": mean_ewcl,
            "std_ewcl": std_ewcl,
            "max_ewcl": max_ewcl,
            "statistics": {
                "ewcl": {
                    "mean": mean_ewcl,
                    "std": std_ewcl,
                    "max": max_ewcl
                }
            }
        }

        if correlation_summary:
            response["correlation_summary"] = correlation_summary

        # Add DisProt validation if available
        validation_result = try_run_disprot_validation(pdb_id, {s["residue_id"]: s["ewcl_score"] for s in annotated_scores})
        if validation_result:
            response["validation"] = validation_result

        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/validate-ewcl-vs-disprot")
def validate_scores(request: ValidationRequest):
    result = validate_ewcl_scores(
        request.ewcl_scores,
        request.disprot_labels,
        request.threshold
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
