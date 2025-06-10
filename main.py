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
from typing import Optional, Dict, List
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import joblib
from predict import predict_hallucination
from ewcl_core import (
    compute_ewcl_scores_from_pdb,
    compute_ewcl_scores_from_alphafold_json,
    compute_ewcl_scores_from_sequence,
    classify_disorder,
    compute_ewcl_api_response
)
import json
from pathlib import Path
from pydantic import BaseModel
from validation import validate_ewcl_scores
from model_loader import EntropyCollapseModel

# ✅ Load the unified model only
try:
    model = EntropyCollapseModel(model_path="models/unified_entropy_model.pkl")
    print("🧠 Unified Entropy Model loaded successfully from: models/unified_entropy_model.pkl")
    unified_model = model.model  # Keep backward compatibility
    entropy_collapse_model = model
except Exception as e:
    print(f"⚠️ Could not load unified entropy model: {e}")
    model = None
    unified_model = None
    entropy_collapse_model = None

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

class CollapseAnalysisRequest(BaseModel):
    b_factor: List[float]
    plddt: List[float]

class StaticInput(BaseModel):
    input: str

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
        raise ValueError(f"No matching residue IDs found between EWCL and DisProt data (input size: {len(ewcl_scores)}, DisProt size: {len(disprot_labels)})")
        
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
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    sequence = None
    normalize = True  # Default to normalized mode
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            sequence = data.get("sequence", None)
            normalize = data.get("normalize", True)  # Accept normalize flag from frontend
            # Also check for pdb_id and pdb_text for backward compatibility
            pdb_id = data.get("pdb_id")
            pdb_text = data.get("pdb_text")
    except Exception as e:
        print("❌ JSON parse error:", e)
        
    print("DEBUG DATA (sequence):", sequence)
    print("🎛️ Normalization mode:", "enabled" if normalize else "disabled")
    
    # Handle legacy payload format
    if 'pdb_id' in locals() and 'pdb_text' in locals() and pdb_text and pdb_id:
        print(f"🧪 Received request for {pdb_id}")
        scores, metadata = compute_ewcl_scores_from_pdb(pdb_text, return_metadata=True)
        
        if not scores:
            print(f"⚠️ EWCL returned empty scores for {pdb_id}")
            return JSONResponse(content={pdb_id: {}}, status_code=200)
            
        print(f"✅ EWCL returned {len(scores)} residues for {pdb_id}")
        
        response = {
            "ewcl_score": scores,  # normalized [0, 1] entropy-based score
            "b_factor": metadata.get("b_factor", {}),  # optional raw values
            "plddt": metadata.get("plddt", {}),        # optional AlphaFold scores
            "residue_ids": metadata.get("residue_ids", list(scores.keys()))
        }
        
        return JSONResponse(content={pdb_id: response}, status_code=200)
    
    try:
        plddt_scores = {}
        bfactors = {}
        source_type = "unknown"
        has_structure = False
        
        if file:
            contents = await file.read()
            filename = file.filename.lower()
            print(f"🧪 Processing file: {filename}")

            if filename.endswith(".pdb"):
                pdb_text = contents.decode("utf-8")
                print("✅ Using PDB-based computation")
                scores = compute_ewcl_scores_from_pdb(pdb_text)
                bfactors = extract_bfactors(pdb_text)
                source_type = "pdb"
                has_structure = True
                
            elif filename.endswith(".json"):
                print("✅ Using AlphaFold JSON computation")
                scores = compute_ewcl_scores_from_alphafold_json(contents)
                plddt_scores = extract_plddt_scores(contents)
                source_type = "alphafold_json"
                has_structure = True
                
            elif filename.endswith(".fasta") or filename.endswith(".fa"):
                fasta = contents.decode("utf-8")
                print("✅ Using FASTA sequence computation")
                scores = compute_ewcl_scores_from_sequence(fasta)
                source_type = "fasta"
                has_structure = False
                
            else:
                print(f"❌ Unsupported file format: {filename}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Unsupported file format"}
                )
        elif sequence:
            print("🧪 Received sequence input with length:", len(sequence))
            print("✅ Using sequence-based computation")
            scores = compute_ewcl_scores_from_sequence(sequence)
            source_type = "sequence"
            has_structure = False
        else:
            print("❌ No input provided")
            return JSONResponse(
                status_code=400,
                content={"error": "No input provided"}
            )

        if not scores:
            print(f"⚠️ EWCL computation returned empty scores for {source_type}")
            return JSONResponse(
                status_code=200,
                content={"error": "No EWCL scores computed", "source_type": source_type}
            )
            
        print(f"✅ EWCL returned {len(scores)} residues for {source_type}")
        print("✅ Using unified entropy model to predict EWCL")

        # Calculate basic EWCL statistics
        ewcl_values = list(scores.values())
        min_score = min(ewcl_values)
        max_score = max(ewcl_values)
        
        # Store raw EWCL range for export
        min_ewcl_raw = round(min_score, 4)
        max_ewcl_raw = round(max_score, 4)

        # 🔁 Forced min-max normalization - always normalize regardless of range
        scores_normalized = {
            k: round((v - min_score) / (max_score - min_score + 1e-6), 4)
            for k, v in scores.items()
        }

        # Calculate statistics using normalized scores
        normalized_values = list(scores_normalized.values())
        mean_ewcl = round(np.mean(normalized_values), 4)
        std_ewcl = round(np.std(normalized_values), 4)
        max_ewcl = round(np.max(normalized_values), 4)

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

        # Predict disorder hallucination likelihood using key EWCL/structure statistics
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
            
            # Use raw score for absolute thresholding to avoid artificial inflation
            raw_score = score
            
            # Classify disorder based on raw EWCL score (absolute thresholding)
            if raw_score > 0.7:
                disorder_class = "disordered"
            elif raw_score < 0.3:
                disorder_class = "ordered"
            else:
                disorder_class = "intermediate"
            
            # Still provide normalized score for visualization purposes
            normalized_score = scores_normalized.get(res_id_int, 0)
            
            score_info = {
                "residue_id": res_id_int,
                "ewcl_score_raw": raw_score,
                "ewcl_score": normalized_score,
                "disorder_class": disorder_class,
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
            "min_ewcl_raw": min_ewcl_raw,
            "max_ewcl_raw": max_ewcl_raw,
            "statistics": {
                "ewcl": {
                    "mean": mean_ewcl,
                    "std": std_ewcl,
                    "max": max_ewcl
                }
            },
            "has_structure": has_structure,
            "source_type": source_type,
            "normalization": "minmax_0_1",
            "score_origin": source_type
        }

        if correlation_summary:
            response["correlation_summary"] = correlation_summary

        # Add DisProt mode classification
        disprot_mode = "none"
        
        # Add DisProt validation if available
        validation_result = try_run_disprot_validation(pdb_id, {s["residue_id"]: s["ewcl_score"] for s in annotated_scores})
        if validation_result:
            disprot_mode = "validated"
            response["validation"] = validation_result
            
        response["disprot_mode"] = disprot_mode

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

@app.post("/analyze-collapse")
async def analyze_collapse(request: CollapseAnalysisRequest):
    """
    Use unified entropy model to analyze collapse risk from B-factor and pLDDT inputs
    """
    if not unified_model:
        return JSONResponse(
            status_code=503,
            content={"error": "Unified entropy model not available"}
        )
    
    try:
        # Convert inputs to pandas Series as expected by the model
        bf_series = pd.Series(request.b_factor)
        plddt_series = pd.Series(request.plddt)
        
        # Use the model's analyze method
        df = unified_model.analyze(bf_series, plddt_series)
        
        # Convert DataFrame to records for JSON response
        result = df.to_dict(orient="records")
        
        return {
            "status": "success",
            "analysis": result,
            "model_version": "unified_entropy_model",
            "input_length": {
                "b_factor": len(request.b_factor),
                "plddt": len(request.plddt)
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model analysis failed: {str(e)}"}
        )

@app.post("/static-entropy")
def static_entropy_route(payload: StaticInput):
    """
    Compute EWCL scores using the static entropy tool for simple sequence analysis
    """
    try:
        from ewcl_toolkit.ewcl_static_tool import ewcl_score_protein, parse_fasta
        
        # Handle both raw sequence and FASTA format
        input_text = payload.input.strip()
        if input_text.startswith(">"):
            # FASTA format
            sequence = parse_fasta(input_text)
        else:
            # Raw sequence
            sequence = input_text.replace("\n", "").replace(" ", "")
        
        if not sequence:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid sequence provided"}
            )
        
        scores = ewcl_score_protein(sequence)
        classes = {
            res_id: "Disordered" if score >= 0.8 else "Medium" if score >= 0.4 else "Ordered"
            for res_id, score in scores.items()
        }
        
        return {
            "scores": scores,
            "classes": classes,
            "metadata": {
                "sequence_length": len(sequence),
                "method": "static_entropy",
                "normalization": "global_frequency"
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Static entropy computation failed: {str(e)}"}
        )

# Manual test function for debugging
def test_ewcl_function():
    """Test EWCL computation locally before running the API"""
    test_pdb_path = "test.pdb"
    
    if os.path.exists(test_pdb_path):
        print(f"🧪 Testing EWCL computation with {test_pdb_path}")
        with open(test_pdb_path) as f:
            pdb_text = f.read()
        
        scores = compute_ewcl_scores_from_pdb(pdb_text)
        print(f"Local test result: {len(scores)} residues computed")
        print(f"Sample scores: {dict(list(scores.items())[:5])}")
        
        if scores:
            print(f"Score range: {min(scores.values()):.4f} - {max(scores.values()):.4f}")
        else:
            print("⚠️ No scores computed - check PDB format")
    else:
        print(f"⚠️ Test file {test_pdb_path} not found")
        print("Available files:", [f for f in os.listdir('.') if f.endswith('.pdb')])

if __name__ == "__main__":
    # Run manual test before starting the API
    test_ewcl_function()
    
    # Start the API server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
