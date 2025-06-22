from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
from ewcl_metrics import compute_metrics

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

class PDBJSONInput(BaseModel):
    pdb_data: str
    normalize: bool = True
    use_raw_ewcl: bool = False    use_raw_ewcl: bool = False

def run_ewcl_analysis(pdb_str: str, normalize: bool = True) -> dict:_ewcl_analysis(pdb_str: str, normalize: bool = True, use_raw_ewcl: bool = False) -> dict:
    """
    Core EWCL analysis function that processes PDB string and returns resultse EWCL analysis function that processes PDB string and returns results
    """
    # Validate PDB string
    if len(pdb_str) < 100:
        raise HTTPException(status_code=400, detail="Invalid PDB: File too short (less than 100 characters)")    raise HTTPException(status_code=400, detail="Invalid PDB: File too short (less than 100 characters)")
    
    if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
        raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")    raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
    
    try:
        structure = parser.get_structure("protein", io.StringIO(pdb_str))structure = parser.get_structure("protein", io.StringIO(pdb_str))
        
        plddt_scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        plddt_scores.append(residue["CA"].get_bfactor())                plddt_scores.append(residue["CA"].get_bfactor())
        
        if not plddt_scores:
            raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")    raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")
        
        # === Predict collapse likelihood ===
        cl_scores = cl_model.score(np.array(plddt_scores))cl_scores_raw = cl_model.score(np.array(plddt_scores))  # Raw CL scores
        
        # === Normalize if requested ===ze if requested ===
        if normalize:
            min_score = np.min(cl_scores)raw)
            max_score = np.max(cl_scores)
            cl_scores_norm = (cl_scores - min_score) / (max_score - min_score + 1e-8)  # avoid div by 0l_scores_normalized = (cl_scores_raw - min_score) / (max_score - min_score + 1e-8)
        else:
            cl_scores_norm = cl_scores    cl_scores_normalized = cl_scores_raw
        
        # === Build response ===response with both raw and scaled scores ===
        results = []
        for i, (cl, plddt) in enumerate(zip(cl_scores_norm, plddt_scores)):_cl, plddt) in enumerate(zip(cl_scores_raw, cl_scores_normalized, plddt_scores)):
            results.append({
                "residue_id": i + 1,
                "cl": round(float(cl), 6),     # scaled 0-1 collapse likelihood
                "plddt": round(float(plddt), 6),   # un-scaled EWCL
                "b_factor": round(float(plddt), 6)  "plddt": round(float(plddt), 6),
            })        "b_factor": round(float(plddt), 6)
        
        # === Compute metrics ===
        metrics = compute_metrics(results)# === Pick which score drives downstream metrics ===
        _scores = np.array([
        return {lse r["cl"])
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "normalized": normalize,
            "generated": datetime.utcnow().isoformat() + "Z",ected scores ===
            "n_residues": len(results),cs(results)
            "results": results,
            "metrics": metricseturn {
        }    "model": "CollapseLikelihood",
        model.lambda_,
    except HTTPException:normalized": normalize,
        raise: use_raw_ewcl,
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDB: Failed to parse - {str(e)}")            "n_residues": len(results),
ts,
@router.post("/analyze-pdb") metrics
async def analyze_pdb(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range")  except HTTPException:
): raise
    """
    Upload PDB file for EWCL analysis with optional normalizatione - {str(e)}")
    Pass ?normalize=true for [0,1] normalization, ?normalize=false for raw scores
    """post("/analyze-pdb")
    try:
        pdb_bytes = await file.read()
        pdb_str = pdb_bytes.decode()"Normalize CL scores to [0, 1] range"),
        result = run_ewcl_analysis(pdb_str, normalize)e, description="Use raw EWCL scores for metrics computation")
        return JSONResponse(content=result)
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid PDB: File encoding error"})nalysis with optional normalization and raw score mode
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})        pdb_str = pdb_bytes.decode()
s(pdb_str, normalize, use_raw_ewcl)
@router.post("/analyze-pdb-json")
async def analyze_pdb_json(input_data: PDBJSONInput):ept UnicodeDecodeError:
    """"Invalid PDB: File encoding error"})
    Analyze PDB from JSON body (alternate route for JSON input)ept HTTPException as e:
    """: e.detail})
    return run_ewcl_analysis(input_data.pdb_data, input_data.normalize)    except Exception as e:
tus_code=500, content={"error": str(e)})
@router.post("/get-raw-scores")
async def get_raw_scores(pdb_file: UploadFile = File(...)):.post("/analyze-pdb-json")
    """
    Get raw entropy scores and normalized CL scores for research/diagnostic purposes
    Independent of the main /analyze-pdb endpointlyze PDB from JSON body (alternate route for JSON input)
    """
    try:pdb_data, input_data.normalize, getattr(input_data, 'use_raw_ewcl', False))
        contents = await pdb_file.read()
        pdb_str = contents.decode()post("/get-raw-scores")
        file: UploadFile = File(...)):
        # Validate PDB string
        if len(pdb_str) < 100:
            raise HTTPException(status_code=400, detail="Invalid PDB: File too short")pendent of the main /analyze-pdb endpoint
        
        if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
            raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")contents = await pdb_file.read()
        
        structure = parser.get_structure("protein", io.StringIO(pdb_str))
        
        # Extract pLDDT scores (B-factors) 100:
        plddt_scores = [](status_code=400, detail="Invalid PDB: File too short")
        for model in structure:
            for chain in model:ATOM', 'HETATM')) for line in pdb_str.split('\n')):
                for residue in chain:code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
                    if "CA" in residue:
                        plddt_scores.append(residue["CA"].get_bfactor())structure = parser.get_structure("protein", io.StringIO(pdb_str))
        
        if not plddt_scores:
            raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")plddt_scores = []
        
        # === Step 1: Compute raw entropy scores using CollapseLikelihood model ===
        plddt_array = np.array(plddt_scores)
        raw_entropy_scores = cl_model.score(plddt_array)  # Raw CL scores before normalization            if "CA" in residue:
        r())
        # === Step 2: Normalize using same logic as /analyze-pdb ===
        min_val = np.min(raw_entropy_scores)
        max_val = np.max(raw_entropy_scores)ound")
        cl_normalized = (raw_entropy_scores - min_val) / (max_val - min_val + 1e-8)
         CollapseLikelihood model ===
        # === Step 3: Return both raw + normalized === = np.array(plddt_scores)
        result = []
        for i, (raw, norm, plddt) in enumerate(zip(raw_entropy_scores, cl_normalized, plddt_scores)):
            result.append({g same logic as /analyze-pdb ===
                "residue_id": i + 1,
                "raw_entropy": round(float(raw), 6),
                "cl_normalized": round(float(norm), 6),n_val) / (max_val - min_val + 1e-8)
                "plddt": round(float(plddt), 6),
                "b_factor": round(float(plddt), 6)Step 3: Return both raw + normalized ===
            })        result = []
enumerate(zip(raw_entropy_scores, cl_normalized, plddt_scores)):
        return JSONResponse(content={
            "model": "EWCL-Raw",
            "lambda": cl_model.lambda_,loat(raw), 6),
            "n_residues": len(result),
            "raw_range": [round(float(min_val), 6), round(float(max_val), 6)],, 6),
            "normalized_range": [0.0, 1.0],
            "generated": datetime.utcnow().isoformat() + "Z",
            "results": result
        })return JSONResponse(content={
        ",
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid PDB: File encoding error"}) len(result),
    except HTTPException:raw_range": [round(float(min_val), 6), round(float(max_val), 6)],
        raisenge": [0.0, 1.0],
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Raw scores computation failed: {str(e)}"})            "results": result

        })
        
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid PDB: File encoding error"})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Raw scores computation failed: {str(e)}"})
