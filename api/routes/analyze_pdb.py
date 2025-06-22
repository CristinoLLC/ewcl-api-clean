from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import logging
from ewcl_metrics import compute_metrics
from pdf_generator import create_pdf_report

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

class PDBJSONInput(BaseModel):
    pdb_data: str
    normalize: bool = True
    use_raw_ewcl: bool = False
    mode: str = "collapse"

def run_ewcl_analysis(pdb_str: str, normalize: bool = True, use_raw_ewcl: bool = False, mode: str = "collapse") -> dict:
    """
    Core EWCL analysis function that processes PDB string and returns results
    """
    # Validate PDB string
    if len(pdb_str) < 100:
        raise HTTPException(status_code=400, detail="Invalid PDB: File too short (less than 100 characters)")
    
    if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
        raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
    
    try:
        structure = parser.get_structure("protein", io.StringIO(pdb_str))
        
        # Extract pLDDT scores with graceful fallback
        plddt_scores = []
        has_bfactor_data = False
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        try:
                            bfactor = residue["CA"].get_bfactor()
                            if bfactor > 0:  # Valid B-factor/pLDDT score
                                has_bfactor_data = True
                            plddt_scores.append(bfactor)
                        except:
                            # Fallback for missing B-factor data
                            plddt_scores.append(50.0)  # Default neutral score
        
        if not plddt_scores:
            raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")
        
        logging.info(f"✅ Extracted {len(plddt_scores)} residues, B-factor data available: {has_bfactor_data}")
        
        # === Predict collapse likelihood ===
        cl_scores_raw = cl_model.score(np.array(plddt_scores))  # Raw CL scores
        
        # Apply mode interpretation
        interpret_as_disorder = mode.lower() == "reverse"
        if interpret_as_disorder:
            # Reverse interpretation: high scores = disorder
            cl_scores_raw = 1 - cl_scores_raw
        
        # === Normalize if requested ===
        if normalize:
            min_score = np.min(cl_scores_raw)
            max_score = np.max(cl_scores_raw)
            cl_scores_normalized = (cl_scores_raw - min_score) / (max_score - min_score + 1e-8)
        else:
            cl_scores_normalized = cl_scores_raw
        
        # === Build response with both raw and scaled scores ===
        results = []
        for i, (raw_cl, norm_cl, plddt) in enumerate(zip(cl_scores_raw, cl_scores_normalized, plddt_scores)):
            results.append({
                "residue_id": i + 1,
                "cl": round(float(norm_cl), 6),      # scaled 0-1 collapse likelihood
                "raw_cl": round(float(raw_cl), 6),   # un-scaled EWCL
                "plddt": round(float(plddt), 6) if has_bfactor_data else None,
                "b_factor": round(float(plddt), 6) if has_bfactor_data else None
            })
        
        # === Compute metrics with graceful handling ===
        try:
            metrics = compute_metrics(results, disorder_labels=None)
            # Tag metrics that may be invalid due to missing data
            if not has_bfactor_data:
                metrics.update({
                    "pearson": None,
                    "spearman": None,
                    "data_warning": "B-factor/pLDDT data missing - correlation metrics unavailable"
                })
        except Exception as e:
            logging.warning(f"Metrics computation failed: {e}")
            metrics = {"error": "Metrics computation failed", "data_warning": str(e)}
        
        return {
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "normalized": normalize,
            "use_raw_ewcl": use_raw_ewcl,
            "mode": mode.lower(),
            "interpretation": "Reverse EWCL (Disorder)" if interpret_as_disorder else "Collapse Likelihood",
            "has_bfactor_data": has_bfactor_data,
            "generated": datetime.utcnow().isoformat() + "Z",
            "n_residues": len(results),
            "results": results,
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("❌ PDB processing failed")
        raise HTTPException(status_code=400, detail=f"Invalid PDB: Failed to parse - {str(e)}")

@router.post("/analyze-pdb")
async def analyze_pdb(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range"),
    use_raw_ewcl: bool = Query(default=False, description="Use raw EWCL scores for metrics computation"),
    mode: str = Query(default="collapse", description="Interpretation mode: 'collapse' or 'reverse'")
):
    """
    Upload PDB file for EWCL analysis with optional normalization, raw score mode, and interpretation mode
    """
    try:
        # Add logging for debugging
        logging.info(f"📂 Processing file: {file.filename}")
        
        pdb_bytes = await file.read()
        pdb_str = pdb_bytes.decode()
        
        logging.info("✅ File received and decoded successfully")
        
        result = run_ewcl_analysis(pdb_str, normalize, use_raw_ewcl, mode)
        return JSONResponse(content=result)
        
    except UnicodeDecodeError:
        logging.error("❌ File encoding error")
        return JSONResponse(status_code=400, content={"error": "Invalid PDB: File encoding error"})
    except HTTPException as e:
        logging.error(f"❌ HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        logging.exception("❌ Unexpected error during PDB analysis")
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)}"})

@router.post("/analyze-pdb-json")
async def analyze_pdb_json(input_data: PDBJSONInput):
    """
    Analyze PDB from JSON body (alternate route for JSON input)
    """
    try:
        return run_ewcl_analysis(input_data.pdb_data, input_data.normalize, input_data.use_raw_ewcl, input_data.mode)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/get-raw-scores")
async def get_raw_scores(pdb_file: UploadFile = File(...)):
    """
    Get raw entropy scores and normalized CL scores for research/diagnostic purposes
    Independent of the main /analyze-pdb endpoint
    """
    try:
        contents = await pdb_file.read()
        pdb_str = contents.decode()
        
        # Validate PDB string
        if len(pdb_str) < 100:
            raise HTTPException(status_code=400, detail="Invalid PDB: File too short")
        
        if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
            raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
        
        structure = parser.get_structure("protein", io.StringIO(pdb_str))
        
        # Extract pLDDT scores (B-factors)
        plddt_scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        plddt_scores.append(residue["CA"].get_bfactor())
        
        if not plddt_scores:
            raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")
        
        # === Step 1: Compute raw entropy scores using CollapseLikelihood model ===
        plddt_array = np.array(plddt_scores)
        raw_entropy_scores = cl_model.score(plddt_array)  # Raw CL scores before normalization
        
        # === Step 2: Normalize using same logic as /analyze-pdb ===
        min_val = np.min(raw_entropy_scores)
        max_val = np.max(raw_entropy_scores)
        cl_normalized = (raw_entropy_scores - min_val) / (max_val - min_val + 1e-8)
        
        # === Step 3: Return both raw + normalized ===
        result = []
        for i, (raw, norm, plddt) in enumerate(zip(raw_entropy_scores, cl_normalized, plddt_scores)):
            result.append({
                "residue_id": i + 1,
                "raw_entropy": round(float(raw), 6),
                "cl_normalized": round(float(norm), 6),
                "plddt": round(float(plddt), 6),
                "b_factor": round(float(plddt), 6)
            })
        
        return JSONResponse(content={
            "model": "EWCL-Raw",
            "lambda": cl_model.lambda_,
            "n_residues": len(result),
            "raw_range": [round(float(min_val), 6), round(float(max_val), 6)],
            "normalized_range": [0.0, 1.0],
            "generated": datetime.utcnow().isoformat() + "Z",
            "results": result
        })
        
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid PDB: File encoding error"})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Raw scores computation failed: {str(e)}"})

@router.post("/analyze-pdb-pdf")
async def analyze_pdb_pdf(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range"),
    use_raw_ewcl: bool = Query(default=False, description="Use raw EWCL scores for metrics computation")
):
    """
    Generate PDF report from PDB analysis
    """
    try:
        pdb_bytes = await file.read()
        pdb_str = pdb_bytes.decode()
        
        # Get analysis results
        analysis_data = run_ewcl_analysis(pdb_str, normalize, use_raw_ewcl)
        
        # Generate PDF
        pdf_bytes = create_pdf_report(analysis_data)
        
        # Return PDF as streaming response
        pdf_buffer = io.BytesIO(pdf_bytes)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=ewcl_analysis_report.pdf"}
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"PDF generation failed: {str(e)}"})
