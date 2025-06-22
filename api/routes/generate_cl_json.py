from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser
from datetime import datetime
import numpy as np
import io
from ewcl_metrics import compute_metrics
import logging

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

@router.post("/generate-cl-json")
async def generate_cl_json(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range"),
    use_raw_ewcl: bool = Query(default=False, description="Use raw EWCL scores for metrics computation"),
    mode: str = Query(default="collapse", description="Interpretation mode: 'collapse' or 'reverse'"),
    disorder_labels: str = Query(default=None, description="Optional comma-separated binary labels for disorder regions")
):
    """
    Generate CL JSON from PDB upload with optional normalization, raw EWCL scores, mode, and disorder labels
    """
    try:
        pdb_bytes = await file.read()
        pdb_str = pdb_bytes.decode()
        
        # Validate PDB string
        if len(pdb_str) < 100:
            raise HTTPException(status_code=400, detail="Invalid PDB: File too short")
        
        if not any(line.startswith(('ATOM', 'HETATM')) for line in pdb_str.split('\n')):
            raise HTTPException(status_code=400, detail="Invalid PDB: No ATOM or HETATM lines found")
        
        structure = parser.get_structure("u", io.StringIO(pdb_str))
        
        # Extract B-factors with graceful fallback
        plddt_scores = []
        has_valid_bfactors = False
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        try:
                            bfactor = residue["CA"].get_bfactor()
                            if bfactor > 0:  # Valid B-factor
                                has_valid_bfactors = True
                            plddt_scores.append(bfactor)
                        except:
                            # Fallback for missing B-factor
                            plddt_scores.append(50.0)  # Neutral score

        if not plddt_scores:
            raise HTTPException(status_code=400, detail="Invalid PDB: No CA atoms found")

        # === Predict collapse likelihood ===
        cl_scores_raw = cl_model.score(np.array(plddt_scores))
        
        # Apply mode interpretation
        interpret_as_disorder = mode.lower() == "reverse"
        if interpret_as_disorder:
            # Reverse interpretation: high scores = disorder
            cl_scores_raw = 1 - cl_scores_raw
        
        # === Normalize if requested ===
        if normalize:
            min_score = np.min(cl_scores_raw)
            max_score = np.max(cl_scores_raw)
            if max_score == min_score:
                # Avoid division by zero
                cl_scores_normalized = np.full_like(cl_scores_raw, 0.5)
            else:
                cl_scores_normalized = (cl_scores_raw - min_score) / (max_score - min_score + 1e-8)
        else:
            cl_scores_normalized = cl_scores_raw

        # === Build response with both raw and scaled scores ===
        scores = []
        for i, (raw_cl, norm_cl, plddt) in enumerate(zip(cl_scores_raw, cl_scores_normalized, plddt_scores)):
            scores.append({
                "residue_id": i + 1,
                "cl": round(float(norm_cl), 6),      # scaled 0-1 collapse likelihood  
                "raw_cl": round(float(raw_cl), 6),   # un-scaled EWCL
                "plddt": round(float(plddt), 6) if has_valid_bfactors else None,
                "b_factor": round(float(plddt), 6) if has_valid_bfactors else None
            })

        # === Compute metrics ===
        # Parse disorder labels if provided
        parsed_labels = None
        if disorder_labels:
            try:
                parsed_labels = [int(x.strip()) for x in disorder_labels.split(',')]
                if len(parsed_labels) != len(scores):
                    logging.warning(f"Label count mismatch: {len(parsed_labels)} labels vs {len(scores)} residues")
                    parsed_labels = None
            except ValueError:
                logging.warning("Invalid disorder labels format - expected comma-separated 0/1 values")
                parsed_labels = None
        
        try:
            metrics = compute_metrics(scores, disorder_labels=parsed_labels)
            if not has_valid_bfactors:
                metrics["data_warning"] = "B-factor data missing or invalid - correlation metrics may be unreliable"
        except Exception as e:
            logging.warning(f"Metrics computation failed: {e}")
            metrics = {"error": "Metrics computation failed"}

        response = {
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "normalized": normalize,
            "use_raw_ewcl": use_raw_ewcl,
            "mode": mode.lower(),
            "interpretation": "Reverse EWCL (Disorder)" if interpret_as_disorder else "Collapse Likelihood",
            "has_valid_bfactors": has_valid_bfactors,
            "generated": datetime.utcnow().isoformat() + "Z",
            "scores": scores,
            "metrics": metrics
        }

        return JSONResponse(content=response)

    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid PDB: File encoding error"})
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("❌ Error processing PDB file")
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
