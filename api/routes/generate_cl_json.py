from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from datetime import datetime
import numpy as np
import io
from ewcl_metrics import compute_metrics
from correlation_metrics import compute_correlation_metrics_cleaned
import logging
import pandas as pd

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

@router.post("/generate-cl-json")
async def generate_cl_json(
    file: UploadFile = File(...),
    normalize: bool = Query(default=True, description="Normalize CL scores to [0, 1] range"),
    use_raw_ewcl: bool = Query(default=False, description="Use raw EWCL scores for metrics computation"),
    mode: str = Query(default="collapse", description="Interpretation mode: 'collapse' or 'reverse'"),
    threshold: float = Query(default=None, description="Custom CL threshold (default: 0.500 for collapse, 0.609 for reverse)"),
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
        
        # Extract B-factors, amino acids, and chain IDs with graceful fallback
        plddt_scores = []
        amino_acids = []
        chain_ids = []
        has_valid_bfactors = False
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue and is_aa(residue):
                        try:
                            bfactor = residue["CA"].get_bfactor()
                            if bfactor > 0:  # Valid B-factor
                                has_valid_bfactors = True
                            plddt_scores.append(bfactor)
                        except:
                            # Fallback for missing B-factor
                            plddt_scores.append(50.0)  # Neutral score
                        
                        # Extract amino acid
                        resname = residue.get_resname().title()  # e.g., 'Ala'
                        aa = protein_letters_3to1.get(resname, 'X')  # fallback to 'X' if unknown
                        amino_acids.append(aa)
                        
                        # Extract chain ID
                        chain_ids.append(chain.id)

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
        for i, (raw_cl, norm_cl, plddt, aa, chain_id) in enumerate(zip(cl_scores_raw, cl_scores_normalized, plddt_scores, amino_acids, chain_ids)):
            # Calculate mismatch score properly
            mismatch_score = None
            if norm_cl is not None and plddt is not None and has_valid_bfactors:
                if not (np.isnan(norm_cl) or np.isnan(plddt)):
                    mismatch_score = round(abs(norm_cl - (plddt / 100.0)), 4)
            
            scores.append({
                "residue_id": i + 1,
                "chain": chain_id if chain_id else "A",     # default to chain A if missing
                "aa": aa if aa else "X",                    # default to X if unknown
                "cl": round(float(norm_cl), 6),             # scaled 0-1 collapse likelihood  
                "raw_cl": round(float(raw_cl), 6),          # un-scaled EWCL
                "plddt": round(float(plddt), 6) if has_valid_bfactors and not np.isnan(plddt) else None,
                "b_factor": round(float(plddt), 6) if has_valid_bfactors and not np.isnan(plddt) else None,
                "mismatch_score": mismatch_score            # difference between CL and normalized pLDDT
            })

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
            # Comprehensive metrics from ewcl_metrics
            metrics = compute_metrics(
                scores, 
                cl_thresh=threshold,
                disorder_labels=parsed_labels, 
                mode=mode.lower()
            )
            
            # Additional focused correlation metrics with proper data cleaning
            correlation_metrics = compute_correlation_metrics_cleaned(scores, mode.lower())
            
            # Combine metrics
            metrics["correlation_analysis"] = correlation_metrics
            
            if not has_valid_bfactors:
                metrics["data_warning"] = "B-factor data missing or invalid - correlation metrics may be unreliable"
        except Exception as e:
            logging.warning(f"Metrics computation failed: {e}")
            metrics = {"error": "Metrics computation failed"}

        response = {
            "model": "CollapseLikelihood",
            "lambda": cl_model.lambda_,
            "mode": mode.lower(),
            "normalized": normalize,
            "use_raw_ewcl": use_raw_ewcl,
            "interpretation": "Reverse EWCL (Disorder)" if interpret_as_disorder else "Collapse Likelihood",
            "has_valid_bfactors": has_valid_bfactors,
            "generated": datetime.utcnow().isoformat() + "Z",
            "n_residues": len(scores),
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
