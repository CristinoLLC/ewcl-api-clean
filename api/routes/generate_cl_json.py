from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from models.collapse_likelihood import CollapseLikelihood
from Bio.PDB import PDBParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from datetime import datetime
import numpy as np
import io
from ewcl_metrics import compute_metrics
import logging
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
from core.utils import classify_risk_and_color

router = APIRouter()
cl_model = CollapseLikelihood(lambda_=3.0)
parser = PDBParser(QUIET=True)

THR_DISORDER = 0.609  # DisProt threshold

def calc_mismatch(cl, ref):
    """Calculate mismatch score between CL and reference, handling None/NaN values"""
    if cl is None or ref is None or np.isnan(cl) or np.isnan(ref):
        return None
    return round(abs(cl - (ref / 100.0)), 4)  # Normalize ref to 0-1 scale

def build_summary(results):
    """
    results : list[dict] – each residue dict has keys
        cl, plddt, b_factor, disorder (optional), ...
    """
    # Pull the numeric columns once
    cl = np.asarray([r["cl"] for r in results], dtype=float)
    plddt = np.asarray([r.get("plddt") for r in results if r.get("plddt") is not None], dtype=float)
    bfactor = np.asarray([r.get("b_factor") for r in results if r.get("b_factor") is not None], dtype=float)
    disorder = np.asarray([r.get("disorder") for r in results if r.get("disorder") is not None], dtype=float)

    summary = {}

    # ---------- pLDDT correlations ----------
    if len(plddt) == len(cl) and np.isfinite(plddt).all() and len(set(cl)) > 1:
        summary["plddt_spearman"] = round(spearmanr(cl, plddt)[0], 3)
        summary["plddt_kendall"] = round(kendalltau(cl, plddt)[0], 3)

    # ---------- B-factor correlations ----------
    if len(bfactor) == len(cl) and np.isfinite(bfactor).all() and len(set(cl)) > 1:
        summary["bfactor_spearman"] = round(spearmanr(cl, bfactor)[0], 3)
        summary["bfactor_kendall"] = round(kendalltau(cl, bfactor)[0], 3)

    # ---------- DisProt AUC ----------
    if len(disorder) == len(cl) and np.isfinite(disorder).all():
        labels = (disorder > THR_DISORDER).astype(int)
        if len(set(labels)) > 1:  # Need both classes for AUC
            summary["disprot_auc"] = round(roc_auc_score(labels, cl), 3)

    return summary

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
        
        # Extract protein name from header
        protein_name = "Unknown Protein"
        if structure.header and "name" in structure.header:
            protein_name = structure.header.get("name", "Unknown Protein")
        elif structure.header and "compound" in structure.header:
            try:
                protein_name = structure.header["compound"]["1"]["mol_id"]
            except (KeyError, IndexError):
                pass

        # Build source block
        source_info = {
            "pdb_id": file.filename,
            "uniprot_id": None,  # Cannot determine from PDB file alone
            "name": protein_name,
            "uploaded_by": "user"
        }
        
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
        missing_plddt_count = 0
        missing_bfactor_count = 0
        mismatch_none_count = 0
        
        for i, (raw_cl, norm_cl, plddt, aa, chain_id) in enumerate(zip(cl_scores_raw, cl_scores_normalized, plddt_scores, amino_acids, chain_ids)):
            # Track missing data
            if plddt is None or np.isnan(plddt):
                missing_plddt_count += 1
            if not has_valid_bfactors:
                missing_bfactor_count += 1
                
            # Calculate mismatch score properly using helper function
            mismatch_score = calc_mismatch(norm_cl, plddt)
            if mismatch_score is None:
                mismatch_none_count += 1
            
            risk_info = classify_risk_and_color(norm_cl)
            
            # Ensure all mandatory fields are present
            scores.append({
                "residue_id": i + 1,
                "chain": chain_id if chain_id else "A",                    # default to chain A
                "aa": aa if aa else "X",                                   # default to X if unknown
                "cl": round(float(norm_cl), 6),                           # scaled 0-1 collapse likelihood  
                "raw_cl": round(float(raw_cl), 6),                        # un-scaled EWCL
                "plddt": round(float(plddt), 6) if plddt is not None and not np.isnan(plddt) else None,
                "b_factor": round(float(plddt), 6) if plddt is not None and not np.isnan(plddt) else None,
                "mismatch_score": mismatch_score,                         # proper mismatch calculation
                "risk_class": risk_info["risk_class"],
                "color_hex": risk_info["color_hex"]
            })

        # Log data quality metrics
        logging.info(f"📊 Data Quality - Total residues: {len(scores)}, Missing pLDDT: {missing_plddt_count}, Missing B-factor: {missing_bfactor_count}, Mismatch None: {mismatch_none_count}")

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
            
            # Enhanced correlation metrics with proper data cleaning
            correlation_results = compute_enhanced_correlations(scores)
            
            # Add robust summary metrics
            summary_metrics = build_summary(scores)
            
            # Combine all metrics
            metrics["correlation_results"] = correlation_results
            metrics["summary_metrics"] = summary_metrics
            
            if not has_valid_bfactors:
                metrics["data_warning"] = "B-factor data missing or invalid - correlation metrics may be unreliable"
        except Exception as e:
            logging.warning(f"Metrics computation failed: {e}")
            metrics = {"error": "Metrics computation failed"}

        # Enhanced interpretation tag
        interpretation_text = {
            "reverse": "High score = high disorder likelihood",
            "collapse": "High score = high collapse likelihood"
        }

        response = {
            "model": "CollapseLikelihood",
            "mode": mode.lower(),
            "interpretation": interpretation_text.get(mode.lower(), "Score interpretation"),
            "generated": datetime.utcnow().isoformat() + "Z",
            "source": source_info,
            "n_residues": len(scores),
            "data_quality": {
                "missing_plddt": missing_plddt_count,
                "missing_bfactor": missing_bfactor_count,
                "mismatch_none": mismatch_none_count
            },
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

def compute_enhanced_correlations(scores):
    """Compute robust correlations with proper data cleaning"""
    try:
        # Extract data for correlation analysis
        data_rows = []
        for score in scores:
            cl_val = score.get('cl')
            plddt_val = score.get('plddt') 
            bfactor_val = score.get('b_factor')
            
            if cl_val is not None and not np.isnan(cl_val):
                row = {'cl': cl_val}
                if plddt_val is not None and not np.isnan(plddt_val):
                    row['plddt'] = plddt_val
                if bfactor_val is not None and not np.isnan(bfactor_val):
                    row['b_factor'] = bfactor_val
                data_rows.append(row)
        
        if not data_rows:
            return {"error": "No valid data for correlation analysis"}
        
        # Create DataFrame and compute correlations with dropna
        df = pd.DataFrame(data_rows)
        
        correlation_results = {
            "vs_pLDDT": {"pearson": None, "spearman": None, "kendall": None},
            "vs_b_factor": {"pearson": None, "spearman": None, "kendall": None},
            "n_valid_samples": len(df)
        }
        
        # pLDDT correlations
        if 'plddt' in df.columns:
            plddt_df = df[['cl', 'plddt']].dropna()
            if len(plddt_df) >= 3 and len(plddt_df['cl'].unique()) > 1:
                correlation_results["vs_pLDDT"]["pearson"] = round(float(plddt_df["cl"].corr(plddt_df["plddt"], method="pearson")), 4)
                correlation_results["vs_pLDDT"]["spearman"] = round(float(plddt_df["cl"].corr(plddt_df["plddt"], method="spearman")), 4)
                correlation_results["vs_pLDDT"]["kendall"] = round(float(plddt_df["cl"].corr(plddt_df["plddt"], method="kendall")), 4)
                correlation_results["vs_pLDDT"]["n_samples"] = len(plddt_df)
        
        # B-factor correlations
        if 'b_factor' in df.columns:
            bfactor_df = df[['cl', 'b_factor']].dropna()
            if len(bfactor_df) >= 3 and len(bfactor_df['cl'].unique()) > 1:
                correlation_results["vs_b_factor"]["pearson"] = round(float(bfactor_df["cl"].corr(bfactor_df["b_factor"], method="pearson")), 4)
                correlation_results["vs_b_factor"]["spearman"] = round(float(bfactor_df["cl"].corr(bfactor_df["b_factor"], method="spearman")), 4)
                correlation_results["vs_b_factor"]["kendall"] = round(float(bfactor_df["cl"].corr(bfactor_df["b_factor"], method="kendall")), 4)
                correlation_results["vs_b_factor"]["n_samples"] = len(bfactor_df)
        
        return correlation_results
        
    except Exception as e:
        logging.error(f"Enhanced correlation computation failed: {e}")
        return {"error": f"Correlation analysis failed: {str(e)}"}
