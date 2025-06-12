"""
Entropy Collapse Model for Protein Analysis
Real implementation with sliding window entropy calculation
"""

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1 as three_to_one
from typing import List, Dict
import os
import logging

logger = logging.getLogger(__name__)

def extract_residues(file_path: str):
    """Extract residues from PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    model = structure[0]
    residues = [res for chain in model for res in chain if res.id[0] == " "]
    return residues

def compute_ewcl_entropy(residues, window=5):
    """Compute entropy scores using sliding window approach"""
    scores = []
    for i in range(len(residues)):
        window_res = residues[max(0, i - window):min(len(residues), i + window + 1)]
        counts = {}
        for res in window_res:
            try:
                aa = three_to_one(res.resname)
                counts[aa] = counts.get(aa, 0) + 1
            except Exception:
                continue
        total = sum(counts.values())
        probs = [v / total for v in counts.values()] if total > 0 else []
        entropy = -sum(p * np.log2(p) for p in probs if p > 0) if probs else 0
        scores.append(entropy)
    return scores

def infer_entropy_from_pdb(path: str, reverse: bool = False) -> Dict:
    """
    Main function to analyze PDB file and return entropy scores
    
    Args:
        path: Path to PDB file
        reverse: If True, applies reverse mode for disordered proteins
        
    Returns:
        Complete JSON response with status, mode, and results
    """
    logger.info("🧠 Real entropy model processing PDB file")
    
    try:
        residues = extract_residues(path)
        logger.info(f"📊 Extracted {len(residues)} residues from PDB")
        
        if not residues:
            logger.warning("❌ No residues found in PDB file")
            return {
                "status": "error",
                "mode": "reverse" if reverse else "normal",
                "reverse": reverse,
                "results": [],
                "message": "No valid residues found in PDB file"
            }
        
        entropy_scores = compute_ewcl_entropy(residues)
        logger.info(f"📊 Computed {len(entropy_scores)} entropy scores")

        results = []
        for i, (res, score) in enumerate(zip(residues, entropy_scores)):
            try:
                aa = three_to_one(res.resname)
            except Exception:
                aa = "X"
            
            # Apply reverse mode if requested
            final_score = (1.0 - score) if reverse else score
            
            results.append({
                "residue_id": int(res.id[1]),
                "aa": aa,
                "ewcl_score": round(final_score, 6)
            })

        logger.info(f"✅ Successfully processed {len(results)} residues with real entropy model")
        
        # Ensure we have results before returning
        if not results:
            logger.warning("❌ No valid results generated")
            return {
                "status": "error", 
                "mode": "reverse" if reverse else "normal",
                "reverse": reverse,
                "results": [],
                "message": "Failed to generate valid entropy scores"
            }
        
        # Return complete JSON response for frontend compatibility
        return {
            "status": "success",
            "mode": "reverse" if reverse else "normal",
            "reverse": reverse,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error in entropy analysis: {e}")
        return {
            "status": "error",
            "mode": "reverse" if reverse else "normal", 
            "reverse": reverse,
            "results": [],
            "message": str(e)
        }