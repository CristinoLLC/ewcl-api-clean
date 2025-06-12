"""
Entropy Collapse Model for Protein Analysis
Real implementation with sliding window entropy calculation
"""

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1 as three_to_one
from typing import List, Dict, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_ewcl(residues, window=5):
    """
    Compute entropy-weighted collapse likelihood scores using sliding window
    
    Args:
        residues: List of Bio.PDB residue objects
        window: Window size for local entropy calculation
        
    Returns:
        List of entropy scores for each residue
    """
    entropy_scores = []
    for i in range(len(residues)):
        # Extract window of residues around current position
        window_residues = residues[max(0, i - window):min(len(residues), i + window + 1)]
        
        # Count amino acid occurrences in window
        counts = {}
        for res in window_residues:
            try:
                aa = three_to_one(res.get_resname())
                counts[aa] = counts.get(aa, 0) + 1
            except:
                continue
        
        # Calculate Shannon entropy
        total = sum(counts.values())
        if total > 0:
            probs = [v / total for v in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            entropy = 0
            
        entropy_scores.append(entropy)
    
    return entropy_scores

def infer_entropy_from_pdb(pdb_path: str, reverse: bool = False) -> List[Dict[str, Union[int, str, float]]]:
    """
    Analyze a PDB file and return entropy collapse predictions for each residue
    
    Args:
        pdb_path: Path to the PDB file
        reverse: If True, applies reverse mode logic for disordered proteins
        
    Returns:
        List of dictionaries: [{"residue_id": int, "aa": str, "ewcl_score": float}, ...]
    """
    logger.info("🧠 Real entropy model loaded successfully.")
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        
        results = []
        residues = []
        
        # Collect all residues first
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == " ":  # Standard amino acid
                        residues.append(residue)
            break  # Only process first model
        
        if not residues:
            logger.warning("No valid residues found in PDB file")
            return []
        
        # Compute real entropy scores using sliding window
        scores = compute_ewcl(residues)
        
        # Build results with real entropy calculations
        for i, residue in enumerate(residues):
            try:
                residue_id = residue.get_id()[1]
                aa = three_to_one(residue.get_resname())
                base_score = scores[i] if i < len(scores) else 0.0
                
                # Apply reverse mode if requested
                if reverse:
                    ewcl_score = 1.0 - base_score
                else:
                    ewcl_score = base_score
                
                results.append({
                    "residue_id": residue_id,
                    "aa": aa,
                    "ewcl_score": round(ewcl_score, 6)
                })
            except Exception as e:
                logger.warning(f"Error processing residue {residue.get_id()}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(results)} residues with real entropy model from {pdb_path}")
        return results
        
    except Exception as e:
        logger.error(f"Error processing PDB file {pdb_path}: {e}")
        raise