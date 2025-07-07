import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, is_aa
from sklearn.cluster import DBSCAN
import io
from models.disprot_integration import extract_uniprot_from_header, get_disprot_labels, compute_disprot_metrics

def is_alphafold_pdb(pdb_content: str) -> bool:
    """
    Detect AlphaFold PDB using header patterns
    """
    lines = pdb_content.splitlines()[:10]
    for line in lines:
        if "AF-" in line and ("F1-model" in line or "ALPHAFOLD" in line.upper()):
            return True
    return False

def three_to_one(residue_name):
    """Convert three-letter amino acid code to one-letter"""
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    return aa_dict.get(residue_name, 'X')

def parse_pdb(pdb_content: str, model_type: str) -> list[dict]:
    """Parse PDB using the enhanced EWCL physics model"""
    import tempfile
    
    # Save content to temporary file for EWCL processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
        tmp.write(pdb_content)
        tmp_path = tmp.name
    
    try:
        # Use the enhanced physics-based EWCL model
        from models.ewcl_physics import compute_ewcl_from_pdb
        residues = compute_ewcl_from_pdb(tmp_path)
        return residues
    finally:
        import os
        os.remove(tmp_path)

def annotate_residues(residues: list[dict], metric: str, pdb_content: str = None) -> list[dict]:
    """
    Enrich residues with physics-based features only
    """
    # Use the proper EWCL physics computation from ewcl_physics.py
    # The CL scores are already computed there using pure physics
    
    # Add risk level buckets based on CL
    for r in residues:
        if r["cl"] > 0.7:
            r["risk_level"] = "Very Flexible"
        elif r["cl"] > 0.4:
            r["risk_level"] = "Medium-High" 
        elif r["cl"] > 0.2:
            r["risk_level"] = "Medium"
        else:
            r["risk_level"] = "Low"
        
        # Hallucination detection using EWCL vs confidence mismatch
        # This is now proper since CL is independent of pLDDT/B-factor
        conf = r["plddt"] if metric == "pLDDT" else (100 - r["b_factor"])
        r["hallucination"] = r["cl"] > 0.7 and conf < 70
    
    # Add DisProt annotations if PDB content is available
    if pdb_content:
        uniprot_id = extract_uniprot_from_header(pdb_content)
        disprot_labels = get_disprot_labels(uniprot_id, len(residues))
        
        for r, label in zip(residues, disprot_labels):
            r["disprot"] = label
            r["uniprot_id"] = uniprot_id
    else:
        for r in residues:
            r["disprot"] = None
            r["uniprot_id"] = None
    
    # Clustering based on CL and confidence (now properly independent)
    if len(residues) > 3:
        features = np.array([[r["cl"], r["plddt"] or r["b_factor"]] for r in residues])
        labels = DBSCAN(eps=0.15, min_samples=3).fit_predict(features)
        
        for r, label in zip(residues, labels):
            r["cluster_id"] = int(label)
    else:
        for r in residues:
            r["cluster_id"] = 0
    
    return residues
