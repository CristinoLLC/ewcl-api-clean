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
    """Parse PDB and extract residue data with B-factor/pLDDT info"""
    alphafold = model_type.lower() == "alphafold"
    residues = []
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", io.StringIO(pdb_content))
    
    for res in structure[0].get_residues():
        if not is_aa(res, standard=True) or "CA" not in res:
            continue
            
        # Get average B-factor for the residue
        b_factor = np.mean([atom.get_bfactor() for atom in res])
        
        residues.append({
            "chain": res.get_parent().id,
            "residue_id": int(res.id[1]),
            "aa": three_to_one(res.get_resname()),
            "b_factor": round(float(b_factor), 2),
            "plddt": round(float(b_factor), 2) if alphafold else None,
        })
    
    return residues

def annotate_residues(residues: list[dict], metric: str, pdb_content: str = None) -> list[dict]:
    """
    Enrich residues with:
    - cl (collapse likelihood) 
    - risk_level
    - hallucination flag
    - cluster_id
    - disprot labels
    """
    from models.ewcl_physics import compute_ewcl_from_pdb
    import tempfile
    
    # Compute CL scores (using simplified approach for now)
    cl_scores = []
    for r in residues:
        # Simplified CL calculation - you can replace with full EWCL logic
        conf = r["plddt"] if metric == "pLDDT" else r["b_factor"]
        # Inverse relationship: high confidence = low collapse likelihood
        cl = max(0, min(1, (100 - conf) / 100)) if conf else 0.5
        cl_scores.append(cl)
    
    # Add CL scores to residues
    for r, cl in zip(residues, cl_scores):
        r["cl"] = round(float(cl), 3)
        
        # Risk level buckets
        if r["cl"] > 0.7:
            r["risk_level"] = "Very Flexible"
        elif r["cl"] > 0.4:
            r["risk_level"] = "Medium-High" 
        elif r["cl"] > 0.2:
            r["risk_level"] = "Medium"
        else:
            r["risk_level"] = "Low"
        
        # Hallucination detection
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
    
    # Clustering based on CL and confidence
    if len(residues) > 3:
        features = np.array([[r["cl"], r["plddt"] or r["b_factor"]] for r in residues])
        labels = DBSCAN(eps=0.15, min_samples=3).fit_predict(features)
        
        for r, label in zip(residues, labels):
            r["cluster_id"] = int(label)  # -1 for noise
    else:
        for r in residues:
            r["cluster_id"] = 0
    
    return residues
