import re
import json
import os
from typing import Optional, List

# Mock DisProt database - in production, load from file or API
DISPROT_DB = {
    "P37840": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0] * 10,  # Alpha-synuclein example
    "P04637": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0] * 20,  # p53 example
    # Add more entries as needed
}

def extract_uniprot_from_header(pdb_content: str) -> Optional[str]:
    """Extract UniProt ID from AlphaFold PDB header"""
    # Look for AF-P37840-F1-model_v4 pattern
    match = re.search(r'AF-([A-Z0-9]+)-', pdb_content)
    if match:
        return match.group(1)
    
    # Also check DBREF lines for UniProt references
    lines = pdb_content.splitlines()
    for line in lines:
        if line.startswith("DBREF") and "UNP" in line:
            parts = line.split()
            for part in parts:
                if part.startswith(("P", "Q")) and len(part) >= 6:
                    return part
    
    return None

def get_disprot_labels(uniprot_id: str, residue_count: int) -> List[Optional[int]]:
    """Get DisProt disorder labels for a UniProt ID"""
    if not uniprot_id or uniprot_id not in DISPROT_DB:
        return [None] * residue_count
    
    labels = DISPROT_DB[uniprot_id]
    
    # Adjust length to match residue count
    if len(labels) >= residue_count:
        return labels[:residue_count]
    else:
        return labels + [None] * (residue_count - len(labels))

def load_disprot_from_file(filepath: str) -> dict:
    """Load DisProt database from JSON file if available"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def compute_disprot_metrics(residues: List[dict]) -> dict:
    """Compute metrics comparing EWCL predictions with DisProt labels"""
    valid_pairs = [(r["disprot"], r["cl"]) for r in residues 
                   if r.get("disprot") is not None]
    
    if not valid_pairs:
        return {
            "disprot_coverage": 0,
            "auc_vs_disprot": None,
            "correlation": None
        }
    
    y_true, y_score = zip(*valid_pairs)
    
    try:
        from sklearn.metrics import roc_auc_score
        from scipy.stats import pearsonr
        
        auc = roc_auc_score(y_true, y_score)
        correlation, _ = pearsonr(y_true, y_score)
        
        return {
            "disprot_coverage": len(valid_pairs) / len(residues),
            "auc_vs_disprot": round(float(auc), 3),
            "correlation": round(float(correlation), 3)
        }
    except ImportError:
        return {
            "disprot_coverage": len(valid_pairs) / len(residues),
            "auc_vs_disprot": None,
            "correlation": None
        }
