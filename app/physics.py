"""
Physics wrapper for enhanced_ewcl_af
"""

import io
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser

# Import the full extractor
from models.enhanced_ewcl_af import compute_curvature_features

def run_physics(pdb_bytes: bytes) -> pd.DataFrame:
    """
    Call your physics extractor on a raw PDB byte string.
    Returns a DataFrame with columns: residue_id, bfactor, cl, etc.
    """
    tmp_file = Path("/tmp/_p.pdb")
    tmp_file.write_bytes(pdb_bytes)
    
    rows = compute_curvature_features(str(tmp_file))   # returns list[dict]
    df = pd.DataFrame(rows)
    
    # Clean up temp file
    tmp_file.unlink(missing_ok=True)
    
    return df
