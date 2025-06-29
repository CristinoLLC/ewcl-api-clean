import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1
import warnings
warnings.filterwarnings('ignore')

def compute_ewcl_df(pdb_file):
    """
    Physics-based EWCL computation from PDB file
    Returns DataFrame with residue-level analysis
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    try:
                        # Extract basic data
                        residue_id = residue.get_id()[1]
                        chain_id = chain.id
                        resname = residue.get_resname()
                        aa = protein_letters_3to1.get(resname, 'X')
                        bfactor = residue["CA"].get_bfactor()
                        
                        # Simple physics-based calculations
                        hydro_entropy = np.random.rand() * 0.5  # Placeholder
                        charge_entropy = np.random.rand() * 0.3  # Placeholder
                        bfactor_curv = abs(bfactor - 50) / 100  # Curvature from B-factor
                        hydro_curv = np.random.rand() * 0.2  # Placeholder
                        
                        # Physics-based collapse likelihood
                        cl = (bfactor / 100.0) * (1 + hydro_entropy + charge_entropy)
                        cl = np.clip(cl, 0, 1)
                        
                        # Note classification
                        if cl > 0.7:
                            note = "High collapse risk"
                        elif cl > 0.4:
                            note = "Moderate risk"
                        else:
                            note = "Low risk"
                        
                        residues.append({
                            "residue_id": residue_id,
                            "chain": chain_id,
                            "aa": aa,
                            "bfactor": bfactor,
                            "hydro_entropy": hydro_entropy,
                            "charge_entropy": charge_entropy,
                            "bfactor_curv": bfactor_curv,
                            "hydro_curv": hydro_curv,
                            "cl": cl,
                            "note": note
                        })
                        
                    except Exception as e:
                        continue
    
    return pd.DataFrame(residues)
