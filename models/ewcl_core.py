"""
Fresh EWCL Physics Model - Clean Implementation
"""

import os, re
import numpy as np
from scipy.stats import entropy
from Bio.PDB import PDBParser
from typing import List, Dict

# Amino acid properties
HYDROPATHY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}
CHARGE = {aa: 0 for aa in HYDROPATHY}
CHARGE.update({'ASP': -1, 'GLU': -1, 'ARG': 1, 'LYS': 1, 'HIS': 0.5})

def normalize(x):
    """Min-max normalization to [0,1]"""
    x = np.array(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def sliding_entropy(arr, window=5):
    """Shannon entropy in sliding window"""
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window//2)
        end = min(len(arr), i + window//2 + 1)
        window_data = arr[start:end]
        hist, _ = np.histogram(window_data, bins=10, density=True)
        result[i] = entropy(hist + 1e-9)
    return result

def analyze_pdb(pdb_path: str) -> List[Dict]:
    """Main EWCL analysis function"""
    
    # Check if AlphaFold
    with open(pdb_path, 'r') as f:
        header = ''.join([f.readline() for _ in range(50)])
    is_alphafold = 'ALPHAFOLD' in header.upper() or os.path.basename(pdb_path).startswith('AF-')
    
    # Parse structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Extract residue data
    residues_data = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    bfactor = residue['CA'].get_bfactor()
                    residues_data.append({
                        'aa': residue.get_resname(),
                        'bfactor': bfactor,
                        'plddt': bfactor if is_alphafold else None,
                        'residue_id': residue.id[1],
                        'chain': chain.id
                    })
    
    if not residues_data:
        return []
    
    # Extract features
    hydropathy = [HYDROPATHY.get(r['aa'], 0) for r in residues_data]
    charge = [CHARGE.get(r['aa'], 0) for r in residues_data]
    bfactors = [r['bfactor'] for r in residues_data]
    
    # Calculate entropies
    hydro_entropy = sliding_entropy(np.array(hydropathy))
    charge_entropy = sliding_entropy(np.array(charge))
    
    # Normalize features
    hydro_norm = normalize(hydro_entropy)
    charge_norm = normalize(charge_entropy)
    bfactor_norm = normalize(bfactors)
    
    # Calculate collapse likelihood
    cl_scores = 0.35 * hydro_norm + 0.35 * charge_norm + 0.30 * bfactor_norm
    
    # Build results
    results = []
    for i, residue in enumerate(residues_data):
        results.append({
            'residue_id': residue['residue_id'],
            'chain': residue['chain'],
            'aa': residue['aa'],
            'bfactor': round(residue['bfactor'], 2),
            'plddt': round(residue['plddt'], 2) if residue['plddt'] else None,
            'cl': round(float(cl_scores[i]), 3),
            'stability': 'Unstable' if cl_scores[i] > 0.6 else 'Stable'
        })
    
    return results
