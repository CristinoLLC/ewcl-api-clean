"""
Structure Parser for EWCL-H (Hallucination Detector)
====================================================

Robust CIF/PDB parsing using gemmi for extracting residue positions and pLDDT values.
Handles both AlphaFold structures (with pLDDT) and experimental structures.
"""

import gemmi
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# Standard amino acids for compatibility check
STANDARD_AA_3LETTER = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HYP", "SEP", "TPO", "PTR", "CSO", "CME", "KCX", 
    "MLZ", "FME"
}

def is_amino_acid_residue(residue) -> bool:
    """Check if residue is an amino acid (gemmi compatibility fix)."""
    return residue.name in STANDARD_AA_3LETTER

def guess_accession_from_unit(unit_name: str) -> str:
    """Extract UniProt accession from filename."""
    # AF-P12345-F1.CIF → P12345
    m = re.search(r'AF-([A-Z0-9]+)-F\d+', unit_name.upper())
    return m.group(1) if m else unit_name.split('.')[0].upper()

def load_chains(path: str) -> List[str]:
    """Load all protein chain IDs from structure file."""
    st = gemmi.read_structure(path)
    chains = set()
    for model in st:
        for ch in model:
            # proteins only: skip ligands/water
            if any(is_amino_acid_residue(res) for res in ch):
                chains.add(ch.name)
    return sorted(chains)

def extract_residue_table(path: str, chain_id: str) -> Tuple[List[int], List[float], Optional[List[float]], bool, List[str]]:
    """
    Extract residue positions and pLDDT values from structure.
    
    Returns:
        positions (1-based), empty list (placeholder), plddt list if AF-like, is_af_like flag, warnings list
    """
    st = gemmi.read_structure(path)
    warnings = []
    pos_list, bf_list = [], []
    
    # Find the requested chain
    chain = None
    for model in st:
        if chain_id in [c.name for c in model]:
            chain = model[chain_id]
            break
    
    if chain is None:
        # Fallback: search all models
        warnings.append(f"Chain {chain_id} not found in first model; attempting any model.")
        for model in st:
            for c in model:
                if c.name == chain_id:
                    chain = c
                    break
            if chain:
                break
    
    if chain is None:
        raise ValueError(f"Chain {chain_id} not found in file.")
    
    # Extract residue data
    for res in chain:
        if not is_amino_acid_residue(res):
            continue
        
        pos = res.seqid.num
        # Calculate mean B-factor per residue
        b_values = [atom.b_iso for atom in res if atom.b_iso is not None and atom.b_iso >= 0]
        mean_b = float(np.mean(b_values)) if b_values else float("nan")
        
        pos_list.append(pos)
        bf_list.append(mean_b)
    
    # Determine if this is AlphaFold-like (pLDDT in B-factors)
    finite_b = np.array([b for b in bf_list if np.isfinite(b)])
    af_like = False
    
    if len(finite_b) > 0:
        # Heuristic: AF pLDDT values are typically in [0,100] with median > 40
        af_like = (np.max(finite_b) <= 100.0) and (np.median(finite_b) >= 40.0)
    
    plddt = bf_list if af_like else None
    return pos_list, [], plddt, af_like, warnings

def load_structure_residue_plddt(path: str, chain_id: str = None) -> Tuple[Dict[int, float], str]:
    """
    Load pLDDT values from structure file.
    
    Returns:
        dict: {res_index(1-based): mean_plddt} if available
        str: status string ("ok" or "no_confidence_available")
    """
    st = gemmi.read_structure(path)
    is_af_like = False
    res2bf = {}
    
    # If no chain specified, use first protein chain
    if chain_id is None:
        chains = load_chains(path)
        if not chains:
            return {}, "no_confidence_available"
        chain_id = chains[0]
    
    for model in st:
        chain = None
        for c in model:
            if c.name == chain_id:
                chain = c
                break
        
        if chain is None:
            continue
            
        for residue in chain:
            if not is_amino_acid_residue(residue):
                continue
                
            b_list = []
            for atom in residue:
                b = atom.b_iso
                if b is not None and b >= 0:
                    b_list.append(b)
            
            if b_list:
                mean_b = float(np.mean(b_list))
                # Heuristic: AF pLDDT values are typically in [0,100]
                if mean_b <= 100.0:
                    is_af_like = True
                
                pos = residue.seqid.num
                res2bf[pos] = mean_b
        break  # Only first model
    
    if is_af_like and len(res2bf) > 0:
        return res2bf, "ok"
    else:
        return {}, "no_confidence_available"