from typing import Dict, List, Optional, Tuple, Union
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from io import StringIO
import numpy as np
import json
import requests
import subprocess
import os
import tempfile
import re

def extract_sequence_from_pdb(pdb_text: str) -> Tuple[str, Dict[int, str]]:
    """Extract amino acid sequence from PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_text))
    
    sequence = ""
    residue_mapping = {}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # Standard amino acid
                    try:
                        aa = seq1(residue.get_resname())
                        sequence += aa
                        residue_mapping[residue.get_id()[1]] = aa
                    except:
                        continue
            break  # Only process first chain
        break  # Only process first model
            
    return sequence, residue_mapping

def compute_bfactor_scores(pdb_text: str) -> Dict[int, float]:
    """Calculate entropy scores from B-factors in PDB structure"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_text))
    
    residue_scores = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                # Get B-factor of all atoms in the residue
                b_factors = [atom.get_bfactor() for atom in residue if atom.get_bfactor() < 100.0]
                if not b_factors:
                    continue
                avg_b = np.mean(b_factors)
                
                # Normalize and convert to entropy-like score (scale 0–1)
                norm_score = round(min(avg_b / 100.0, 1.0), 4)
                
                # Use res id as global key
                residue_scores[residue.get_id()[1]] = norm_score
    
    return residue_scores

def run_iupred(sequence: str) -> Dict[int, float]:
    """Run IUPred2A for disorder prediction via web API"""
    url = "https://iupred2a.elte.hu/iupred2a"
    payload = {
        'seq': sequence,
        'iupred2': 'long',  # Use long disorder prediction
    }
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            return {}
            
        # Parse IUPred results (tab-separated values)
        scores = {}
        for line in response.text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    pos = int(parts[0])
                    score = float(parts[2])
                    scores[pos] = score
                except (ValueError, IndexError):
                    continue
                    
        return scores
        
    except Exception as e:
        print(f"Error running IUPred: {e}")
        return {}

def parse_alphafold_json(json_text: str) -> Dict[int, float]:
    """Extract pLDDT scores from AlphaFold JSON and convert to entropy scores"""
    try:
        data = json.loads(json_text)
        
        # Different AF JSON formats may store pLDDT differently
        plddt_scores = {}
        
        # Try to find pLDDT values
        if "plddt" in data:
            plddt_array = data["plddt"]
        elif "confidenceScore" in data:
            plddt_array = data["confidenceScore"]["plddt"]
        else:
            # Try to find pLDDT values in a deeply nested structure
            for key in data:
                if isinstance(data[key], dict) and "plddt" in data[key]:
                    plddt_array = data[key]["plddt"]
                    break
            else:
                return {}
        
        # Convert pLDDT to entropy score (higher pLDDT = lower entropy)
        for i, score in enumerate(plddt_array, 1):
            # Invert and normalize: 100 pLDDT → 0 entropy, 0 pLDDT → 1 entropy
            entropy_score = round(1 - (score / 100), 4)
            plddt_scores[i] = entropy_score
            
        return plddt_scores
        
    except Exception as e:
        print(f"Error parsing AlphaFold JSON: {e}")
        return {}

def combine_entropy_sources(
    residue_ids: List[int],
    b_factor_scores: Dict[int, float] = None,
    disorder_scores: Dict[int, float] = None, 
    plddt_scores: Dict[int, float] = None,
    weights: Dict[str, float] = None
) -> Dict[int, float]:
    """Combine multiple entropy sources with weighted averaging"""
    if weights is None:
        # Default weights if not specified
        weights = {
            "b_factor": 0.4,
            "disorder": 0.3,
            "plddt": 0.3
        }
    
    # Normalize weights to sum to 1.0
    total_weight = sum(w for src, w in weights.items() 
                     if (src == "b_factor" and b_factor_scores) or
                        (src == "disorder" and disorder_scores) or
                        (src == "plddt" and plddt_scores))
                        
    if total_weight == 0:
        return {}
        
    norm_weights = {k: v/total_weight for k, v in weights.items()}
    
    combined_scores = {}
    for res_id in residue_ids:
        score_sum = 0
        
        if b_factor_scores and res_id in b_factor_scores:
            score_sum += b_factor_scores[res_id] * norm_weights.get("b_factor", 0)
            
        if disorder_scores and res_id in disorder_scores:
            score_sum += disorder_scores[res_id] * norm_weights.get("disorder", 0)
            
        if plddt_scores and res_id in plddt_scores:
            score_sum += plddt_scores[res_id] * norm_weights.get("plddt", 0)
            
        combined_scores[res_id] = round(score_sum, 4)
        
    return combined_scores

def detect_input_type(input_text: str) -> str:
    """Detect if input is PDB, FASTA, or AlphaFold JSON"""
    # Check for PDB format
    if "ATOM  " in input_text or "HETATM" in input_text:
        return "pdb"
        
    # Check for AlphaFold JSON
    if input_text.strip().startswith("{") and "plddt" in input_text:
        return "alphafold"
        
    # Check for FASTA format
    if ">" in input_text and any(aa in input_text for aa in "ACDEFGHIKLMNPQRSTVWY"):
        return "fasta"
        
    # Default to PDB if unsure
    return "pdb"

def compute_ewcl_scores(input_text: str, weights: Dict[str, float] = None) -> Dict[int, float]:
    """
    Compute EWCL scores from input structure/sequence using multiple entropy sources
    
    Args:
        input_text: PDB file content, FASTA sequence, or AlphaFold JSON
        weights: Dictionary of weights for each entropy source
        
    Returns:
        Dictionary mapping residue IDs to EWCL scores
    """
    input_type = detect_input_type(input_text)
    
    # Initialize scores
    b_factor_scores = {}
    disorder_scores = {}
    plddt_scores = {}
    
    # Extract sequence for IUPred if needed
    sequence = ""
    residue_mapping = {}
    
    # Process based on input type
    if input_type == "pdb":
        # Get B-factor scores
        b_factor_scores = compute_bfactor_scores(input_text)
        
        # Extract sequence for IUPred
        sequence, residue_mapping = extract_sequence_from_pdb(input_text)
        
        # Run IUPred on the sequence
        if sequence:
            raw_disorder_scores = run_iupred(sequence)
            # Map scores back to PDB residue IDs
            for seq_pos, score in raw_disorder_scores.items():
                for res_id, aa in residue_mapping.items():
                    if seq_pos == list(residue_mapping.keys()).index(res_id) + 1:
                        disorder_scores[res_id] = score
                        break
    
    elif input_type == "alphafold":
        # Parse AlphaFold JSON for pLDDT scores
        plddt_scores = parse_alphafold_json(input_text)
        
    elif input_type == "fasta":
        # Extract sequence from FASTA
        sequence = ""
        for line in input_text.strip().split("\n"):
            if not line.startswith(">"):
                sequence += line.strip()
        
        # Run IUPred on the sequence
        if sequence:
            disorder_scores = run_iupred(sequence)
            # For FASTA, residue IDs are just positions in sequence
            residue_mapping = {i+1: aa for i, aa in enumerate(sequence)}
            
    # Combine available entropy sources
    residue_ids = set(b_factor_scores.keys()) | set(disorder_scores.keys()) | set(plddt_scores.keys())
    combined_scores = combine_entropy_sources(
        list(residue_ids), 
        b_factor_scores, 
        disorder_scores, 
        plddt_scores,
        weights
    )
    
    return combined_scores

def compute_ewcl_scores_from_pdb(pdb_text: str) -> Dict[int, float]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_text))
    scores = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                b_factors = [atom.get_bfactor() for atom in residue if atom.get_bfactor() < 100.0]
                if not b_factors:
                    continue
                avg_b = np.mean(b_factors)
                scores[residue.get_id()[1]] = round(min(avg_b / 100.0, 1.0), 4)
    return scores

def compute_ewcl_scores_from_alphafold_json(json_bytes: bytes) -> Dict[int, float]:
    data = json.loads(json_bytes)
    plddt = data.get("plddt", [])
    scores = {i + 1: round(1.0 - (val / 100), 4) for i, val in enumerate(plddt)}
    return scores

def compute_ewcl_scores_from_sequence(fasta_text: str) -> Dict[int, float]:
    lines = fasta_text.splitlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    scores = {}
    for i, _ in enumerate(sequence):
        x = abs((len(sequence) / 2) - i) / (len(sequence) / 2)
        score = round(0.2 + (0.8 * (1 - x)), 4)  # entropy high in center
        scores[i + 1] = score
    return scores