def ewcl_score_protein(sequence: str):
    """
    Calculate entropy-weighted collapse likelihood scores for a protein sequence.
    Uses Shannon entropy based on global amino acid frequency distribution.
    
    Args:
        sequence: Protein sequence string (single letter amino acid codes)
        
    Returns:
        Dictionary mapping residue positions (1-indexed) to normalized EWCL scores [0,1]
    """
    import numpy as np
    from collections import Counter

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_counts = Counter(sequence)
    total = sum(aa_counts[aa] for aa in amino_acids if aa in aa_counts)

    entropy_map = {}
    for i, aa in enumerate(sequence):
        if aa not in amino_acids:
            # Handle non-standard amino acids gracefully
            score = 0.0
        else:
            p = aa_counts[aa] / total if total > 0 else 0
            score = -p * np.log2(p) if p > 0 else 0
        entropy_map[i+1] = score

    # Normalize scores to [0, 1] range
    if entropy_map:
        max_entropy = max(entropy_map.values())
        if max_entropy > 0:
            for pos in entropy_map:
                entropy_map[pos] = round(entropy_map[pos] / max_entropy, 4)

    return entropy_map


def parse_fasta(fasta_str: str) -> str:
    """
    Parse FASTA format string to extract protein sequence.
    
    Args:
        fasta_str: FASTA formatted string with header and sequence
        
    Returns:
        Clean protein sequence string without headers or whitespace
    """
    lines = fasta_str.strip().split("\n")
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def ewcl_from_fasta(fasta_str: str):
    """
    Calculate EWCL scores directly from FASTA format input.
    
    Args:
        fasta_str: FASTA formatted string
        
    Returns:
        Dictionary mapping residue positions to normalized EWCL scores
    """
    sequence = parse_fasta(fasta_str)
    return ewcl_score_protein(sequence)