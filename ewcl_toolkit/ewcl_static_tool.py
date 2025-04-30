def ewcl_score_protein(sequence):
    import numpy as np
    from collections import Counter

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_counts = Counter(sequence)
    total = sum(aa_counts[aa] for aa in amino_acids if aa in aa_counts)

    entropy_map = {}
    for i, aa in enumerate(sequence):
        if aa not in amino_acids:
            score = 0.0
        else:
            p = aa_counts[aa] / total if total > 0 else 0
            score = -p * np.log2(p) if p > 0 else 0
        entropy_map[i+1] = score
    return entropy_map