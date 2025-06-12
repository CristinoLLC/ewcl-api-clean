import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1 as three_to_one
from typing import Dict
import os
import logging
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

def extract_residues(file_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    model = structure[0]
    residues = [res for chain in model for res in chain if res.id[0] == " "]
    return residues

def compute_ewcl_entropy(residues, window=5):
    scores = []
    for i in range(len(residues)):
        window_res = residues[max(0, i - window):min(len(residues), i + window + 1)]
        counts = {}
        for res in window_res:
            try:
                aa = three_to_one(res.resname)
                counts[aa] = counts.get(aa, 0) + 1
            except Exception:
                continue
        total = sum(counts.values())
        probs = [v / total for v in counts.values()] if total > 0 else []
        entropy = -sum(p * np.log2(p) for p in probs if p > 0) if probs else 0
        scores.append(entropy)
    return scores

def normalize_all_methods(raw_scores):
    scores = np.array(raw_scores)

    # Reverse raw scores
    scores = scores.max() - scores

    # Min-Max Normalization
    min_val, max_val = scores.min(), scores.max()
    minmax = (scores - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(scores)

    # Z-score Normalization
    mean, std = scores.mean(), scores.std()
    zscore = (scores - mean) / std if std != 0 else np.zeros_like(scores)

    # Quantile Normalization (0-1 scale based on rank)
    quantile = (rankdata(scores, method='average') - 1) / (len(scores) - 1) if len(scores) > 1 else np.zeros_like(scores)

    # Return as list of dicts per residue
    return [
        {
            "residue_id": i + 1,
            "ewcl_score_raw": float(raw),
            "ewcl_minmax": float(mm),
            "ewcl_zscore": float(z),
            "ewcl_quantile": float(q),
        }
        for i, (raw, mm, z, q) in enumerate(zip(raw_scores, minmax, zscore, quantile))
    ]

def infer_entropy_from_pdb(path: str) -> Dict:
    logger.info("🧠 Reverse entropy model processing PDB file")

    try:
        residues = extract_residues(path)
        logger.info(f"📊 Extracted {len(residues)} residues from PDB")

        if not residues:
            logger.warning("❌ No residues found")
            return {
                "status": "error",
                "mode": "reverse",
                "reverse": True,
                "results": [],
                "message": "No valid residues found"
            }

        entropy_scores = compute_ewcl_entropy(residues)

        # Normalize using all methods
        normalized_results = normalize_all_methods(entropy_scores)

        # Add amino acid and residue metadata
        for res, norm in zip(residues, normalized_results):
            try:
                aa = three_to_one(res.resname)
            except Exception:
                aa = "X"

            norm.update({
                "aa": aa,
                "resi": int(res.id[1]),
                "resiLabel": str(res.id[1])
            })

        logger.info(f"✅ Successfully processed {len(normalized_results)} residues")
        return {
            "status": "success",
            "mode": "reverse",
            "reverse": True,
            "scores": normalized_results
        }

    except Exception as e:
        logger.error(f"❌ Error in reverse entropy analysis: {e}")
        return {
            "status": "error",
            "mode": "reverse",
            "reverse": True,
            "results": [],
            "message": str(e)
        }