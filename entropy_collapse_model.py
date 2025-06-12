import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1 as three_to_one
from typing import Dict
import os
import logging

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

def infer_entropy_from_pdb(path: str, reverse: bool = False) -> Dict:
    logger.info("🧠 Real entropy model processing PDB file")
    
    try:
        residues = extract_residues(path)
        logger.info(f"📊 Extracted {len(residues)} residues from PDB")

        if not residues:
            logger.warning("❌ No residues found")
            return {
                "status": "error",
                "mode": "reverse" if reverse else "normal",
                "reverse": reverse,
                "results": [],
                "message": "No valid residues found"
            }

        entropy_scores = compute_ewcl_entropy(residues)

        # ✅ Normalize entropy (Y)
        min_score = min(entropy_scores)
        max_score = max(entropy_scores)
        normalized_scores = [
            (s - min_score) / (max_score - min_score) if max_score > min_score else 0.5
            for s in entropy_scores
        ]

        results = []
        for i, (res, raw, norm) in enumerate(zip(residues, entropy_scores, normalized_scores)):
            try:
                aa = three_to_one(res.resname)
            except Exception:
                aa = "X"

            score = 1.0 - norm if reverse else norm  # reverse entropy if needed
            results.append({
                "residue_id": int(res.id[1]),
                "aa": aa,
                "ewcl_score": round(score, 6),         # normalized (0–1), reverse if requested
                "ewcl_score_raw": round(raw, 6),       # original entropy (e.g. 2.4–3.2)
                "resi": int(res.id[1]),
                "resiLabel": str(res.id[1])
            })

        logger.info(f"✅ Successfully processed {len(results)} residues")
        return {
            "status": "success",
            "mode": "reverse" if reverse else "normal",
            "reverse": reverse,
            "scores": results
        }

    except Exception as e:
        logger.error(f"❌ Error in entropy analysis: {e}")
        return {
            "status": "error",
            "mode": "reverse" if reverse else "normal",
            "reverse": reverse,
            "results": [],
            "message": str(e)
        }