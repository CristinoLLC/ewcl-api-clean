from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def validate_ewcl_scores(
    ewcl_scores: Dict[int, float],
    disprot_labels: Dict[int, int],
    threshold: float = 0.297
) -> Dict[str, float]:
    residues = sorted(set(ewcl_scores.keys()) & set(disprot_labels.keys()))
    if not residues:
        return {"error": "No overlapping residues for validation."}

    y_true = [disprot_labels[r] for r in residues]
    y_pred = [1 if ewcl_scores[r] >= threshold else 0 for r in residues]
    y_score = [ewcl_scores[r] for r in residues]

    return {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "matching_residues": len(residues)
    }