from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def validate_ewcl_scores(
    ewcl_scores: Dict[int, float],
    disprot_labels: Dict[int, int],
    threshold: float = 0.297
) -> Dict[str, float]:
    """
    Evaluate EWCL scores against DisProt binary disorder annotations.
    Assumes scores are normalized to [0, 1].
    """
    # Intersect residues
    residues = sorted(set(ewcl_scores) & set(disprot_labels))
    if not residues:
        return {"error": "No overlapping residues for validation."}

    # Create label and prediction arrays
    y_true = [disprot_labels[r] for r in residues]
    y_pred = [1 if ewcl_scores[r] >= threshold else 0 for r in residues]
    y_score = [ewcl_scores[r] for r in residues]

    try:
        return {
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_true, y_score), 4),
            "matching_residues": len(residues)
        }
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}