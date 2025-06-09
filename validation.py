from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def validate_ewcl_scores(
    ewcl_scores: Dict[int, float],
    disprot_labels: Dict[int, int],
    threshold: float = 0.297
) -> Dict[str, float]:
    """
    Evaluate EWCL scores against DisProt binary disorder