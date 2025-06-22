import numpy as np
import json
from scipy.stats import spearmanr, kendalltau
import logging

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("sklearn not available - AUC metrics will be disabled")

def compute_auc_kendall(y_true, y_score):
    """
    Compute AUC and Kendall's tau metrics for binary classification
    
    Args:
        y_true: Binary labels (0/1 or False/True)
        y_score: Prediction scores (e.g., CL scores)
    
    Returns:
        Dict with auc, kendall_tau, kendall_p
    """
    result = {}
    
    # AUC computation
    if HAS_SKLEARN:
        try:
            result['auc'] = round(float(roc_auc_score(y_true, y_score)), 3)
        except ValueError as e:
            result['auc'] = None  # happens if only one class is present
            logging.warning(f"AUC computation failed: {e}")
    else:
        result['auc'] = None

    # Kendall's tau computation
    try:
        tau, p = kendalltau(y_true, y_score)
        result['kendall_tau'] = round(float(tau), 3) if not np.isnan(tau) else None
        result['kendall_p'] = round(float(p), 6) if not np.isnan(p) else None
    except Exception as e:
        result['kendall_tau'] = None
        result['kendall_p'] = None
        logging.warning(f"Kendall's tau computation failed: {e}")

    return result

def compute_metrics(results_data, cl_thresh=0.6, plddt_thresh=70, window=15, disorder_labels=None):
    """
    Compute EWCL metrics from results data with graceful handling of missing data
    
    Args:
        results_data: Either a file path (str) or list of result dictionaries
        cl_thresh: CL threshold for mismatch detection
        plddt_thresh: pLDDT threshold for mismatch detection  
        window: Window size for local correlation
        disorder_labels: Optional binary labels for AUC/Kendall computation
    
    Returns:
        Dict with correlation metrics and optionally AUC/Kendall metrics
    """
    # Handle both file path and direct data input
    if isinstance(results_data, str):
        with open(results_data) as f:
            data = json.load(f)["results"]
    else:
        data = results_data

    # Extract raw arrays - DO NOT sort or pre-normalize
    cl = np.array([d["cl"] for d in data])
    
    # Handle missing pLDDT data gracefully
    plddt_values = [d.get("plddt") for d in data]
    has_plddt = any(p is not None for p in plddt_values)
    
    if has_plddt:
        plddt = np.array([p if p is not None else 50.0 for p in plddt_values])
    else:
        plddt = None
        logging.warning("No pLDDT data available - correlation metrics will be skipped")

    # Pearson correlation
    if plddt is not None:
        try:
            pearson = np.corrcoef(cl, plddt)[0, 1]
        except:
            pearson = float("nan")
    else:
        pearson = None

    # Spearman correlation
