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
    Compute EWCL metrics from results data
    
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
    plddt = np.array([d["plddt"] for d in data])
    
    # Debug logging for first few values
    logging.info(f"Spearman inputs - CL: {cl[:5]}, pLDDT: {plddt[:5]}")

    # Pearson correlation
    pearson = np.corrcoef(cl, plddt)[0, 1]

    # Spearman correlation
    spearman, _ = spearmanr(cl, plddt)

    # Local correlation metrics
    local_rhos = []
    window_size = window
    for i in range(len(cl) - window_size + 1):
        local_rho, _ = spearmanr(cl[i:i + window_size], plddt[i:i + window_size])
        local_rhos.append(local_rho)
    local_avg = np.mean(local_rhos) if local_rhos else None
    local_std = np.std(local_rhos) if local_rhos else None

    # Mismatch detection
    mismatches = sum((cl > cl_thresh) & (plddt < plddt_thresh))

    metrics_result = {
        "pearson": round(float(pearson), 3) if not np.isnan(pearson) else None,
        "spearman": round(float(spearman), 3) if not np.isnan(spearman) else None,
        "spearman_local_avg": round(float(local_avg), 3) if local_avg is not None else None,
        "spearman_local_std": round(float(local_std), 3) if local_std is not None else None,
        "local_windows_count": len(local_rhos),
        "window_size": window_size,
        "n_mismatches": int(mismatches),
        "total_residues": len(data),
        "cl_range": [round(float(np.min(cl)), 3), round(float(np.max(cl)), 3)],
        "plddt_range": [round(float(np.min(plddt)), 3), round(float(np.max(plddt)), 3)]
    }
    
    # Add AUC and Kendall metrics if disorder labels are provided
    if disorder_labels is not None:
        auc_kendall_results = compute_auc_kendall(disorder_labels, cl)
        metrics_result.update(auc_kendall_results)
    
    return metrics_result
