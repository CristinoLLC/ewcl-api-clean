import numpy as np
import json
from scipy.stats import spearmanr, kendalltau, pearsonr
import logging
from typing import List, Dict

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("sklearn not available - AUC metrics will be disabled")

def compute_auc_kendall(y_true, y_score):
    """
    Compute AUC and Kendall's tau metrics for binary classification
    """
    result = {}
    
    # AUC computation
    if HAS_SKLEARN:
        try:
            result['auc'] = round(float(roc_auc_score(y_true, y_score)), 3)
        except ValueError as e:
            result['auc'] = None
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

def compute_metrics(results_data, cl_thresh=0.6, plddt_thresh=70, window=15, disorder_labels=None, mode="collapse"):
    """
    Compute EWCL metrics from results data with proper handling of reverse mode
    
    Args:
        results_data: Either a file path (str) or list of result dictionaries
        cl_thresh: CL threshold for mismatch detection
        plddt_thresh: pLDDT threshold for mismatch detection  
        window: Window size for local correlation
        disorder_labels: Optional binary labels for AUC/Kendall computation
        mode: "collapse" or "reverse" mode for score interpretation
    
    Returns:
        Dict with correlation metrics and optionally AUC/Kendall metrics
    """
    # Handle both file path and direct data input
    if isinstance(results_data, str):
        with open(results_data) as f:
            data = json.load(f)["results"]
    else:
        data = results_data

    # Extract values for metrics computation
    cl_values = []
    plddt_values = []
    b_factor_values = []
    binary_disorder = []

    for entry in data:
        # Get the correct CL score based on mode
        raw_cl = entry.get("raw_cl", 0)
        cl = entry.get("cl", 0)  # This should already be mode-adjusted in the API
        
        plddt = entry.get("plddt")
        bfac = entry.get("b_factor")

        if plddt is not None:
            cl_values.append(cl)
            plddt_values.append(plddt)
            b_factor_values.append(bfac if bfac is not None else plddt)
            
            # Using pLDDT < 70 as disorder proxy for AUC
            binary_disorder.append(1 if plddt < plddt_thresh else 0)

    if not cl_values:
        return {"error": "No valid data for metrics computation"}

    cl_array = np.array(cl_values)
    plddt_array = np.array(plddt_values)
    binary_disorder = np.array(binary_disorder)

    # Correlation metrics
    metrics_result = {}
    
    try:
        if len(set(cl_values)) > 1 and len(set(plddt_values)) > 1:
            metrics_result["pearson"] = round(float(pearsonr(cl_array, plddt_array)[0]), 4)
            metrics_result["spearman"] = round(float(spearmanr(cl_array, plddt_array)[0]), 4)
            metrics_result["kendall_tau"] = round(float(kendalltau(cl_array, plddt_array)[0]), 4)
        else:
            metrics_result["pearson"] = None
            metrics_result["spearman"] = None
            metrics_result["kendall_tau"] = None
            logging.warning("Insufficient variation in data for correlation metrics")
    except Exception as e:
        logging.warning(f"Correlation computation failed: {e}")
        metrics_result["pearson"] = None
        metrics_result["spearman"] = None
        metrics_result["kendall_tau"] = None

    # AUC computation with binary disorder labels
    try:
        if HAS_SKLEARN and len(set(binary_disorder)) > 1:
            metrics_result["auc_pseudo_plddt"] = round(float(roc_auc_score(binary_disorder, cl_array)), 4)
        else:
            metrics_result["auc_pseudo_plddt"] = None
    except Exception as e:
        metrics_result["auc_pseudo_plddt"] = None
        logging.warning(f"AUC computation failed: {e}")

    # Local correlation computation
    local_rhos = []
    window_size = min(window, len(cl_array))
    
    if window_size >= 3:
        for i in range(len(cl_array) - window_size + 1):
            cl_slice = cl_array[i:i + window_size]
            plddt_slice = plddt_array[i:i + window_size]
            
            if len(set(cl_slice)) > 1 and len(set(plddt_slice)) > 1:
                try:
                    rho, _ = spearmanr(cl_slice, plddt_slice)
                    if not np.isnan(rho):
                        local_rhos.append(rho)
                except:
                    continue

    if local_rhos:
        metrics_result["spearman_local_avg"] = round(float(np.mean(local_rhos)), 3)
        metrics_result["spearman_local_std"] = round(float(np.std(local_rhos)), 3)
    else:
        metrics_result["spearman_local_avg"] = None
        metrics_result["spearman_local_std"] = None

    # Mismatches calculation
    mismatches = sum((cl_array >= cl_thresh) & (plddt_array >= plddt_thresh))
    
    metrics_result.update({
        "local_windows_count": len(local_rhos),
        "window_size": window_size,
        "n_mismatches": int(mismatches),
        "total_residues": len(data),
        "cl_range": [round(float(np.min(cl_array)), 3), round(float(np.max(cl_array)), 3)],
        "plddt_range": [round(float(np.min(plddt_array)), 3), round(float(np.max(plddt_array)), 3)],
        "mode": mode
    })
    
    # Add AUC and Kendall metrics if disorder labels are provided
    if disorder_labels is not None:
        auc_kendall_results = compute_auc_kendall(disorder_labels, cl_array)
        metrics_result.update(auc_kendall_results)
    
    return metrics_result
