import numpy as np
import json
from scipy.stats import spearmanr
import logging

def compute_metrics(results_data, cl_thresh=0.6, plddt_thresh=70, window=15):
    """
    Compute EWCL metrics from results data
    
    Args:
        results_data: Either a file path (str) or list of result dictionaries
        cl_thresh: CL threshold for mismatch detection
        plddt_thresh: pLDDT threshold for mismatch detection  
        window: Window size for local correlation
    
    Returns:
        Dict with pearson, spearman, local_mean, local_sd, n_mismatches
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
