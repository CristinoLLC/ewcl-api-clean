import numpy as np
import json
from scipy.stats import spearmanr

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

    cl = np.array([d["cl"] for d in data])
    plddt = np.array([d["plddt"] for d in data])

    # Pearson correlation
    pearson = np.corrcoef(cl, plddt)[0, 1]

    # Spearman correlation
    spearman = spearmanr(cl, plddt).correlation

    # Local Spearman (sliding window)
    local_rs = []
    for i in range(len(cl) - window + 1):
        r = spearmanr(cl[i:i+window], plddt[i:i+window]).correlation
        if not np.isnan(r):
            local_rs.append(r)

    local_mean = np.mean(local_rs) if local_rs else 0.0
    local_sd = np.std(local_rs) if local_rs else 0.0

    # Mismatches: High CL + High pLDDT
    mismatches = sum((cl >= cl_thresh) & (plddt >= plddt_thresh))

    return {
        "pearson": round(float(pearson), 3) if not np.isnan(pearson) else 0.0,
        "spearman": round(float(spearman), 3) if not np.isnan(spearman) else 0.0,
        "local_mean": round(float(local_mean), 3),
        "local_sd": round(float(local_sd), 3),
        "n_mismatches": int(mismatches),
        "total_residues": len(data)
    }
