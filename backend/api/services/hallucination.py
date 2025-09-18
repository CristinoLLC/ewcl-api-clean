"""
EWCL-H Hallucination Detection Service
=====================================

Computes hallucination scores from EWCL predictions and pLDDT confidence values.
"""

import numpy as np
from typing import Optional, Dict, Tuple


def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_h_per_res(ewcl: np.ndarray,
                      plddt: Optional[np.ndarray],
                      lambda_h: float = 0.871,
                      tau: float = 0.5,
                      plddt_strict: float = 70.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """
    Compute hallucination scores per residue.
    
    Args:
        ewcl: EWCL disorder scores [0,1]
        plddt: pLDDT confidence scores [0,100] or None
        lambda_h: hallucination sensitivity parameter
        tau: threshold for high hallucination
        plddt_strict: threshold for "confident" pLDDT
    
    Returns:
        H scores, is_high_H flags, is_disagree flags, summary stats
    """
    if plddt is None:
        return (None, None, None, None)  # no confidence available
    
    # Normalize pLDDT to [0,1]
    plddt_norm = np.clip(plddt / 100.0, 0.0, 1.0)
    
    # Compute hallucination score
    # H is high when EWCL is high (disorder) but pLDDT is also high (confident)
    H = sigmoid(lambda_h * (ewcl - (1.0 - plddt_norm)))
    
    # Flags
    is_high = H >= tau
    is_disagree = (plddt >= plddt_strict) & is_high
    
    # Summary statistics
    stats = {
        "p95_H": float(np.nanpercentile(H, 95)),
        "frac_high_H": float(np.nanmean(is_high)),
        "frac_disagree": float(np.nanmean(is_disagree))
    }
    
    return H, is_high, is_disagree, stats