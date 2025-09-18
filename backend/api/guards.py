"""
Input validation guards for EWCL-H API
=====================================

File size limits, format validation, and safety checks.
"""

import os
from typing import List

# Configuration - Extended to support AlphaFold filenames
ALLOWED_SUFFIX = {".cif", ".mmcif", ".pdb", ".cif.gz", ".pdb.gz"}
MAX_BYTES = 100 * 1024 * 1024  # 100 MB


def guard_uploaded_path(path: str):
    """Validate uploaded file path and size."""
    # Check file extension - handle complex AlphaFold filenames
    filename = os.path.basename(path).lower()
    
    # Handle cases like "AF-P37840-F1-model_v4.cif"
    valid_extension = any(filename.endswith(suffix) for suffix in ALLOWED_SUFFIX)
    
    if not valid_extension:
        raise ValueError(f"Unsupported file extension. File: {filename}. Allowed: {sorted(ALLOWED_SUFFIX)}")
    
    # Check file size
    size = os.path.getsize(path)
    if size > MAX_BYTES:
        raise ValueError(f"File too large: {size} bytes (limit {MAX_BYTES}).") 


def validate_chains(requested: List[str], available: List[str]) -> List[str]:
    """Validate requested chains against available chains."""
    if not requested:
        return available
    
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f"Chains not found: {missing}. Available: {available}")
    
    return requested


def validate_hallucination_params(lambda_h: float, tau: float, plddt_strict: float):
    """Validate hallucination detection parameters."""
    if not (0.1 <= lambda_h <= 5.0):
        raise ValueError(f"lambda_h must be between 0.1 and 5.0, got {lambda_h}")
    
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be between 0.0 and 1.0, got {tau}")
    
    if not (0.0 <= plddt_strict <= 100.0):
        raise ValueError(f"plddt_strict must be between 0.0 and 100.0, got {plddt_strict}")