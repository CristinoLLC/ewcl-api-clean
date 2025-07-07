import numpy as np
import pandas as pd

def compute_hallucination(residues_data, mismatch_threshold=0.3):
    """
    Computes hallucination scores using EWCL vs pLDDT/B-factor mismatch.
    
    Args:
        residues_data: List of residue dictionaries from EWCL analysis
        mismatch_threshold: Threshold for significant mismatch detection
    
    Returns:
        List of residue dictionaries with added hallucination metrics
    """
    df = pd.DataFrame(residues_data)
    
    # Normalize pLDDT/B-factor to 0-1 range for comparison with CL
    if 'bfactor' in df.columns:
        # For AlphaFold: pLDDT is already 0-100, normalize to 0-1
        # For X-ray: B-factors vary widely, use percentile normalization
        plddt_normalized = df['bfactor'] / 100.0 if df['bfactor'].max() <= 100 else \
                          (df['bfactor'] - df['bfactor'].min()) / (df['bfactor'].max() - df['bfactor'].min())
    else:
        plddt_normalized = pd.Series([0.5] * len(df))  # Default if no B-factor data
    
    # Step 1: Compute mismatch between EWCL and normalized confidence
    mismatch = np.abs(df['cl'] - plddt_normalized)
    
    # Step 2: Add mismatch metrics
    df['plddt_normalized'] = plddt_normalized
    df['mismatch'] = mismatch
    df['mismatch_flag'] = mismatch > mismatch_threshold
    
    # Step 3: Compute hallucination score using mismatch and curvature
    normalized_curvature = (df['curvature'] - df['curvature'].min()) / \
                          (df['curvature'].max() - df['curvature'].min() + 1e-9)
    
    df['hallucination_score'] = (
        0.6 * df['mismatch'] +
        0.4 * normalized_curvature
    ).clip(0, 1)
    
    # Step 4: Flag as hallucinated if score > 0.75
    df['hallucinated'] = df['hallucination_score'] > 0.75
    
    return df.to_dict(orient='records')
