"""
EWCL utility functions for feature extraction and processing
"""

import pandas as pd
import numpy as np
from models.enhanced_ewcl_af import compute_curvature_features

def extract_features_from_pdb(pdb_path: str) -> pd.DataFrame:
    """
    Extract features from PDB file using the enhanced EWCL physics extractor
    Returns DataFrame with required features for various ML models
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        DataFrame with columns: position, chain, aa, rev_cl, entropy, slope, 
        curvature, mean_rev_cl_7, mean_entropy_7, is_boundary
    """
    try:
        # Use the enhanced physics extractor
        rows = compute_curvature_features(pdb_path)
        df = pd.DataFrame(rows)
        
        if df.empty:
            raise ValueError("No features extracted from PDB")
        
        # Basic field mapping
        field_mapping = {
            "residue_id": "position",
            "hydro_entropy": "entropy",
            "bfactor_curv": "curvature"
        }
        
        for old_name, new_name in field_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Calculate reverse collapse likelihood
        if "cl" in df.columns:
            df["rev_cl"] = 1.0 - df["cl"]
        
        # Calculate slope (gradient of collapse likelihood)
        if "cl" in df.columns:
            df["slope"] = np.gradient(df["cl"])
        
        # Calculate rolling means
        if "rev_cl" in df.columns:
            df["mean_rev_cl_7"] = df["rev_cl"].rolling(window=7, center=True).mean().fillna(df["rev_cl"])
        
        if "entropy" in df.columns:
            df["mean_entropy_7"] = df["entropy"].rolling(window=7, center=True).mean().fillna(df["entropy"])
        
        # Identify boundary residues (first and last 5 residues)
        df["is_boundary"] = 0
        if len(df) > 10:
            df.iloc[:5, df.columns.get_loc("is_boundary")] = 1
            df.iloc[-5:, df.columns.get_loc("is_boundary")] = 1
        
        # Ensure required columns exist with defaults
        required_columns = {
            "position": range(1, len(df) + 1),
            "chain": "A",
            "aa": "UNK",
            "rev_cl": 0.0,
            "entropy": 0.0,
            "slope": 0.0,
            "curvature": 0.0,
            "mean_rev_cl_7": 0.0,
            "mean_entropy_7": 0.0,
            "is_boundary": 0
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        return df
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        raise

def validate_disprot_features(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required features for DisProt prediction
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if all required features are present
    """
    required_features = ["rev_cl", "entropy", "slope", "curvature", 
                        "mean_rev_cl_7", "mean_entropy_7", "is_boundary"]
    
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing required features: {missing_features}")
        return False
    
    return True

def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Normalize specified feature columns to 0-1 range
    
    Args:
        df: Input DataFrame
        feature_cols: List of column names to normalize
        
    Returns:
        DataFrame with normalized features
    """
    df_normalized = df.copy()
    
    for col in feature_cols:
        if col in df_normalized.columns:
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            col_range = col_max - col_min
            
            if col_range > 0:
                df_normalized[col] = (df_normalized[col] - col_min) / col_range
            else:
                df_normalized[col] = 0.0
    
    return df_normalized
