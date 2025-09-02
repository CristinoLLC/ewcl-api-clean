from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List


def prepare_features_ewclv1(payload: Dict[str, Any], required_features: List[str]) -> pd.DataFrame:
    if "features" not in payload:
        raise ValueError("No features provided. Supply 'features'.")
    
    # Start with the provided features
    features_dict = dict(payload["features"])
    
    # Add any missing required features with default 0.0
    for f in required_features:
        if f not in features_dict:
            features_dict[f] = 0.0
    
    # Create DataFrame in one go to avoid fragmentation
    df = pd.DataFrame([features_dict])
    
    # Ensure numeric types
    for c in df.columns:
        if df[c].dtype == 'O':
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)


