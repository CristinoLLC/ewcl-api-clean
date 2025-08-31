from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List


def prepare_features_ewclv1(payload: Dict[str, Any], required_features: List[str]) -> pd.DataFrame:
    if "features" in payload:
        df = pd.DataFrame([payload["features"]])
    else:
        raise ValueError("No features provided. Supply 'features'.")
    for f in required_features:
        if f not in df.columns:
            df[f] = 0.0
    for c in df.columns:
        if df[c].dtype == 'O':
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)


