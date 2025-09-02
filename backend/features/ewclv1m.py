from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List

AA20 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
PSSM_DERIVED = ['pssm_entropy','pssm_max_score','pssm_variance','has_pssm_data']

DEFAULTS = {
    **{aa: 0.0 for aa in AA20},
    'pssm_entropy': 3.0,
    'pssm_max_score': 0.1,
    'pssm_variance': 0.0,
    'has_pssm_data': 0
}


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == 'O':
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)


def _set_pssm_defaults(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    need_pssm = [c for c in (AA20 + PSSM_DERIVED) if c in required]
    missing = [c for c in need_pssm if c not in df.columns]
    for c in missing:
        df[c] = DEFAULTS[c]
    return df


def prepare_features_ewclv1m(payload: Dict[str, Any], required_features: List[str]) -> pd.DataFrame:
    if "features" not in payload:
        raise ValueError("Missing 'features' object")
    
    # Start with the provided features
    features_dict = dict(payload["features"])
    
    # Add PSSM defaults if needed
    if bool(payload.get("sequence_only", False)):
        need_pssm = [c for c in (AA20 + PSSM_DERIVED) if c in required_features]
        for c in need_pssm:
            if c not in features_dict:
                features_dict[c] = DEFAULTS[c]
    
    # Add any missing required features with default 0.0
    for f in required_features:
        if f not in features_dict:
            features_dict[f] = 0.0
    
    # Create DataFrame in one go to avoid fragmentation
    df = pd.DataFrame([features_dict])
    df = _ensure_numeric(df)
    return df


