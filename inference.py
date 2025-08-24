import os
import joblib
import numpy as np
import pandas as pd

# Lazy model holder
MODEL_PATH = os.environ.get("EWCLV5_MODEL_PATH", os.path.join("models", "ewclv5_full.pkl"))
_MODEL_BUNDLE = None
_MODELS = None
_META = None


def _load_model_once():
    global _MODEL_BUNDLE, _MODELS, _META
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE, _MODELS, _META
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"EWCL V5 model not found at '{MODEL_PATH}'. Set EWCLV5_MODEL_PATH or deploy models/ewclv5_full.pkl"
        )
    print(f"Loading EWCL V5 model from {MODEL_PATH} ...")
    bundle = joblib.load(MODEL_PATH)
    _MODEL_BUNDLE = bundle
    _MODELS = bundle["models"]
    _META = bundle["meta"]
    return _MODEL_BUNDLE, _MODELS, _META


def predict_protein(df: pd.DataFrame, protein_id: str, model_type: str = "AF2") -> dict:
    """
    Run inference on a protein dataframe.
    df must contain features from preprocessing (plddt, bfactor, hydropathy, etc.)
    """
    bundle, models, meta = _load_model_once()
    X_cols = meta.get("X_cols", [])
    if not X_cols:
        raise ValueError("Model bundle missing 'X_cols' metadata")

    X = df[X_cols].astype(float)
    preds = np.zeros(len(df), dtype=float)
    for m in models:
        preds += m["cal"].predict_proba(X)[:, 1]
    preds /= max(1, len(models))

    result = {
        "protein_id": protein_id,
        "model_info": {
            "model_name": "EWCL V5 (Hybrid ML)",
            "version": str(meta.get("version", "1.0")),
            "notes": meta.get("notes", "Trained on AF2, X-ray, and NMR datasets with Flip/Fail flags."),
        },
        "flags": {
            "is_flip_protein": bool(df.get("is_flip_protein", [0])[0]),
            "is_fail_protein": bool(df.get("is_fail_protein", [0])[0]),
        },
        "sequence": "".join(df["aa"].tolist()) if "aa" in df.columns else "",
        "residues": [],
    }

    for i, row in df.iterrows():
        features = {
            "plddt": row.get("plddt", None),
            "bfactor": row.get("bfactor", None),
            "rmsf": row.get("rmsf", None),
            "curvature": row.get("curvature", None),
            "hydropathy": row.get("hydropathy", None),
            "charge": row.get("charge", None),
            "hydro_entropy": row.get("hydro_entropy", None),
            "charge_entropy": row.get("charge_entropy", None),
            "flexibility": (
                row.get("bfactor", None) if model_type == "Xray"
                else row.get("plddt", None) if model_type == "AF2"
                else row.get("rmsf", None)
            ),
        }

        result["residues"].append({
            "residue_index": int(row.get("residue_index", i + 1)),
            "amino_acid": row.get("aa", None),
            "features": features,
            "prediction": {
                "disorder_score": float(preds[i]),
                "confidence_percentage": round(100 * float(preds[i]), 2),
            },
        })

    return result


