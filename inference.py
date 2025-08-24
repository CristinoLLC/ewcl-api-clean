import joblib
import numpy as np
import pandas as pd

# Load model at startup
MODEL_PATH = "models/ewclv5_full.pkl"
model_bundle = joblib.load(MODEL_PATH)
models = model_bundle["models"]
meta = model_bundle["meta"]


def predict_protein(df: pd.DataFrame, protein_id: str, model_type: str = "AF2") -> dict:
    """
    Run inference on a protein dataframe.
    df must contain features from preprocessing (plddt, bfactor, hydropathy, etc.)
    """
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


