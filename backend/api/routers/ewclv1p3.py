from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from backend.api.utils.parser_p3 import parser_pdb_p3
from backend.models.loader import load_all
from backend.config import settings

router = APIRouter(prefix="/ewcl", tags=["EWCLv1p3-PDB"])

# Load ewclv1p3 model from bundle
_MODELS = load_all(settings.ewcl_bundle_dir)
_P3 = _MODELS.get("ewclv1p3")

@router.post("/analyze-pdb/ewclv1p3")
async def analyze_pdb(file: UploadFile = File(...)):
    if _P3 is None:
        raise HTTPException(status_code=503, detail="ewclv1p3 model not available")
    try:
        pdb_bytes = await file.read()
        df = parser_pdb_p3(pdb_bytes)

        # Prepare features DataFrame according to model requirements
        feats_required = _P3.feature_info.get("all_features", [])
        X = pd.DataFrame([{f: 0.0 for f in feats_required} for _ in range(len(df))])
        # Opportunistically project known features if names overlap
        for col in ("curvature","hydro_entropy","charge_entropy","flips"):
            if col in feats_required:
                X[col] = df[col].astype(float).values

        try:
            probs = _P3.predict_proba(X).values
        except Exception as e:
            raise HTTPException(422, f"Model inference failed: {e}")

        residues = []
        for i in range(len(df)):
            residues.append({
                "residue_index": int(df["residue_index"].iat[i]),
                "aa": str(df["aa"].iat[i]),
                "CL": float(np.clip(probs[i], 0.0, 1.0)),
                "confidence": float(1.0 - abs(0.5 - np.clip(probs[i], 0.0, 1.0))),
                "curvature": float(df["curvature"].iat[i]),
                "hydro_entropy": float(df["hydro_entropy"].iat[i]),
                "charge_entropy": float(df["charge_entropy"].iat[i]),
                "flips": int(df["flips"].iat[i]),
                "support": float(df["support"].iat[i]),
                "support_type": str(df["support_type"].iat[i]),
                "chain": str(df["chain"].iat[i]),
            })

        return JSONResponse(content={
            "id": file.filename,
            "model": "ewclv1p3",
            "length": len(residues),
            "residues": residues,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EWCLv1p3 PDB analysis failed: {e}")


