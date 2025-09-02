from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from Bio import SeqIO
import io
import numpy as np
import pandas as pd
from backend.api.utils.sequencer import parser_ewclv1
from backend.models.loader import load_all
from backend.features.ewclv1 import prepare_features_ewclv1
from pathlib import Path

router = APIRouter(prefix="/ewcl", tags=["EWCLv1-FASTA"])

# Load models once at startup
BUNDLE_DIR = Path("/app/backend_bundle")
MODELS = load_all(BUNDLE_DIR)
if "ewclv1" not in MODELS:
    raise RuntimeError("EWCLv1 model not found")

def _require_list(mb) -> list[str]:
    fi = mb.feature_info
    if isinstance(fi, dict):
        feats = fi.get("all_features")
        if isinstance(feats, list):
            return feats
    if isinstance(fi, list):
        return fi
    return []

REQUIRED_FEATURES = _require_list(MODELS["ewclv1"])

@router.post("/analyze-fasta/ewclv1")
async def analyze_fasta(file: UploadFile = File(...)):
    try:
        fasta_text = (await file.read()).decode("utf-8")
        record = next(SeqIO.parse(io.StringIO(fasta_text), "fasta"))
        seq = str(record.seq)

        df = parser_ewclv1(seq)
        if df.empty:
            return JSONResponse(content={"id": record.id if record.id else None, "model": "ewclv1", "length": 0, "residues": []})

        # Call real EWCLv1 model directly
        results = []
        for _, row in df.iterrows():
            # Convert row to features dict (excluding metadata columns)
            features = {}
            for col in df.columns:
                if col not in ["residue_index", "aa"]:
                    features[col] = float(row[col])
            
            # Prepare features for the model
            sample_data = {"features": features}
            X = prepare_features_ewclv1(sample_data, REQUIRED_FEATURES)
            
            try:
                prob = float(MODELS["ewclv1"].predict_proba(X).iloc[0])
                results.append({
                    "residue_index": int(row["residue_index"]),
                    "prob": prob
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model prediction failed for residue {row['residue_index']}: {e}")

        # Create prediction dictionary
        pred_dict = {r["residue_index"]: r["prob"] for r in results}
        
        residues = []
        for _, row in df.iterrows():
            residue_idx = int(row["residue_index"])
            prob = pred_dict.get(residue_idx, 0.0)
            confidence = 1.0 - abs(0.5 - prob)  # Distance from 0.5
            
            residues.append({
                "residue_index": residue_idx,
                "aa": row["aa"],
                "CL": round(float(prob), 4),
                "confidence": round(float(confidence), 4),
                "curvature": round(float(row["curvature"]), 4),
                "hydro_entropy": round(float(row["hydro_entropy"]), 4),
                "charge_entropy": round(float(row["charge_entropy"]), 4),
                "flips": int(row["flips"]),
            })

        return JSONResponse(content={
            "id": record.id if record.id else None,
            "model": "ewclv1",
            "length": len(seq),
            "residues": residues,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EWCLv1 FASTA analysis failed: {e}")


