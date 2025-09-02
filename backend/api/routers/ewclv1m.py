from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from Bio import SeqIO
import io
import numpy as np
import httpx
from backend.api.utils.sequencer import parser_ewclv1m

router = APIRouter(prefix="/ewcl", tags=["EWCLv1m-FASTA"])

@router.post("/analyze-fasta/ewclv1m")
async def analyze_fasta(file: UploadFile = File(...)):
    try:
        fasta_text = (await file.read()).decode("utf-8")
        record = next(SeqIO.parse(io.StringIO(fasta_text), "fasta"))
        seq = str(record.seq)

        df = parser_ewclv1m(seq)
        if df.empty:
            return JSONResponse(content={"id": record.id if record.id else None, "model": "ewclv1m", "length": 0, "residues": []})

        # Call real EWCLv1-M model via internal API
        samples = []
        for _, row in df.iterrows():
            # Convert row to features dict (excluding metadata columns)
            features = {}
            for col in df.columns:
                if col not in ["residue_index", "aa"]:
                    features[col] = float(row[col])
            
            samples.append({
                "residue_index": int(row["residue_index"]),
                "sequence_only": True,  # FASTA analysis is sequence-only
                "features": features
            })
        
        # Call the internal prediction endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/ewcl/predict/ewclv1m/samples",
                    json={"samples": samples},
                    timeout=30.0
                )
                response.raise_for_status()
                predictions = response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

        # Combine predictions with original data
        pred_dict = {p["residue_index"]: p["prob"] for p in predictions["results"]}
        
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
            "model": "ewclv1m",
            "length": len(seq),
            "residues": residues,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EWCLv1m FASTA analysis failed: {e}")


