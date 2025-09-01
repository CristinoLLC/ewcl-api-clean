from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from Bio import SeqIO
import io
import numpy as np
from backend.api.utils.sequencer import parser_ewclv1

router = APIRouter(prefix="/ewcl", tags=["EWCLv1-FASTA"])

@router.post("/analyze-fasta/ewclv1")
async def analyze_fasta(file: UploadFile = File(...)):
    try:
        fasta_text = (await file.read()).decode("utf-8")
        record = next(SeqIO.parse(io.StringIO(fasta_text), "fasta"))
        seq = str(record.seq)

        df = parser_ewclv1(seq)
        if df.empty:
            return JSONResponse(content={"id": record.id if record.id else None, "model": "ewclv1", "length": 0, "residues": []})

        # Placeholder CL/confidence; wire real model later
        df["CL"] = np.clip(df["curvature"] * 0.5 + df["hydro_entropy"] * 0.3 + df["charge_entropy"] * 0.2, 0, 1)
        df["confidence"] = 1.0 - np.abs(0.5 - df["CL"])

        residues = []
        for _, row in df.iterrows():
            residues.append({
                "residue_index": int(row["residue_index"]),
                "aa": row["aa"],
                "CL": round(float(row["CL"]), 4),
                "confidence": round(float(row["confidence"]), 4),
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


