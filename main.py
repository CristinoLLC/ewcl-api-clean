from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import shutil
import uuid
from ewcl_real_model import compute_ewcl_df

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_pdb(file: UploadFile = File(...)):
    # Save uploaded file
    temp_filename = f"/tmp/{uuid.uuid4().hex}.pdb"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model
    df = compute_ewcl_df(temp_filename)
    df = df.round(3)

    return {
        "results": df[[
            "residue_id", "chain", "aa",
            "bfactor", "hydro_entropy", "charge_entropy",
            "bfactor_curv", "hydro_curv", "cl", "note"
        ]].to_dict(orient="records")
    }
