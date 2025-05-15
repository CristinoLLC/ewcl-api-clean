from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import os

# Optional: from ewcl_core import compute_ewcl_scores

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:7173",
        "https://ewclx.com",
        "https://www.ewclx.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdb", ".cif"]:
        return JSONResponse(status_code=400, content={"error": "Only .pdb or .cif files are supported"})

    try:
        contents = await file.read()
        pdb_text = contents.decode("utf-8")

        # Replace with: scores = compute_ewcl_scores(pdb_text)
        scores = {i: round(1.0 / (i + 1) + 0.005 * (i % 7), 4) for i in range(1, 10001)}  # Simulate 10k

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_5000 = dict(sorted_scores[:5000])

        collapse_likelihood = round(sum(scores.values()) / len(scores), 4)
        most_unstable_residue = max(scores, key=scores.get)

        return {
            "source": ext,
            "collapse_likelihood": collapse_likelihood,
            "most_unstable_residue": most_unstable_residue,
            "filtered_scores": top_5000,
            "residue_scores": scores
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})