from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import os

app = FastAPI()

# ✅ Allow CORS from local and prod domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://ewclx.com",
        "https://www.ewclx.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check route for Render and debugging
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ✅ Sample placeholder PDB output route
@app.get("/sample-pdb")
def get_sample_pdb(id: str):
    return {
        "protein_id": id,
        "ewcl_scores": [0.2, 0.4, 0.6, 0.85]
    }

# ✅ Main EWCL analysis route — auto-detects JSON or file
@app.post("/analyze")
async def analyze(request: Request, file: Optional[UploadFile] = File(None)):
    content_type = request.headers.get("content-type", "")

    # === CASE 1: JSON payload (collapse scores)
    if "application/json" in content_type:
        try:
            body = await request.json()
            scores = body.get("scores")
            if not scores:
                return JSONResponse(status_code=422, content={"error": "Missing 'scores' in JSON"})

            collapse_likelihood = sum(scores.values()) / len(scores)
            max_res = max(scores, key=scores.get)

            return {
                "source": "json",
                "collapse_likelihood": round(collapse_likelihood, 4),
                "most_unstable_residue": int(max_res),
                "residue_scores": scores
            }

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # === CASE 2: File upload (.pdb or .cif)
    elif file:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdb", ".cif"]:
            return JSONResponse(status_code=400, content={"error": "Only .pdb or .cif files are supported"})

        contents = await file.read()
        # Placeholder dummy scoring logic — replace with real EWCL pipeline
        scores = {i: round(1.0 / (i + 1), 4) for i in range(1, 51)}

        return {
            "source": ext,
            "collapse_likelihood": round(sum(scores.values()) / len(scores), 4),
            "residue_scores": scores
        }

    # === CASE 3: No input detected
    return JSONResponse(status_code=400, content={"error": "No valid input provided"})