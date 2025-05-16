from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Tuple
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from io import StringIO
import os
import re

from ewcl_core import (
    compute_ewcl_scores_from_pdb,
    compute_ewcl_scores_from_alphafold_json,
    compute_ewcl_scores_from_sequence
)

app = FastAPI(
    title="EWCL API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Define a custom function to allow regex origins
def regex_origin(origin: str) -> bool:
    allowed_patterns = [
        r"https://www\.ewclx\.com$",                  # Custom domain
        r"https://.*\.vercel\.app$",                  # All Vercel deployments
        r"https://.*\.vercel-insights\.com$",         # Vercel Insights
        r"https://.*\.vercel-scripts\.com$",          # Vercel Scripts
        r"https://.*\.v0\.dev$",                      # V0 frontends
        r"http://localhost:3000$",                    # Local dev
        r"http://localhost:7173$",                    # Existing local dev
        r"https://ewclx\.com$",                       # Existing domain
        r"https://next-webapp-with-mol-pvqM9XLgrJc\.v0\.dev$",  # Existing specific domain
    ]
    return any(re.match(pattern, origin) for pattern in allowed_patterns)

# Update your middleware to use the custom regex function
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",  # Allow all origins initially, we'll filter manually
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Custom middleware to enforce regex-based origin checking
@app.middleware("http")
async def custom_cors_middleware(request, call_next):
    origin = request.headers.get("origin")
    response = await call_next(request)
    if origin and regex_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response

@app.get("/")
def root():
    return {"status": "online", "message": "EWCL API running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".pdb") or filename.endswith(".cif"):
            pdb_text = contents.decode("utf-8")
            scores = compute_ewcl_scores_from_pdb(pdb_text)
            source_type = "pdb"

        elif filename.endswith(".json"):
            scores = compute_ewcl_scores_from_alphafold_json(contents)
            source_type = "alphafold"

        elif filename.endswith(".fasta") or filename.endswith(".fa"):
            fasta = contents.decode("utf-8")
            scores = compute_ewcl_scores_from_sequence(fasta)
            source_type = "sequence"

        else:
            return JSONResponse(
                status_code=400, 
                content={"error": "Supported files: .pdb, .cif, .json (AlphaFold), .fa/.fasta"}
            )

        # Sort and extract top 5k unstable residues
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_5000 = dict(sorted_scores[:min(5000, len(sorted_scores))])
        
        if scores:
            collapse_likelihood = round(sum(scores.values()) / len(scores), 4)
            most_unstable_residue = max(scores, key=scores.get)
        else:
            collapse_likelihood = 0
            most_unstable_residue = None

        return {
            "source": source_type,
            "collapse_likelihood": collapse_likelihood,
            "most_unstable_residue": most_unstable_residue,
            "filtered_scores": top_5000,
            "residue_scores": scores,
            "entropy_sources_used": {
                "b_factor": source_type == "pdb",
                "disorder": source_type == "sequence",
                "plddt": source_type == "alphafold"
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def extract_sequence_from_pdb(pdb_text: str) -> Tuple[str, Dict[int, str]]:
    """Extract amino acid sequence from PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_text))
    
    sequence = ""
    residue_mapping = {}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # Standard amino acid
                    try:
                        aa = seq1(residue.get_resname())
                        sequence += aa
                        residue_mapping[residue.get_id()[1]] = aa
                    except:
                        continue
            break  # Only process first chain
        break  # Only process first model
            
    return sequence, residue_mapping