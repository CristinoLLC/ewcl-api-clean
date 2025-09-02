from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys, os
from pathlib import Path
from Bio import SeqIO
import io, json

# ──────────────────────────────
# Ensure backend dir importable
# ──────────────────────────────
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

app = FastAPI(
    title="EWCL API",
    version="2025.09",
    description="Entropy Weighted Collapse Likelihood – disorder & physics predictors"
)

# ──────────────────────────────
# CORS (allow all, incl. Colab)
# ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # wildcard covers localhost + ewclx
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────
# Health endpoints
# ──────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "msg": "EWCL API alive"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/ewcl/health")
def ewcl_health():
    return {
        "status": "ok",
        "loaded_models": ["ewclv1", "ewclv1m", "ewclv1p3", "ewclv1c"],
        "bundle_dir": "/app/backend_bundle"
    }

@app.get("/test-cors")
def test_cors(request: Request):
    origin = request.headers.get("origin")
    return {
        "ok": True,
        "msg": "CORS test endpoint",
        "origin": origin,
        "headers": dict(request.headers)
    }

# ──────────────────────────────
# FASTA Analysis Endpoints
# ──────────────────────────────
@app.post("/ewcl/analyze-fasta/{model}")
async def analyze_fasta(model: str, file: UploadFile = File(...)):
    """Analyze FASTA file using specified EWCL model."""
    try:
        # Read FASTA content
        contents = await file.read()
        record = next(SeqIO.parse(io.StringIO(contents.decode()), "fasta"))
        seq_id = record.id
        sequence = str(record.seq)

        # Build payload for internal EWCL model
        payload = {
            "samples": [
                {
                    "uniprot": seq_id,
                    "residue_index": i + 1,
                    "sequence_only": True,
                    "features": {}  # backend sequencer should fill features
                }
                for i in range(len(sequence))
            ]
        }

        # Forward internally to the existing predict route
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.post(f"/ewcl/predict/{model}", json=payload)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FASTA analysis failed: {e}")

@app.post("/ewcl/analyze-pdb/{model}")
async def analyze_pdb(model: str, file: UploadFile = File(...)):
    """Analyze PDB file using specified EWCL model."""
    try:
        # Read PDB content
        contents = await file.read()
        
        # For now, return a placeholder response
        # TODO: Implement PDB analysis logic
        return {
            "filename": file.filename,
            "model": model,
            "analysis_type": "pdb",
            "message": "PDB analysis not yet implemented",
            "file_size": len(contents)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDB analysis failed: {e}")

# ──────────────────────────────
# Mount routers if present
# ──────────────────────────────
try:
    from app.routes.analyze import router as analyze_router
    app.include_router(analyze_router, prefix="/api", tags=["physics"])
except Exception as e:
    print(f"[warn] physics routes not mounted: {e}")

try:
    from backend.api.router import router as ewcl_router
    app.include_router(ewcl_router, prefix="/ewcl", tags=["ewcl"])
except Exception as e:
    print(f"[warn] EWCL routes not mounted: {e}")

try:
    from backend.api.routers.clinvar_v73 import router as clinvar_router
    app.include_router(clinvar_router, prefix="/clinvar", tags=["clinvar"])
except Exception as e:
    print(f"[warn] ClinVar routes not mounted: {e}")

try:
    from backend.api.routers.clinvar_analyze import router as clinvar_analyze_router
    app.include_router(clinvar_analyze_router, prefix="/clinvar", tags=["clinvar-analyze"])
except Exception as e:
    print(f"[warn] ClinVar analyze routes not mounted: {e}")

try:
    from backend.api.routers.ewclv1 import router as ewclv1_router
    app.include_router(ewclv1_router, prefix="/ewcl", tags=["ewclv1-fasta"])
except Exception as e:
    print(f"[warn] EWCLv1 FASTA routes not mounted: {e}")

try:
    from backend.api.routers.ewclv1m import router as ewclv1m_router
    app.include_router(ewclv1m_router, prefix="/ewcl", tags=["ewclv1m-fasta"])
except Exception as e:
    print(f"[warn] EWCLv1-M FASTA routes not mounted: {e}")

try:
    from backend.api.routers.ewclv1p3 import router as ewclv1p3_router
    app.include_router(ewclv1p3_router, prefix="/ewcl", tags=["ewclv1p3-pdb"])
except Exception as e:
    print(f"[warn] EWCLv1-P3 PDB routes not mounted: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
