from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

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
# CORS (explicit + wildcard)
# ──────────────────────────────
origins = [
    "http://localhost:3000",
    "https://ewclx.com",
    "https://www.ewclx.com",
    "*"  # allow Colab, cURL, etc.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
