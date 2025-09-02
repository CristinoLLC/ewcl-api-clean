from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, Dict, Any
import sys, os, io, json, logging

# ── bootstrap sys.path ──────────────────────────────────────────────────────
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# httpx ASGI bridge for clean internal calls
import httpx
from httpx import ASGITransport

# parsing
from Bio import SeqIO

log = logging.getLogger("ewcl")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

app = FastAPI(
    title="EWCL API",
    version="2025.09",
    description="EWCL disorder & ClinVar – one endpoint per model"
)

# ── CORS ────────────────────────────────────────────────────────────────────
origins = [
    "http://localhost:3000",
    "https://ewclx.com",
    "https://www.ewclx.com",
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── payload limit ───────────────────────────────────────────────────────────
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "100000000"))  # 100 MB
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
class BodyLimit(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse({"error": "payload too large"}, status_code=413)
        return await call_next(request)
app.add_middleware(BodyLimit)

# ── env model paths (visibility + quick sanity in /models) ──────────────────
MODEL_ENVS = {
    "ewclv1":   os.environ.get("EWCLV1_MODEL_PATH"),
    "ewclv1-m": os.environ.get("EWCLV1_M_MODEL_PATH"),  # Fixed: underscore instead of hyphen
    "ewclv1-p3": os.environ.get("EWCLV1_P3_MODEL_PATH"),
    "ewclv1-c": os.environ.get("EWCLV1_C_MODEL_PATH"),  # Fixed: underscore instead of hyphen
}
for k,v in MODEL_ENVS.items():
    if v:
        log.info(f"[init] {k} model path = {v} (exists={Path(v).exists()})")

# ── optional raw routers (default OFF to keep public surface small) ─────────
ENABLE_RAW_ROUTERS = os.environ.get("ENABLE_RAW_ROUTERS", "0") in ("1","true","True")

try:
    from app.routes.analyze import router as analyze_router  # physics/proxy (optional)
    app.include_router(analyze_router, prefix="/api", tags=["physics"])
except Exception as e:
    log.warning(f"[warn] physics routes not mounted: {e}")

if ENABLE_RAW_ROUTERS:
    try:
        from backend.api.router import router as ewcl_router
        app.include_router(ewcl_router, prefix="/ewcl", tags=["ewcl-raw"])
        log.info("[init] raw EWCL routes enabled")
    except Exception as e:
        log.warning(f"[warn] EWCL raw routes not mounted: {e}")
    try:
        from backend.api.routers.clinvar_v73 import router as clinvar_router
        app.include_router(clinvar_router, prefix="/clinvar", tags=["clinvar-raw"])
        log.info("[init] raw ClinVar routes enabled")
    except Exception as e:
        log.warning(f"[warn] ClinVar raw routes not mounted: {e}")

try:
    from backend.api.routers.ewclv1_M import router as ewclv1_M_router
    app.include_router(ewclv1_M_router)
    log.info("[init] ewclv1-M router enabled")
except Exception as e:
    log.warning(f"[warn] ewclv1-M router not mounted: {e}")

try:
    from backend.api.routers.ewclv1 import router as ewclv1_router
    app.include_router(ewclv1_router)
    log.info("[init] ewclv1 router enabled")
except Exception as e:
    log.warning(f"[warn] ewclv1 router not mounted: {e}")

try:
    from backend.api.routers.ewclv1p3 import router as ewclv1p3_router
    app.include_router(ewclv1p3_router)
    log.info("[init] ewclv1p3 router enabled")
except Exception as e:
    log.warning(f"[warn] ewclv1p3 router not mounted: {e}")

# ClinVar variants router (always enabled - this is the main ClinVar endpoint)
try:
    from backend.api.routers.clinvar_variants import router as clinvar_variants_router
    app.include_router(clinvar_variants_router)
    log.info("[init] ClinVar variants router enabled")
except Exception as e:
    log.warning(f"[warn] ClinVar variants router not mounted: {e}")

# ── health + models status ──────────────────────────────────────────────────
@app.get("/")
def root(): return {"ok": True, "msg": "EWCL API alive"}

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/models")
async def models():
    """Reports env model paths + loaded models from internal /ewcl/health (if present)."""
    info = {
        "env_paths": {k: {"path": v, "exists": bool(v and Path(v).exists())} for k,v in MODEL_ENVS.items()},
        "loaded_models": [],
        "raw_router_enabled": ENABLE_RAW_ROUTERS,
        "clinvar_models": {}
    }
    # try check raw ewcl health if router mounted
    try:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://internal", timeout=10) as c:
            loaded = []
            for model_name in MODEL_ENVS.keys():
                try:
                    # probe each model's health check if possible
                    if model_name == "ewclv1-M":
                        r = await c.get(f"/ewcl/analyze-fasta/{model_name}/health")
                    elif model_name == "ewclv1":
                        r = await c.get(f"/ewcl/analyze-fasta/{model_name}/health")
                    elif model_name == "ewclv1p3":
                        r = await c.get(f"/ewcl/analyze-pdb/{model_name}/health")
                    
                    if r.status_code == 200 and r.json().get("loaded"):
                        loaded.append(model_name)
                except Exception:
                    pass # ignore errors if endpoint doesnt exist
            info["loaded_models"] = loaded
            
            # Check individual ClinVar model health
            try:
                r = await c.get("/clinvar/health")
                if r.status_code == 200:
                    info["clinvar_models"]["ewclv1c"] = r.json()
            except:
                pass
            
            try:
                r = await c.get("/clinvar-dash/health")
                if r.status_code == 200:
                    info["clinvar_models"]["ewclv1c_dash"] = r.json()
            except:
                pass
                
    except Exception as e:
        info["loaded_models_error"] = str(e)
    return info

# ── internal HTTP helper ───────────────────────────────────────────────────
async def _post_json(path: str, payload: Dict[str, Any], timeout_s: int = 300):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://internal", timeout=timeout_s) as client:
        resp = await client.post(path, json=payload)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.json()

async def _post_multipart(path: str, field_name: str, filename: str, content: bytes, timeout_s: int = 300):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://internal", timeout=timeout_s) as client:
        files = {field_name: (filename, content, "application/octet-stream")}
        resp = await client.post(path, files=files)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.json()

# ── helpers ─────────────────────────────────────────────────────────────────
def _read_fasta(b: bytes) -> Dict[str,str]:
    try:
        rec = next(SeqIO.parse(io.StringIO(b.decode("utf-8", errors="ignore")), "fasta"))
        return {"id": rec.id, "seq": str(rec.seq)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid FASTA: {e}")

def _normalize_residues(obj: Dict[str,Any], fallback_id: str, model_name: str):
    # ensure uniform {id, model, length, residues:[...]}
    rid = obj.get("id") or obj.get("protein_id") or fallback_id
    residues = obj.get("residues") or obj.get("predictions") or []
    length = obj.get("length") or (residues[-1]["residue_index"] if residues else None)
    return {"id": rid, "model": model_name, "length": length, "residues": residues}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT","8080")))
