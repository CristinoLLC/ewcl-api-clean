from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, Dict, Any
import sys, os, io, json, logging, time, requests

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

GIT_SHA = os.environ.get("GIT_SHA", "").strip() or None

# Model download configuration for Railway
MODELS_TO_FETCH = [
    ("EWCLV1_MODEL_URL", "/app/models/disorder/ewclv1.pkl"),
    ("EWCLV1_M_MODEL_URL", "/app/models/disorder/ewclv1-M.pkl"),
    ("EWCLV1_P3_MODEL_URL", "/app/models/pdb/ewclv1p3.pkl"),
    ("EWCLV1_C_MODEL_URL", "/app/models/clinvar/ewclv1-c.pkl"),
    # Removed EWCLV1_C_FEATS_URL - not needed since router uses hardcoded features
]

def _download_if_missing(url: str, dst: str):
    """Download model file if it doesn't exist."""
    p = Path(dst)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        log.info(f"[models] {dst} already exists, skipping download")
        return
    
    log.info(f"[models] Downloading {url} -> {dst}")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(p, "wb") as f:
            for chunk in r.iter_content(1 << 20):  # 1MB chunks
                if chunk:
                    f.write(chunk)
        log.info(f"[models] ✅ Downloaded {dst}")
    except Exception as e:
        log.error(f"[models] ❌ Failed to download {dst}: {e}")
        raise

app = FastAPI(
    title="EWCL API",
    version="2025.09",
    description="EWCL disorder & ClinVar – one endpoint per model"
)

@app.on_event("startup")
async def startup_pull_models():
    """Download models from URLs if they don't exist (Railway deployment)."""
    log.info("🚀 Checking for model downloads...")
    for env_var, dst_path in MODELS_TO_FETCH:
        url = os.getenv(env_var)
        if url:
            try:
                _download_if_missing(url, dst_path)
            except Exception as e:
                log.warning(f"[models] Failed to fetch {env_var}: {e}")
    log.info("✅ Model download check complete")

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

# ── env model paths (snake_case keys) ───────────────────────────────────────
MODEL_ENVS = {
    "ewclv1":    os.environ.get("EWCLV1_MODEL_PATH"),
    "ewclv1_m":  os.environ.get("EWCLV1_M_MODEL_PATH"),
    "ewclv1_p3": os.environ.get("EWCLV1_P3_MODEL_PATH"),
    "ewclv1_c":  os.environ.get("EWCLV1_C_MODEL_PATH"),
}
# keep features path separately (not probed as a model)
EWCLV1_C_FEATURES_PATH = os.environ.get("EWCLV1_C_FEATURES_PATH")

for k, v in MODEL_ENVS.items():
    if v:
        log.info(f"[init] {k} model path = {v} (exists={Path(v).exists()})")
if EWCLV1_C_FEATURES_PATH:
    log.info(f"[init] ewclv1_c_features path = {EWCLV1_C_FEATURES_PATH} (exists={Path(EWCLV1_C_FEATURES_PATH).exists()})")

# Public route slugs + health endpoints (internal probes)
PROBE_MAP = {
    "ewclv1":    "/ewcl/analyze-fasta/ewclv1/health",
    "ewclv1_m":  "/ewcl/analyze-fasta/ewclv1-m/health",
    "ewclv1_p3": "/ewcl/analyze-pdb/ewclv1-p3/health",
    "ewclv1_c":  "/clinvar/health",
}

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

# Re-enable ewclv1_C router now that features issue is fixed
try:
    from backend.api.routers.ewclv1_C import router as ewclv1_C_router
    app.include_router(ewclv1_C_router)
    log.info("[init] ewclv1-C router enabled (prefix /clinvar/ewclv1-C)")
except Exception as e:
    log.warning(f"[warn] ewclv1-C router not mounted: {e}")

# ClinVar variants router (always enabled - this is the main ClinVar endpoint)
try:
    from backend.api.routers.clinvar_variants import router as clinvar_variants_router
    app.include_router(clinvar_variants_router)
    log.info("[init] ClinVar variants router enabled")
except Exception as e:
    log.warning(f"[warn] ClinVar variants router not mounted: {e}")

# ── health + models status ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "msg": "EWCL API alive", "version": "2025.09", "git": GIT_SHA}

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/readyz")
def readyz():
    missing = [k for k, v in MODEL_ENVS.items() if v and not Path(v).exists()]
    return {
        "ok": len(missing) == 0,
        "missing_models": missing,
        "configured": [k for k, v in MODEL_ENVS.items() if v],
        "version": "2025.09",
        "git": GIT_SHA
    }

# cache for /models
_HEALTH_CACHE = {"ts": 0.0, "data": None}
_HEALTH_TTL_SECONDS = 30.0

@app.get("/models")
async def models():
    """Report env paths and live router health (cached)."""
    now = time.monotonic()
    if _HEALTH_CACHE["data"] is not None and (now - _HEALTH_CACHE["ts"] < _HEALTH_TTL_SECONDS):
        cached = _HEALTH_CACHE["data"].copy()
        # refresh exists flags quickly
        cached["env_paths"] = {k: {"path": v, "exists": bool(v and Path(v).exists())} for k, v in MODEL_ENVS.items()}
        return cached

    info = {
        "env_paths": {k: {"path": v, "exists": bool(v and Path(v).exists())} for k, v in MODEL_ENVS.items()},
        "loaded_models": [],
        "raw_router_enabled": ENABLE_RAW_ROUTERS,
        "clinvar_models": {}
    }
    if EWCLV1_C_FEATURES_PATH:
        info["env_paths"]["ewclv1_c_features"] = {"path": EWCLV1_C_FEATURES_PATH, "exists": Path(EWCLV1_C_FEATURES_PATH).exists()}

    try:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://internal", timeout=10) as c:
            loaded = []
            for key, health_path in PROBE_MAP.items():
                try:
                    r = await c.get(health_path)
                    if r.status_code == 200:
                        j = r.json()
                        if j.get("ok", True) and j.get("loaded", False):
                            loaded.append(key)
                        if key == "ewclv1_c":
                            info["clinvar_models"]["ewclv1c"] = j
                except Exception:
                    pass
            info["loaded_models"] = loaded
    except Exception as e:
        info["loaded_models_error"] = str(e)

    _HEALTH_CACHE["ts"] = now
    _HEALTH_CACHE["data"] = info.copy()
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
    import uvicorn, os
    # Read PORT from environment, fallback to 8080
    port = int(os.getenv("PORT", "8080"))
    log.info(f"[startup] Starting server on 0.0.0.0:{port} (git={GIT_SHA})")
    uvicorn.run(app, host="0.0.0.0", port=port)
