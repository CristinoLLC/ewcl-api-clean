import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router as ewcl_router
try:
    from backend.api.routers.clinvar_v73 import router as clinvar_router
except Exception as _e:
    clinvar_router = None


app = FastAPI(title="EWCL Inference API", version="1.0.0")

# CORS: ALLOWED_ORIGINS="*" or comma-separated
origins_env = os.environ.get("ALLOWED_ORIGINS")
if origins_env:
    if origins_env.strip() == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ewcl_router)
if clinvar_router is not None:
    app.include_router(clinvar_router)


@app.get("/")
def root():
    routes = ["/ewcl/health", "/ewcl/predict/ewclv1m", "/ewcl/predict/ewclv1"]
    if clinvar_router is not None:
        routes += ["/clinvar/v7_3/health", "/clinvar/v7_3/predict", "/clinvar/v7_3/predict_gated"]
    return {"status": "ok", "message": "EWCL Inference API", "routes": routes}

# --- Ops endpoints ---
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    try:
        # If import succeeded, models are loaded at module import time
        return {"ok": True}
    except Exception:
        return {"ok": False}


