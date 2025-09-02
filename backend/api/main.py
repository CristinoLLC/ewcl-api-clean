import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router as ewcl_router
from backend.models.singleton import get_model_manager  # Import singleton manager

try:
    from backend.api.routers.clinvar_v73 import router as clinvar_router
except Exception as _e:
    clinvar_router = None

app = FastAPI(title="EWCL Inference API", version="1.0.0")

# Initialize model manager at startup - this loads all models into memory
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting EWCL API - initializing model manager...")
    model_manager = get_model_manager()
    loaded_models = model_manager.get_loaded_models()
    print(f"✅ Model manager initialized with models: {loaded_models}")

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
        # Check if model manager is initialized and models are loaded
        model_manager = get_model_manager()
        loaded_models = model_manager.get_loaded_models()
        return {"ok": True, "loaded_models": loaded_models, "count": len(loaded_models)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/models")
def models_status():
    """Enhanced models endpoint showing detailed loading status."""
    try:
        model_manager = get_model_manager()
        
        # Check environment paths
        env_paths = {}
        for name, env_var in [
            ("ewclv1", "EWCLV1_MODEL_PATH"),
            ("ewclv1-m", "EWCLV1_M_MODEL_PATH"), 
            ("ewclv1-p3", "EWCLV1_P3_MODEL_PATH"),
            ("ewclv1-c", "EWCLV1_C_MODEL_PATH")
        ]:
            path = os.environ.get(env_var)
            env_paths[name] = {
                "path": path,
                "exists": os.path.exists(path) if path else False
            }
        
        return {
            "env_paths": env_paths,
            "loaded_models": model_manager.get_loaded_models(),
            "raw_router_enabled": os.environ.get("ENABLE_RAW_ROUTERS", "0") == "1",
            "clinvar_models": {
                "ewclv1c": {
                    "ok": model_manager.is_loaded("ewclv1-c"),
                    "model": "ewclv1-c",
                    "loaded": model_manager.is_loaded("ewclv1-c"),
                    "features": len(model_manager.get_feature_order("ewclv1-c")) if model_manager.is_loaded("ewclv1-c") else 0,
                    "ready": model_manager.is_loaded("ewclv1-c")
                }
            }
        }
    except Exception as e:
        return {"error": str(e), "loaded_models": []}


