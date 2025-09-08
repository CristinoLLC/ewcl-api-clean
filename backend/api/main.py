import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router as ewcl_router
from backend.models.model_manager import load_all_models, get_loaded_models  # Use new model manager

# Import individual model routers with REAL features (no generic Column_X)
from backend.api.routers.ewclv1p3_fresh import router as ewclv1p3_router  # Fresh implementation with all 302 features
from backend.api.routers.ewclv1_M import router as ewclv1m_router
from backend.api.routers.ewclv1 import router as ewclv1_router
from backend.api.routers.ewclv1_C import router as ewclv1c_router, EWCLV1_C_FEATURES

try:
    from backend.api.routers.clinvar_v73 import router as clinvar_router
except Exception as _e:
    clinvar_router = None

app = FastAPI(title="EWCL Inference API", version="1.0.0")

# Initialize model manager at startup - this loads all models into memory
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting EWCL API - initializing model manager...")
    load_all_models()  # Simple function call, no singleton
    loaded_models = get_loaded_models()
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

# Include all routers
app.include_router(ewcl_router)  # Legacy bundle-based router
app.include_router(ewclv1p3_router)  # PDB model with 302 REAL features
app.include_router(ewclv1m_router)   # Disorder model with 255 REAL features
app.include_router(ewclv1_router)    # FASTA model with 249 REAL features
app.include_router(ewclv1c_router)   # ClinVar model with 47 REAL features

if clinvar_router is not None:
    app.include_router(clinvar_router)

@app.get("/")
def root():
    routes = [
        "/ewcl/health", "/ewcl/predict/ewclv1m", "/ewcl/analyze-fasta/ewclv1",
        "/ewcl/analyze-pdb/ewclv1-p3", "/ewcl/analyze-fasta/ewclv1-m", "/clinvar/ewclv1-C/analyze-variants"
    ]
    if clinvar_router is not None:
        routes += ["/clinvar/v7_3/health", "/clinvar/v7_3/predict", "/clinvar/v7_3/predict_gated"]
    return {"status": "ok", "message": "EWCL Inference API - Dr. Uversky Benchmark Ready", "routes": routes}

# --- Ops endpoints ---
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    try:
        # Check if model manager is initialized and models are loaded
        loaded_models = get_loaded_models()
        return {"ok": True, "loaded_models": loaded_models, "count": len(loaded_models)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/models")
def models_status():
    """Enhanced models endpoint showing detailed loading status with REAL features."""
    try:
        from backend.models.model_manager import is_loaded
        
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
            "loaded_models": get_loaded_models(),
            "raw_router_enabled": os.environ.get("ENABLE_RAW_ROUTERS", "0") == "1",
            "real_features": True,  # NO generic Column_X features
            "benchmark_ready": True,  # Ready for Dr. Uversky benchmark
            "individual_models": {
                "ewclv1": {
                    "ok": is_loaded("ewclv1"),
                    "features": 249,
                    "type": "FASTA",
                    "endpoint": "/ewcl/analyze-fasta/ewclv1"
                },
                "ewclv1-p3": {
                    "ok": is_loaded("ewclv1-p3"),
                    "features": 302,
                    "type": "PDB",
                    "endpoint": "/ewcl/analyze-pdb/ewclv1-p3"
                },
                "ewclv1-m": {
                    "ok": is_loaded("ewclv1-m"),
                    "features": 255,
                    "type": "Disorder",
                    "endpoint": "/ewcl/analyze-fasta/ewclv1-m"
                },
                "ewclv1-c": {
                    "ok": is_loaded("ewclv1-c"),
                    "features": 47,
                    "type": "ClinVar",
                    "endpoint": "/clinvar/ewclv1-C/analyze-variants"
                }
            },
            "clinvar_models": {
                "ewclv1c": {
                    "ok": is_loaded("ewclv1-c"),
                    "model": "ewclv1-c",
                    "loaded": is_loaded("ewclv1-c"),
                    "features": 47,
                    "ready": is_loaded("ewclv1-c")
                }
            }
        }
    except Exception as e:
        return {"error": str(e), "loaded_models": []}

@app.get("/features")
def get_all_features():
    """Get actual feature names for all EWCL models."""
    try:
        # Import feature lists from fresh router implementation
        from backend.api.routers.ewclv1p3_fresh import FEATURE_NAMES as EWCLV1_P3_FEATURES
        
        # Use hardcoded feature counts since parsers are being removed
        ewclv1_features = ["hydropathy", "charge", "flexibility", "secondary_structure"] # 249 features total
        ewclv1m_features = ["sequence_properties", "composition", "flexibility", "pssm"] # 255 features total

        return {
            "models": {
                "ewclv1": {
                    "endpoint": "/ewcl/analyze-fasta/ewclv1",
                    "description": "Protein disorder prediction from FASTA sequences",
                    "feature_count": 249,
                    "features": ["Feature extraction handled by router"],
                    "note": "Parsers removed - features computed directly in router"
                },
                "ewclv1-m": {
                    "endpoint": "/ewcl/analyze-fasta/ewclv1-m", 
                    "description": "Enhanced disorder prediction with PSSM features",
                    "feature_count": 255,
                    "features": ["Feature extraction handled by router"],
                    "note": "Parsers removed - features computed directly in router"
                },
                "ewclv1-p3": {
                    "endpoint": "/ewcl/analyze-pdb/ewclv1-p3",
                    "description": "PDB structure analysis with FRESH 302 features implementation", 
                    "feature_count": len(EWCLV1_P3_FEATURES),
                    "features": EWCLV1_P3_FEATURES[:10] + ["..."],
                    "all_features": EWCLV1_P3_FEATURES,
                    "implementation": "fresh_complete_feature_engineering"
                },
                "ewclv1-c": {
                    "endpoint": "/clinvar/ewclv1-C/analyze-variants",
                    "description": "ClinVar variant pathogenicity prediction",
                    "feature_count": len(EWCLV1_C_FEATURES), 
                    "features": EWCLV1_C_FEATURES,
                    "all_features": EWCLV1_C_FEATURES
                }
            },
            "summary": {
                "total_models": 4,
                "ewclv1_features": 249,
                "ewclv1m_features": 255, 
                "ewclv1p3_features": len(EWCLV1_P3_FEATURES),
                "ewclv1c_features": len(EWCLV1_C_FEATURES),
                "note": "EWCLv1-P3 now uses FRESH implementation with complete feature engineering",
                "fresh_p3_parser": True
            }
        }
    except Exception as e:
        return {"error": f"Failed to load feature information: {str(e)}"}


