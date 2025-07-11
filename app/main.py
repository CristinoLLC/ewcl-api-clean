"""
EWCL Collapse-Likelihood API with four endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, joblib, pandas as pd, numpy as np
import warnings

# ───────────────────────────────────────────
# 1)  load physics extractor
# ───────────────────────────────────────────
from models.enhanced_ewcl_af import compute_curvature_features  # physics extractor

def run_physics(pdb_bytes: bytes) -> pd.DataFrame:
    """Run the physics-only EWCL extractor on raw PDB bytes."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp.write(pdb_bytes)
    tmp.close()

    rows = compute_curvature_features(tmp.name)  # returns list[dict]
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No CA atoms found or extractor failed")

    return df

# ───────────────────────────────────────────
# 2)  load ML models & scalers with error handling
# ───────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

def load_model_safely(model_path, model_name):
    """Safely load model with detailed error logging and fallbacks"""
    try:
        print(f"📁 Attempting to load {model_name} from: {model_path}")
        print(f"📊 File exists: {model_path.exists()}")
        if model_path.exists():
            print(f"📏 File size: {model_path.stat().st_size} bytes")
        
        # Suppress sklearn version warnings temporarily
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            model = joblib.load(model_path)
        
        print(f"✅ Successfully loaded {model_name}")
        return model
    except ModuleNotFoundError as e:
        print(f"❌ Module error loading {model_name}: {e}")
        print(f"💡 This is likely a scikit-learn/numpy version mismatch")
        return None
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}: {type(e).__name__}: {e}")
        print(f"📍 Full error details: {str(e)}")
        return None

# Debug: Print model directory info
print(f"🔍 Model directory: {MODEL_DIR}")
print(f"📂 Model directory exists: {MODEL_DIR.exists()}")
if MODEL_DIR.exists():
    print(f"📋 Files in models/: {list(MODEL_DIR.iterdir())}")

REGRESSOR = load_model_safely(MODEL_DIR / "ewcl_regressor_model.pkl", "regressor")
HIGH_MODEL = load_model_safely(MODEL_DIR / "ewcl_residue_local_high_model.pkl", "high_model") 
HIGH_SCALER = load_model_safely(MODEL_DIR / "ewcl_residue_local_high_scaler.pkl", "high_scaler")
HALLUC_MODEL = load_model_safely(MODEL_DIR / "hallucination_detector_model.pkl", "halluc_model")

# Print startup summary
print("\n" + "="*50)
print("🚀 EWCL API STARTUP SUMMARY")
print("="*50)
print(f"✅ Physics extractor: READY")
print(f"{'✅' if REGRESSOR else '❌'} Regressor model: {'LOADED' if REGRESSOR else 'FAILED (using physics fallback)'}")
print(f"{'✅' if HIGH_MODEL else '❌'} High refinement: {'LOADED' if HIGH_MODEL else 'FAILED (using physics fallback)'}")
print(f"{'✅' if HALLUC_MODEL else '❌'} Hallucination detector: {'LOADED' if HALLUC_MODEL else 'FAILED (using physics fallback)'}")
print("\n📡 Available endpoints:")
print("  • /api/analyze/raw - Physics-only EWCL")
print("  • /api/analyze/regressor - Physics + ML regressor (fallback if model unavailable)")
print("  • /api/analyze/refined - Physics + refined model (fallback if model unavailable)")
print("  • /api/analyze/hallucination - Physics + hallucination detection (fallback if model unavailable)")
print("  • /health - Health check and model status")
print("\n🚀 LOCAL DEVELOPMENT:")
print("  Run with: uvicorn app.main:app --reload --port 8000")
print("  Test at: http://localhost:8000/health")
print("="*50)

# ───────────────────────────────────────────
# Feature definitions with safe fallbacks
# ───────────────────────────────────────────
REG_FEATS = [
    "bfactor",
    "plddt", 
    "bfactor_norm",
    "hydro_entropy",
    "charge_entropy",
    "bfactor_curv",
    "bfactor_curv_entropy",
    "bfactor_curv_flips"
]

HIGH_FEATS = [
    "bfactor",
    "plddt",
    "bfactor_norm", 
    "hydro_entropy",
    "charge_entropy",
    "bfactor_curv",
    "bfactor_curv_entropy",
    "bfactor_curv_flips"
]

HAL_FEATS = [
    "cl_diff",
    "cl_diff_slope",
    "cl_diff_curv",
    "cl_diff_flips",
    "bfactor",
    "bfactor_norm",
    "hydro_entropy",
    "charge_entropy", 
    "bfactor_curv",
    "bfactor_curv_entropy",
    "bfactor_curv_flips"
]

def get_safe_features(df: pd.DataFrame, expected_features: list) -> list:
    """Get only features that exist in DataFrame and were used during training"""
    available_features = [f for f in expected_features if f in df.columns]
    if len(available_features) != len(expected_features):
        missing = set(expected_features) - set(available_features)
        print(f"⚠️ Missing features: {missing}")
        print(f"✅ Using available features: {available_features}")
    return available_features

def load_pickle(filepath):
    """Helper to load pickle files with error handling"""
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

# ───────────────────────────────────────────
# 3)  ML Prediction Helpers with robust feature filtering
# ───────────────────────────────────────────
def add_main_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add predictions from the main EWCL regressor model to the DataFrame.
    Automatically filters to only expected features and avoids unseen ones like plddt.
    """
    try:
        if REGRESSOR is None:
            raise HTTPException(status_code=503, detail="Regressor model not available")
        
        expected_features = [
            "bfactor", "plddt", "bfactor_norm", 
            "hydro_entropy", "charge_entropy", 
            "bfactor_curv", "bfactor_curv_entropy", 
            "bfactor_curv_flips"
        ]

        # Ensure plddt column exists for compatibility
        if "plddt" not in df.columns:
            df["plddt"] = df["bfactor"]

        # Only use features that exist in df
        available_features = [f for f in expected_features if f in df.columns]
        if not available_features:
            raise ValueError("No expected features found in input DataFrame.")
        
        print(f"📊 Using features for regressor: {available_features}")
        if len(available_features) != len(expected_features):
            missing = set(expected_features) - set(available_features)
            print(f"⚠️ Missing features: {missing}")

        X = df[available_features]
        df["cl_pred"] = REGRESSOR.predict(X)
        return df

    except Exception as e:
        raise RuntimeError(f"❌ add_main_prediction failed: {e}")

def add_refined_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add predictions from the high-confidence refiner model to the DataFrame.
    Uses scaler and filters to expected features only.
    """
    try:
        if HIGH_MODEL is None or HIGH_SCALER is None:
            raise HTTPException(status_code=503, detail="High refinement models not available")
        
        expected_features = [
            "bfactor", "plddt", "bfactor_norm",
            "hydro_entropy", "charge_entropy",
            "bfactor_curv", "bfactor_curv_entropy",
            "bfactor_curv_flips"
        ]

        # Ensure plddt column exists for compatibility
        if "plddt" not in df.columns:
            df["plddt"] = df["bfactor"]

        # Only use features that exist in df
        available_features = [f for f in expected_features if f in df.columns]
        if not available_features:
            raise ValueError("No expected features found in input DataFrame.")
        
        print(f"📊 Using features for refiner: {available_features}")
        if len(available_features) != len(expected_features):
            missing = set(expected_features) - set(available_features)
            print(f"⚠️ Missing features: {missing}")

        X = df[available_features]
        X_scaled = HIGH_SCALER.transform(X)
        df["cl_refined"] = HIGH_MODEL.predict(X_scaled)
        return df

    except Exception as e:
        raise RuntimeError(f"❌ add_refined_prediction failed: {e}")

def add_hallucination_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hallucination detection predictions to the DataFrame.
    Handles missing values and ensures robust prediction.
    """
    try:
        if HALLUC_MODEL is None:
            raise HTTPException(status_code=503, detail="Hallucination model not available")
        if "cl_pred" not in df.columns:
            df = add_main_prediction(df)
        
        # Calculate difference features
        df["cl_diff"] = (df["cl_pred"] - df["cl"]).abs()
        df["cl_diff_slope"] = np.gradient(df["cl_diff"])
        df["cl_diff_curv"]  = np.gradient(df["cl_diff_slope"])
        df["cl_diff_flips"] = (
            pd.Series(np.sign(df["cl_diff_slope"])).diff().abs().fillna(0)
        )
        
        expected_features = [
            "cl_diff", "cl_diff_slope", "cl_diff_curv", "cl_diff_flips",
            "bfactor", "bfactor_norm", "hydro_entropy", "charge_entropy",
            "bfactor_curv", "bfactor_curv_entropy", "bfactor_curv_flips"
        ]

        # Sanity check - add missing columns with defaults
        for col in expected_features:
            if col not in df.columns:
                print(f"⚠️ Missing feature {col}, filling with 0.0")
                df[col] = 0.0

        # Only use features that exist in df
        available_features = [f for f in expected_features if f in df.columns]
        if not available_features:
            raise ValueError("No expected features found in input DataFrame.")

        X = df[available_features].copy()
        
        # ✅ Handle NaNs before prediction
        if X.isnull().values.any():
            print("⚠️ NaNs detected in hallucination input. Filling with 0.")
            X = X.fillna(0)
        
        print(f"📊 Using features for hallucination: {available_features}")

        # Predict hallucination likelihood
        df["hallucination"] = HALLUC_MODEL.predict(X)
        df["halluc_score"] = HALLUC_MODEL.predict_proba(X)[:, 1]
        
        # Add logging for detected hallucinations
        hallucinated = df[df["hallucination"] == 1]
        print(f"✅ Hallucinated residues: {len(hallucinated)} / {len(df)}")
        
        return df

    except Exception as e:
        print(f"❌ Hallucination prediction failed: {e}")
        # Fallback: add empty hallucination columns
        df["hallucination"] = 0
        df["halluc_score"] = 0.0
        return df

# ───────────────────────────────────────────
# Individual Model Functions
# ───────────────────────────────────────────
async def run_physics_only(file: UploadFile) -> dict:
    """Run only the physics-based EWCL extractor"""
    df = run_physics(await file.read())
    
    # Always return physics results, with optional ML enhancement
    result_cols = ["chain", "position", "aa", "cl", "bfactor", "plddt"]
    
    # Only try ML if models are available
    if REGRESSOR is not None and HALLUC_MODEL is not None:
        try:
            df = add_main_prediction(df)
            df = add_hallucination_prediction(df)
            result_cols.extend(["cl_pred", "hallucination", "halluc_score"])
            print("🔬 Auto-added ML predictions to physics analysis")
        except Exception as e:
            print(f"⚠️ ML enhancement failed, returning physics-only: {e}")
    else:
        print("📊 Models not available, returning physics-only results")
    
    # Filter to only available columns
    available_cols = [col for col in result_cols if col in df.columns]
    return df[available_cols].to_dict("records")

async def run_regressor_model(file: UploadFile) -> dict:
    """Run physics + main regressor model (with fallback)"""
    df = run_physics(await file.read())
    
    if REGRESSOR is None:
        print("⚠️ Regressor not available, returning physics-only")
        return df[["chain", "position", "aa", "cl", "bfactor", "plddt"]].to_dict("records")
    
    try:
        df = add_main_prediction(df)
        result_cols = ["chain", "position", "aa", "cl", "cl_pred", "bfactor", "plddt"]
        
        # Auto-add hallucination if possible
        if HALLUC_MODEL is not None:
            try:
                df = add_hallucination_prediction(df)
                result_cols.extend(["hallucination", "halluc_score"])
            except Exception as e:
                print(f"⚠️ Hallucination detection failed: {e}")
        
        available_cols = [col for col in result_cols if col in df.columns]
        return df[available_cols].to_dict("records")
    
    except Exception as e:
        print(f"⚠️ Regressor failed, falling back to physics-only: {e}")
        return df[["chain", "position", "aa", "cl", "bfactor", "plddt"]].to_dict("records")

async def run_refined_model(file: UploadFile) -> dict:
    """Run physics + refined model (with fallback)"""
    df = run_physics(await file.read())
    
    if HIGH_MODEL is None or HIGH_SCALER is None:
        print("⚠️ Refined model not available, returning physics-only")
        return df[["chain", "position", "aa", "cl", "bfactor", "plddt"]].to_dict("records")
    
    try:
        df = add_refined_prediction(df)
        return df[["chain", "position", "aa", "cl", "cl_refined", "bfactor", "plddt"]].to_dict("records")
    except Exception as e:
        print(f"⚠️ Refined model failed, falling back to physics-only: {e}")
        return df[["chain", "position", "aa", "cl", "bfactor", "plddt"]].to_dict("records")

async def run_hallucination_model(file: UploadFile) -> dict:
    """Run hallucination detection (with fallback)"""
    df = run_physics(await file.read())
    
    if REGRESSOR is None or HALLUC_MODEL is None:
        print("⚠️ Hallucination models not available, returning physics-only")
        df["hallucination"] = 0
        df["halluc_score"] = 0.0
        return df[["chain", "position", "aa", "cl", "hallucination", "halluc_score", "bfactor", "plddt"]].to_dict("records")
    
    try:
        df = add_main_prediction(df)
        df = add_hallucination_prediction(df)
        return df[["chain", "position", "aa", "cl", "cl_pred", "hallucination", "halluc_score", "bfactor", "plddt"]].to_dict("records")
    except Exception as e:
        print(f"⚠️ Hallucination detection failed, adding empty values: {e}")
        df["cl_pred"] = df["cl"]  # Fallback
        df["hallucination"] = 0
        df["halluc_score"] = 0.0
        return df[["chain", "position", "aa", "cl", "cl_pred", "hallucination", "halluc_score", "bfactor", "plddt"]].to_dict("records")

# ───────────────────────────────────────────
# 4)  FastAPI app (renamed for uvicorn compatibility)
# ───────────────────────────────────────────
app = FastAPI(
    title="EWCL Collapse-Likelihood API",
    version="2025.0.1",
    description="Physics-based + ML refined EWCL with hallucination flags",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.ewclx.com",
        "https://ewclx.com", 
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

@api_router.post("/analyze/raw")
async def analyze_raw(file: UploadFile = File(...)):
    """Physics-only EWCL analysis"""
    try:
        result = await run_physics_only(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/analyze/regressor")
async def analyze_regressor(file: UploadFile = File(...)):
    """Physics + main regressor model"""
    try:
        result = await run_regressor_model(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/analyze/refined")
async def analyze_refined(file: UploadFile = File(...)):
    """Physics + refined high-confidence model"""
    try:
        result = await run_refined_model(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/analyze/hallucination")
async def analyze_hallucination(file: UploadFile = File(...)):
    """Physics + regressor + hallucination detection"""
    try:
        result = await run_hallucination_model(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/analyze-pdb")
async def analyze_pdb_legacy(pdb: UploadFile = File(...)):
    """Legacy: analyze-pdb endpoint"""
    return await analyze_raw(pdb)

# Include the router
app.include_router(api_router)

@app.get("/")
def health_check():
    return {
        "status": "EWCL API v2025.0.1", 
        "endpoints": {
            "api/analyze/raw": "Physics-only EWCL (no ML)",
            "api/analyze/regressor": "Physics + main regressor",
            "api/analyze/refined": "High-confidence refiner",
            "api/analyze/hallucination": "Hallucination detection",
            "api/analyze-pdb": "Legacy analyze-pdb endpoint"
        },
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "high_model": HIGH_MODEL is not None,
            "high_scaler": HIGH_SCALER is not None,
            "halluc_model": HALLUC_MODEL is not None
        }
    }

@app.get("/health")
def health():
    model_files = []
    if MODEL_DIR.exists():
        model_files = [f.name for f in MODEL_DIR.glob("*")]
    
    return {
        "status": "ok",
        "endpoints": {
            "api/analyze/raw": "Physics-only EWCL",
            "api/analyze/regressor": "Physics + main regressor", 
            "api/analyze/refined": "Physics + refined model",
            "api/analyze/hallucination": "Physics + hallucination detection"
        },
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "refiner": HIGH_MODEL is not None,
            "hallucination": HALLUC_MODEL is not None,
            "scaler": HIGH_SCALER is not None
        },
        "version": "2025.0.1",
        "python_version": "3.13.4",
        "scikit_learn_version": "1.6.1 (pinned)",
        "model_dir_exists": MODEL_DIR.exists(),
        "model_files": model_files,
        "model_dir_path": str(MODEL_DIR)
    }

# ───────────────────────────────────────────
# Legacy endpoints (keep for compatibility)
# ───────────────────────────────────────────

@app.post("/raw-physics/")
async def raw_physics(pdb: UploadFile = File(...)):
    """Legacy: Physics-only EWCL"""
    return await analyze_raw(pdb)

@app.post("/analyze-ewcl/")
async def analyze_ewcl(pdb: UploadFile = File(...)):
    """Legacy: Physics + main regressor"""
    return await analyze_regressor(pdb)

@app.post("/refined-ewcl/")
async def refined_ewcl(pdb: UploadFile = File(...)):
    """Legacy: Physics + refined model"""
    return await analyze_refined(pdb)

@app.post("/detect-hallucination/")
async def detect_hallucination(pdb: UploadFile = File(...)):
    """Legacy: Hallucination detection"""
    return await analyze_hallucination(pdb)

@app.post("/analyze/full")
async def analyze_full(pdb: UploadFile = File(...)):
    """Legacy: All models at once"""
    try:
        df = run_physics(await pdb.read())
        df = add_main_prediction(df)
        df = add_refined_prediction(df)
        df = add_hallucination_prediction(df)
        
        output_cols = [
            "chain", "position", "aa", "cl", "cl_pred", 
            "cl_refined", "hallucination", "halluc_score", "bfactor", "plddt"
        ]
        final_cols = [col for col in output_cols if col in df.columns]
        
        return JSONResponse(df[final_cols].to_dict("records"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Keep api reference for backward compatibility
api = app
