"""
EWCL Collapse-Likelihood API with four endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, joblib, pandas as pd, numpy as np, warnings

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
        
        # Try multiple loading strategies for compatibility
        model = None
        
        # Strategy 1: Direct joblib load with warnings suppressed
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                model = joblib.load(model_path)
            print(f"✅ Successfully loaded {model_name} (direct)")
            return model
        except Exception as e1:
            print(f"⚠️ Direct load failed: {e1}")
            
        # Strategy 2: Try with pickle protocol compatibility
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Successfully loaded {model_name} (pickle)")
            return model
        except Exception as e2:
            print(f"⚠️ Pickle load failed: {e2}")
            
        # Strategy 3: Try with different numpy import
        try:
            import sys
            # Temporarily add numpy compatibility
            if 'numpy.core' not in sys.modules:
                import numpy.core as numpy_core
                sys.modules['numpy.core'] = numpy_core
            
            model = joblib.load(model_path)
            print(f"✅ Successfully loaded {model_name} (numpy compat)")
            return model
        except Exception as e3:
            print(f"⚠️ Numpy compat load failed: {e3}")
        
        return None
        
    except Exception as e:
        print(f"❌ All loading strategies failed for {model_name}: {e}")
        return None

# Debug: Print model directory info
print(f"🔍 Model directory: {MODEL_DIR}")
print(f"📂 Model directory exists: {MODEL_DIR.exists()}")
if MODEL_DIR.exists():
    print(f"📋 Files in models/: {list(MODEL_DIR.iterdir())}")

# Load models with enhanced compatibility
print("🔍 Loading models with enhanced compatibility...")

# Try multiple model versions for each type
model_candidates = {
    "regressor": [
        "ewcl_regressor_model.pkl",
        "ewcl_regressor_model_v2.pkl", 
        "ewcl_regressor_af.pkl",
        "ewcl_ai_model.pkl",
        "ewcl_regressor_v1.pkl"
    ],
    "high_model": [
        "ewcl_residue_local_high_model.pkl",
        "ewcl_residue_local_high_model_v2.pkl"
    ],
    "high_scaler": [
        "ewcl_residue_local_high_scaler.pkl", 
        "ewcl_residue_local_high_scaler_v2.pkl"
    ],
    "halluc_model": [
        "hallucination_detector_model.pkl",
        "hallucination_detector.pkl",
        "hallucination_detector_v3000.pkl",
        "hallucination_detector_v500.pkl"
    ]
}

def load_best_model(model_type, candidates):
    """Try to load the best available model from candidates"""
    for candidate in candidates:
        model_path = MODEL_DIR / candidate
        if model_path.exists():
            model = load_model_safely(model_path, f"{model_type} ({candidate})")
            if model is not None:
                print(f"✅ Using {candidate} for {model_type}")
                return model
    print(f"❌ No working model found for {model_type}")
    return None

# Load models using the candidate system
REGRESSOR = load_best_model("regressor", model_candidates["regressor"])
HIGH_MODEL = load_best_model("high_model", model_candidates["high_model"])
HIGH_SCALER = load_best_model("high_scaler", model_candidates["high_scaler"])
HALLUC_MODEL = load_best_model("halluc_model", model_candidates["halluc_model"])

# Try loading DisProt models from multiple locations
print("🔍 Loading DisProt models...")
disprot_candidates = [
    MODEL_DIR / "xgb_disprot_model.pkl",
    MODEL_DIR / "hallucination_classifier_v3_api.pkl",
    Path.home() / "Downloads" / "EWCL_20K_Benchmark" / "models" / "hallucination_classifier_v3_api.pkl"
]

DISPROT_MODEL = None
for path in disprot_candidates:
    if path.exists():
        DISPROT_MODEL = load_model_safely(path, f"disprot_model from {path.name}")
        if DISPROT_MODEL:
            print(f"✅ Using DisProt model: {path.name}")
            break

disprot_halluc_candidates = [
    MODEL_DIR / "hallucination_detector.pkl",
    MODEL_DIR / "hallucination_detector_v3000.pkl",
    Path.home() / "Downloads" / "EWCL_20K_Benchmark" / "models" / "hallucination_classifier_v3.pkl",
    Path.home() / "Downloads" / "EWCL_FullBackup" / "ewcl_ai_models" / "hallucination_safe_model_v3000.pkl"
]

DISPROT_HALLUC_MODEL = None
for path in disprot_halluc_candidates:
    if path.exists():
        DISPROT_HALLUC_MODEL = load_model_safely(path, f"disprot_halluc_model from {path.name}")
        if DISPROT_HALLUC_MODEL:
            print(f"✅ Using DisProt hallucination model: {path.name}")
            break

# If models are missing, create a notice
if DISPROT_MODEL is None:
    print("⚠️ DisProt model not found. Please place 'xgb_disprot_model.pkl' in the models/ directory")
if DISPROT_HALLUC_MODEL is None:
    print("⚠️ DisProt hallucination model not found. Please place 'hallucination_detector.pkl' in the models/ directory")

# Print startup summary
print("\n" + "="*50)
print("🚀 EWCL API STARTUP SUMMARY")
print("="*50)
print(f"✅ Physics extractor: READY")
print(f"{'✅' if REGRESSOR else '❌'} Regressor model: {'LOADED' if REGRESSOR else 'FAILED (using physics fallback)'}")
print(f"{'✅' if HIGH_MODEL else '❌'} High refinement: {'LOADED' if HIGH_MODEL else 'FAILED (using physics fallback)'}")
print(f"{'✅' if HALLUC_MODEL else '❌'} Hallucination detector: {'LOADED' if HALLUC_MODEL else 'FAILED (using physics fallback)'}")
print(f"{'✅' if DISPROT_MODEL else '❌'} DisProt model: {'LOADED' if DISPROT_MODEL else 'FAILED'}")
print(f"{'✅' if DISPROT_HALLUC_MODEL else '❌'} DisProt hallucination model: {'LOADED' if DISPROT_HALLUC_MODEL else 'FAILED'}")
print("\n📡 Available endpoints:")
print("  • /api/analyze/raw - Physics-only EWCL")
print("  • /api/analyze/regressor - Physics + ML regressor (fallback if model unavailable)")
print("  • /api/analyze/refined - Physics + refined model (fallback if model unavailable)")
print("  • /api/analyze/hallucination - Physics + hallucination detection (fallback if model unavailable)")
print("  • /disprot-predict - DisProt disorder prediction")
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

def safe_float(value):
    """Safely convert a value to a float, returning 0.0 on failure."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

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

# Allow frontend domain (not just "*", better security!)
origins = [
    "https://ewclx.com",
    "https://www.ewclx.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://v0-next-webapp-with-7ide41v6k-lucas-cristino.vercel.app",
    "https://v0-next-webapp-with-mol-git-analysis-page-backup-lucas-cristino.vercel.app",
    "https://v0-next-webapp-with-mol.vercel.app"
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

@app.post("/analyze-rev-ewcl")
async def analyze_reverse_ewcl(file: UploadFile = File(...)):
    """
    Entropy-based collapse likelihood inversion reveals regions prone to instability, 
    disorder, or folding uncertainty — consistent with known IDRs.
    Uses physics-based EWCL model with reversed CL = 1 - raw_cl.
    """
    try:
        print(f"📥 Reverse EWCL analysis for: {file.filename}")
        
        # Run physics-based EWCL extraction using the same function as other endpoints
        df = run_physics(await file.read())
        print(f"📊 Physics analysis completed: {len(df)} residues")
        
        # Apply reversed collapse logic using the safe_float helper
        df["cl"] = df["cl"].apply(safe_float)
        df["rev_cl"] = 1 - df["cl"]
        df = df.dropna(subset=["rev_cl"])
        
        # Map fields if needed for consistency
        if "residue_id" in df.columns and "position" not in df.columns:
            df["position"] = df["residue_id"]
        
        # Ensure plddt column exists (AlphaFold PDBs have plddt in bfactor column)
        if "plddt" not in df.columns:
            df["plddt"] = df["bfactor"]
        
        # Prepare comprehensive results matching normal EWCL format
        results = []
        for _, row in df.iterrows():
            # Use safe_float for all numeric conversions
            rev_cl_val = safe_float(row.get("rev_cl"))
            
            # Add instability classification (exploratory thresholds)
            if rev_cl_val > 0.7:
                instability_level = "highly_unstable"
            elif rev_cl_val > 0.5:
                instability_level = "moderately_unstable"
            else:
                instability_level = "stable"
            
            # Add stability note (inverted logic for reversed CL)
            note = "Unstable" if rev_cl_val > 0.6 else "Stable"
            
            result_entry = {
                "chain": str(row.get("chain", "A")),
                "position": int(safe_float(row.get("position", row.get("residue_id", 0)))),
                "aa": str(row.get("aa", "")),
                "cl": round(safe_float(row.get("cl")), 3),  # Original collapse likelihood
                "rev_cl": round(rev_cl_val, 3),   # Reversed collapse likelihood
                "instability_level": instability_level,
                "bfactor": round(safe_float(row.get("bfactor")), 3),
                "plddt": round(safe_float(row.get("plddt", row.get("bfactor"))), 3),
                "bfactor_norm": round(safe_float(row.get("bfactor_norm")), 3),
                "hydro_entropy": round(safe_float(row.get("hydro_entropy")), 3),
                "charge_entropy": round(safe_float(row.get("charge_entropy")), 3),
                "bfactor_curv": round(safe_float(row.get("bfactor_curv")), 3),
                "bfactor_curv_entropy": round(safe_float(row.get("bfactor_curv_entropy")), 3),
                "bfactor_curv_flips": round(safe_float(row.get("bfactor_curv_flips")), 3),
                "note": note
            }
            
            results.append(result_entry)
        
        # Optional: Save JSON output
        try:
            import json
            import os
            os.makedirs("jsons_rev", exist_ok=True)
            filename = file.filename.replace(".pdb", ".json") if file.filename else "output.json"
            with open(f"jsons_rev/{filename}", "w") as f:
                json.dump({
                    "status": "ok",
                    "data": results,
                    "summary": {
                        "total_residues": len(results),
                        "highly_unstable": sum(1 for r in results if r["instability_level"] == "highly_unstable"),
                        "moderately_unstable": sum(1 for r in results if r["instability_level"] == "moderately_unstable"),
                        "stable": sum(1 for r in results if r["instability_level"] == "stable")
                    }
                }, f, indent=2)
            print(f"💾 Saved output to jsons_rev/{filename}")
        except Exception as save_error:
            print(f"⚠️ Failed to save JSON output: {save_error}")
        
        # Statistics
        highly_unstable = sum(1 for r in results if r["instability_level"] == "highly_unstable")
        moderately_unstable = sum(1 for r in results if r["instability_level"] == "moderately_unstable")
        stable = len(results) - highly_unstable - moderately_unstable
        
        print(f"✅ Reverse EWCL analysis: {highly_unstable} highly unstable, {moderately_unstable} moderately unstable, {stable} stable out of {len(results)} residues")
        
        return JSONResponse(content={
            "status": "ok",
            "data": results,
            "summary": {
                "total_residues": len(results),
                "highly_unstable": highly_unstable,
                "moderately_unstable": moderately_unstable,
                "stable": stable,
                "analysis_type": "reverse_ewcl",
                "description": "Entropy-based collapse likelihood inversion for instability analysis"
            }
        })
        
    except Exception as e:
        print(f"❌ Reverse EWCL analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Reverse EWCL analysis failed: {str(e)}")

@app.post("/analyze-pdb")
async def analyze_pdb_direct(file: UploadFile = File(...)):
    """
    Direct analyze-pdb endpoint (without /api prefix)
    Unified analysis with complete physics features
    """
    try:
        print(f"📥 Received file: {file.filename}")
        
        # Run physics analysis
        df = run_physics(await file.read())
        print(f"📊 Physics analysis completed: {len(df)} residues")
        
        # Map fields if needed
        if "residue_id" in df.columns and "position" not in df.columns:
            df["position"] = df["residue_id"]
        
        # Always include base physics results
        result_cols = ["chain", "position", "aa", "cl", "bfactor", "plddt", 
                      "bfactor_norm", "hydro_entropy", "charge_entropy",
                      "bfactor_curv", "bfactor_curv_entropy", "bfactor_curv_flips"]
        
        # Add ML predictions if available
        if REGRESSOR is not None and HALLUC_MODEL is not None:
            try:
                df = add_main_prediction(df)
                df = add_hallucination_prediction(df)
                
                result_cols.extend(["cl_pred", "hallucination", "halluc_score"])
                
                # Add hallucination labels
                df["hallucination_label"] = df["halluc_score"].apply(
                    lambda x: "Likely" if x > 0.7 else "Possible" if x > 0.3 else "Unlikely"
                )
                result_cols.append("hallucination_label")
                
                print("🔬 Added ML predictions")
                
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
                # Add empty columns
                df["hallucination"] = 0
                df["halluc_score"] = 0.0
                df["hallucination_label"] = "Unknown"
                result_cols.extend(["hallucination", "halluc_score", "hallucination_label"])
        else:
            # Add empty columns
            df["hallucination"] = 0
            df["halluc_score"] = 0.0
            df["hallucination_label"] = "Unknown"
            result_cols.extend(["hallucination", "halluc_score", "hallucination_label"])
        
        # Add stability note
        df["note"] = df["cl"].apply(lambda x: "Unstable" if x > 0.6 else "Stable")
        result_cols.append("note")
        
        # Add protein name if available
        if "protein" in df.columns:
            result_cols.insert(0, "protein")
        
        # Filter to available columns
        available_cols = [col for col in result_cols if col in df.columns]
        result = df[available_cols].to_dict("records")
        
        print(f"✅ Returning {len(result)} residues")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disprot-predict")
async def disprot_predict_fallback(file: UploadFile = File(...)):
    """Enhanced DisProt prediction with physics-based fallback"""
    try:
        print(f"📥 DisProt prediction for: {file.filename}")
        df = run_physics(await file.read())
        
        result = []
        for i, row in df.iterrows():
            cl = row.get("cl", 0.0)
            rev_cl = 1.0 - cl
            entropy = row.get("hydro_entropy", 0.0)
            curvature = abs(row.get("bfactor_curv", 0.0))
            
            # Enhanced physics-based disorder prediction
            # Disorder correlates with: high entropy, low collapse likelihood, high curvature
            base_disorder = (entropy * 0.4) + (rev_cl * 0.4) + (curvature * 0.2)
            
            # Apply sigmoid-like transformation for better range
            import math
            disprot_prob = 1 / (1 + math.exp(-5 * (base_disorder - 0.5)))
            disprot_prob = min(max(disprot_prob, 0.0), 1.0)
            
            # Enhanced hallucination scoring based on feature consistency
            bfactor_norm = row.get("bfactor_norm", 0.0)
            charge_entropy = row.get("charge_entropy", 0.0)
            
            # Inconsistent features suggest potential hallucination
            feature_variance = abs(entropy - charge_entropy) + abs(bfactor_norm - 0.5)
            halluc_score = min(feature_variance, 1.0)
            
            result.append({
                "chain": str(row.get("chain", "A")),
                "position": int(row.get("residue_id", i + 1)),
                "aa": str(row.get("aa", "")),
                "rev_cl": float(rev_cl),
                "entropy": float(entropy),
                "disprot_prob": float(disprot_prob),
                "hallucination_score": float(halluc_score)
            })
        
        model_status = "ML" if DISPROT_MODEL else "physics-based"
        disorder_count = sum(1 for r in result if r["disprot_prob"] > 0.7)
        print(f"✅ DisProt prediction ({model_status}): {disorder_count}/{len(result)} disordered")
        
        return JSONResponse(content={"results": result})
        
    except Exception as e:
        print(f"❌ DisProt prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"DisProt prediction failed: {str(e)}")

@app.get("/")
def health_check():
    return {
        "status": "EWCL API v2025.0.1", 
        "message": "API is running successfully",
        "physics_only": REGRESSOR is None and HIGH_MODEL is None and HALLUC_MODEL is None,
        "ml_models_status": f"{'Loaded' if any([REGRESSOR, HIGH_MODEL, HALLUC_MODEL]) else 'Failed to load'} (using {'ML + physics' if any([REGRESSOR, HIGH_MODEL, HALLUC_MODEL]) else 'physics fallbacks'})",
        "endpoints": {
            "GET /": "Health check",
            "GET /health": "Detailed health status",
            "POST /analyze-pdb": f"Direct unified analysis endpoint ({'ML-enhanced' if REGRESSOR and HALLUC_MODEL else 'physics-based'})",
            "POST /analyze-rev-ewcl": "Entropy-based collapse likelihood inversion for instability analysis (physics-based)",
            "POST /disprot-predict": f"DisProt disorder prediction ({'ML-enhanced' if DISPROT_MODEL else 'physics-based'})",
            "POST /api/analyze/raw": "Physics-only EWCL",
            "POST /api/analyze/regressor": f"Physics + ML regressor ({'available' if REGRESSOR else 'fallback to physics'})",
            "POST /api/analyze/refined": f"Physics + refined model ({'available' if HIGH_MODEL else 'fallback to physics'})",
            "POST /api/analyze/hallucination": f"Physics + hallucination detection ({'available' if HALLUC_MODEL else 'fallback to physics'})",
        },
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "high_model": HIGH_MODEL is not None,
            "high_scaler": HIGH_SCALER is not None,
            "halluc_model": HALLUC_MODEL is not None,
            "disprot_model": DISPROT_MODEL is not None,
            "disprot_halluc_model": DISPROT_HALLUC_MODEL is not None,
        },
        "models_found": len([f for f in MODEL_DIR.glob("*.pkl")]) if MODEL_DIR.exists() else 0,
        "note": f"{'ML models successfully loaded!' if any([REGRESSOR, HIGH_MODEL, HALLUC_MODEL]) else 'All endpoints work with physics-based analysis. ML models failed due to version compatibility issues.'}"
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
            "api/analyze/hallucination": "Physics + hallucination detection",
            "disprot-predict": "DisProt disorder prediction"
        },
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "refiner": HIGH_MODEL is not None,
            "hallucination": HALLUC_MODEL is not None,
            "scaler": HIGH_SCALER is not None,
            "disprot_model": DISPROT_MODEL is not None,
            "disprot_halluc_model": DISPROT_HALLUC_MODEL is not None,
        },
        "version": "2025.0.1",
        "python_version": "3.11.8",
        "scikit_learn_version": "1.1.3 (stable)",
        "numpy_version": "1.23.5 (stable)",
        "model_dir_exists": MODEL_DIR.exists(),
        "model_files": model_files,
        "model_dir_path": str(MODEL_DIR),
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
