"""
EWCL Collapse-Likelihood API with four endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
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
    """Safely load model with detailed error logging"""
    try:
        print(f"📁 Attempting to load {model_name} from: {model_path}")
        print(f"📊 File exists: {model_path.exists()}")
        if model_path.exists():
            print(f"📏 File size: {model_path.stat().st_size} bytes")
        
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded {model_name}")
        return model
    except FileNotFoundError:
        print(f"❌ File not found: {model_path}")
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
    Requires cl_pred to be already calculated.
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

        # Only use features that exist in df
        available_features = [f for f in expected_features if f in df.columns]
        if not available_features:
            raise ValueError("No expected features found in input DataFrame.")
        
        print(f"📊 Using features for hallucination: {available_features}")
        if len(available_features) != len(expected_features):
            missing = set(expected_features) - set(available_features)
            print(f"⚠️ Missing features: {missing}")

        X = df[available_features]
        df["hallucination"] = HALLUC_MODEL.predict(X)
        df["halluc_score"]  = HALLUC_MODEL.predict_proba(X)[:, 1]
        return df

    except Exception as e:
        raise RuntimeError(f"❌ add_hallucination_prediction failed: {e}")

# ───────────────────────────────────────────
# 4)  FastAPI app
# ───────────────────────────────────────────
api = FastAPI(
    title="EWCL Collapse-Likelihood API",
    version="2025.0.1",
    description="Physics-based + ML refined EWCL with hallucination flags",
)

# CORS middleware
api.add_middleware(
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

@api.get("/")
def health_check():
    return {
        "status": "EWCL API v2025.0.1", 
        "endpoints": {
            "raw-physics": "Physics-only EWCL (no ML)",
            "analyze-ewcl": "Physics + main regressor",
            "refined-ewcl": "High-confidence refiner",
            "detect-hallucination": "Hallucination detection",
            "analyze/full": "Full pipeline analysis"
        },
        "models_loaded": {
            "regressor": REGRESSOR is not None,
            "high_model": HIGH_MODEL is not None,
            "high_scaler": HIGH_SCALER is not None,
            "halluc_model": HALLUC_MODEL is not None
        }
    }

@api.get("/health")
def health():
    model_files = []
    if MODEL_DIR.exists():
        model_files = [f.name for f in MODEL_DIR.glob("*")]
    
    return {
        "status": "ok",
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

# ─────────────  NEW ALL-IN-ONE ENDPOINT  ─────────────
@api.post("/analyze/full")
async def analyze_full(pdb: UploadFile = File(...)):
    """
    Run full pipeline: physics, regressor, refiner, and hallucination detection.
    """
    try:
        df = run_physics(await pdb.read())
        df = add_main_prediction(df)
        df = add_refined_prediction(df)
        df = add_hallucination_prediction(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    output_cols = [
        "chain", "position", "aa", "cl", "cl_pred", 
        "cl_refined", "hallucination", "halluc_score"
    ]
    final_cols = [col for col in output_cols if col in df.columns]
    
    return JSONResponse(df[final_cols].to_dict("records"))

# ─────────────  ENDPOINT 1  ─────────────
@api.post("/raw-physics/")
async def raw_physics(pdb: UploadFile = File(...)):
    """
    Return physics-only EWCL (`cl`) with chain, position, and aa.
    """
    try:
        df = run_physics(await pdb.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(df[["chain", "position", "aa", "cl"]].to_dict("records"))

# ─────────────  ENDPOINT 2  ─────────────
@api.post("/analyze-ewcl/")
async def analyze_ewcl(pdb: UploadFile = File(...)):
    """
    Physics + main regressor (`cl_pred`).
    """
    try:
        df = run_physics(await pdb.read())
        df = add_main_prediction(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return JSONResponse(
        df[["chain", "position", "aa", "cl", "cl_pred"]].to_dict("records")
    )

# ─────────────  ENDPOINT 3  ─────────────
@api.post("/refined-ewcl/")
async def refined_ewcl(pdb: UploadFile = File(...)):
    """
    Extra-trust refiner (`cl_refined`) using the high-correlation model.
    """
    try:
        df = run_physics(await pdb.read())
        df = add_refined_prediction(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return JSONResponse(
        df[["chain", "position", "aa", "cl", "cl_refined"]].to_dict("records")
    )

# ─────────────  ENDPOINT 4  ─────────────
@api.post("/detect-hallucination/")
async def detect_hallucination(pdb: UploadFile = File(...)):
    """
    Flags hallucinated residues from combined physics + ML features.
    """
    try:
        df = run_physics(await pdb.read())
        df = add_hallucination_prediction(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(
        df[["chain", "position", "aa", "cl", "cl_pred",
            "hallucination", "halluc_score"]].to_dict("records")
    )
