"""
DisProt disorder prediction endpoint
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import the physics extractor
from models.enhanced_ewcl_af import compute_curvature_features

router = APIRouter()

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

def load_model_safely(model_path):
    """Safely load model with error handling"""
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded DisProt model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load DisProt model from {model_path}: {e}")
        return None

# Try to load models
DISPROT_MODEL = load_model_safely(MODEL_DIR / "xgb_disprot_model.pkl")
DISPROT_HALLUC_MODEL = load_model_safely(MODEL_DIR / "hallucination_detector.pkl")

@router.post("/disprot-predict")
async def predict_disprot(file: UploadFile = File(...)):
    """
    DisProt disorder prediction endpoint
    Uses physics-based features with optional ML enhancement
    """
    try:
        print(f"📥 Received file for DisProt prediction: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Extract features using physics extractor
            rows = compute_curvature_features(tmp_path)
            df = pd.DataFrame(rows)
            
            if df.empty:
                raise HTTPException(status_code=400, detail="No features extracted from PDB")
            
            # Prepare basic result structure
            result = []
            for i, row in df.iterrows():
                # Calculate reverse collapse likelihood
                cl = row.get("cl", 0.0)
                rev_cl = 1.0 - cl
                
                # Use hydro_entropy as entropy proxy
                entropy = row.get("hydro_entropy", 0.0)
                
                # Use bfactor_curv as curvature
                curvature = row.get("bfactor_curv", 0.0)
                
                # Physics-based disorder prediction
                # High entropy + low collapse likelihood + high curvature = likely disordered
                physics_disorder_prob = (entropy + rev_cl + abs(curvature)) / 3.0
                physics_disorder_prob = min(max(physics_disorder_prob, 0.0), 1.0)  # Clamp to [0,1]
                
                # If models are available, use them
                disprot_prob = physics_disorder_prob
                halluc_score = 0.0
                
                if DISPROT_MODEL is not None:
                    try:
                        # Prepare features for model prediction
                        import numpy as np
                        features = np.array([[rev_cl, entropy, curvature]])
                        
                        # XGBoost prediction
                        model_prob = DISPROT_MODEL.predict_proba(features)[0, 1]
                        disprot_prob = model_prob
                        
                        # Hallucination check
                        if DISPROT_HALLUC_MODEL is not None:
                            halluc_prob = DISPROT_HALLUC_MODEL.predict_proba(features)[0, 1]
                            halluc_score = halluc_prob
                        
                        print(f"🔬 Using ML models for prediction")
                        
                    except Exception as e:
                        print(f"⚠️ ML model prediction failed, using physics: {e}")
                        disprot_prob = physics_disorder_prob
                else:
                    print("📊 Using physics-based prediction (no ML models)")
                
                result.append({
                    "chain": str(row.get("chain", "A")),
                    "position": int(row.get("residue_id", i + 1)),
                    "aa": str(row.get("aa", "")),
                    "rev_cl": float(rev_cl),
                    "entropy": float(entropy),
                    "disprot_prob": float(disprot_prob),
                    "hallucination_score": float(halluc_score)
                })
            
            # Log results
            disorder_count = sum(1 for r in result if r["disprot_prob"] > 0.7)
            model_status = "ML models" if DISPROT_MODEL is not None else "physics-based"
            print(f"✅ DisProt prediction completed using {model_status}: {disorder_count}/{len(result)} disordered residues")
            
            return JSONResponse(content={"results": result})
            
        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ DisProt prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"DisProt prediction failed: {str(e)}")
